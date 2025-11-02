# bot_runner.py
import os
import time
import threading
import traceback
import requests
import logging
from datetime import datetime, timedelta
import telebot
from telebot import types

# ----- Branding and global constants -----
BRAND_TAG = "\n\n— <b>Destiny Trading Empire Bot 💎</b>"

# Config from env (with sane defaults)
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT").split(",")
# scan all timeframes per your request:
SCAN_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
SIGNAL_INTERVAL_DEFAULT = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "1"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "100.0"))
# Auto-send parameters you confirmed:
AUTO_CONFIDENCE_THRESHOLD = 0.90   # 90%
AUTO_SEND_ONLY_ADMIN = True        # send to admin only

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

# ----- Logging & storage init -----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Import your project modules (must exist)
# They should provide the functions used below.
try:
    from market_providers import fetch_trending_pairs_branded, fetch_klines_multi, get_session, fetch_trending_pairs_text
except Exception:
    # fallback placeholders if not present (so import error surfaces later more gracefully)
    fetch_trending_pairs_branded = None
    fetch_klines_multi = None
    get_session = None
    fetch_trending_pairs_text = None

try:
    from image_utils import build_signal_image, safe_send_with_image
except Exception:
    build_signal_image = None
    safe_send_with_image = None

try:
    from signal_engine import generate_signal
except Exception:
    generate_signal = None

try:
    from storage import ensure_storage, load_data, save_data, record_pnl_screenshot
except Exception:
    ensure_storage = None
    load_data = None
    save_data = None
    record_pnl_screenshot = None

try:
    from ai_client import ai_analysis_text
except Exception:
    ai_analysis_text = None

try:
    from pro_features import top_gainers_pairs, fear_and_greed_index, futures_leverage_suggestion
except Exception:
    top_gainers_pairs = None
    fear_and_greed_index = None
    futures_leverage_suggestion = None

# Ensure storage exists (if module present)
try:
    if ensure_storage:
        ensure_storage()
except Exception:
    logging.warning("Could not run ensure_storage(): module missing or raised error.")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_last_signal_time = {}  # dict mapping (symbol, interval) -> datetime of last auto-send
_scanner_thread = None
_scanner_stop_event = threading.Event()

# ----- helpers -----
def _append_brand(text: str) -> str:
    if BRAND_TAG.strip() not in text:
        return text + BRAND_TAG
    return text

def stop_existing_bot_instances():
    """Try to clear pending getUpdates sessions to reduce 409 conflicts."""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset=-1"
        requests.get(url, timeout=5)
        logging.info("[BOT] Attempted to stop other bot sessions (getUpdates offset -1).")
    except Exception as e:
        logging.warning(f"[BOT] Could not call Telegram getUpdates clear: {e}")

def can_send_signal(symbol: str, interval: str) -> bool:
    """Respect cooldown per symbol+interval (auto-sends)."""
    key = f"{symbol}|{interval}"
    last = _last_signal_time.get(key)
    if not last:
        return True
    return (datetime.utcnow() - last).total_seconds() > COOLDOWN_MIN * 60

def mark_signal_sent(symbol: str, interval: str):
    key = f"{symbol}|{interval}"
    _last_signal_time[key] = datetime.utcnow()

def compute_risk_and_size(entry: float, sl: float, balance: float, risk_percent: float):
    risk_amount = (balance * risk_percent) / 100.0
    diff = abs(entry - sl)
    if diff <= 1e-12:
        return round(risk_amount, 8), 0.0
    pos_size = risk_amount / diff
    return round(risk_amount, 8), round(pos_size, 8)

def _safe_generate_signal(symbol: str, interval: str):
    """Call your signal engine, but catch exceptions and normalize result."""
    if not generate_signal:
        return {"error": "signal_engine.generate_signal not available"}
    try:
        return generate_signal(symbol, interval)
    except Exception as e:
        logging.error("generate_signal error for %s %s: %s", symbol, interval, e)
        traceback.print_exc()
        return {"error": str(e)}

# ----- recording & messaging -----
def record_signal_and_send(sig: dict, chat_id=None, user_id=None, auto=False):
    """Record a signal in storage and send it (image + caption)."""
    # storage
    try:
        d = load_data() if load_data else {}
    except Exception:
        d = {}

    sig_id = f"S{int(time.time())}"
    balance = d.get("challenge", {}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START
    # risk and pos
    risk_amt, pos_size = compute_risk_and_size(sig.get("entry") or sig.get("entry", 0),
                                               sig.get("sl") or sig.get("sl", 0),
                                               balance, RISK_PERCENT)

    rec = {
        "id": sig_id,
        "signal": sig,
        "time": datetime.utcnow().isoformat(),
        "risk_amt": risk_amt,
        "pos_size": pos_size,
        "user": user_id or ADMIN_ID,
        "auto": bool(auto)
    }

    try:
        if isinstance(d, dict):
            d.setdefault("signals", []).append(rec)
            d.setdefault("stats", {})
            d["stats"]["total_signals"] = d["stats"].get("total_signals", 0) + 1
            if save_data:
                save_data(d)
    except Exception:
        logging.exception("Failed to save signal record")

    # Build caption
    try:
        confidence_pct = int(sig.get("confidence", 0) * 100)
    except Exception:
        confidence_pct = 0

    caption = (
        f"🔥 <b>Destiny Trading Empire — Signal</b>\n"
        f"ID: {sig_id}\nPair: {sig.get('symbol')} | TF: {sig.get('interval')}\n"
        f"Signal: <b>{sig.get('signal')}</b>\nEntry: {sig.get('entry')} | SL: {sig.get('sl')} | TP1: {sig.get('tp1')}\n"
        f"Confidence: {confidence_pct}% | Risk (USD): {risk_amt}\n"
        f"Reasons: {', '.join(sig.get('reasons', []) or ['None'])}\n"
    )
    caption = _append_brand(caption)

    # Image creation
    img = None
    try:
        if build_signal_image:
            img = build_signal_image(sig)
    except Exception:
        logging.exception("build_signal_image failed")

    # Keyboard for message
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
    # allow quick share inline
    kb.add(types.InlineKeyboardButton("🔁 Share", switch_inline_query=f"{sig.get('symbol')}"))

    # send (use safe_send_with_image if available)
    try:
        if safe_send_with_image:
            safe_send_with_image(bot, chat_id or ADMIN_ID, caption, img, kb)
        else:
            # fallback
            if img:
                bot.send_photo(chat_id or ADMIN_ID, img, caption=caption, reply_markup=kb)
            else:
                bot.send_message(chat_id or ADMIN_ID, caption, reply_markup=kb)
    except Exception:
        logging.exception("Failed to send signal message")

    return sig_id

# ----- keyboard UI -----
def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📈 Get Signals", callback_data="get_signal"),
        types.InlineKeyboardButton("🔎 Scan Top 4", callback_data="scan_top4")
    )
    kb.add(
        types.InlineKeyboardButton("⚙️ Bot Status", callback_data="bot_status"),
        types.InlineKeyboardButton("🚀 Trending Pairs", callback_data="trending")
    )
    kb.add(
        types.InlineKeyboardButton("📰 Market News", callback_data="market_news"),
        types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status")
    )
    kb.add(
        types.InlineKeyboardButton("📸 Upload PnL", callback_data="pnl_upload"),
        types.InlineKeyboardButton("🧾 History", callback_data="history")
    )
    kb.add(
        types.InlineKeyboardButton("🤖 AI Market Brief", callback_data="ask_ai"),
        types.InlineKeyboardButton("🔄 Refresh Bot", callback_data="refresh_bot")
    )
    kb.add(
        types.InlineKeyboardButton("▶️ Start Auto Scanner", callback_data="start_auto_brief"),
        types.InlineKeyboardButton("⏹ Stop Auto Scanner", callback_data="stop_auto_brief")
    )
    return kb

# ----- Telegram handlers -----
@bot.message_handler(commands=['start', 'menu'])
def cmd_start(msg):
    try:
        bot.send_message(msg.chat.id,
                         _append_brand("👋 Welcome Boss Destiny!\n\nThis is your Trading Empire control panel."),
                         reply_markup=main_keyboard())
    except Exception:
        logging.exception("cmd_start failed")

@bot.message_handler(content_types=['photo'])
def photo_handler(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(fi.file_path)
        if record_pnl_screenshot:
            fname = record_pnl_screenshot(data, datetime.utcnow().strftime("%Y%m%d_%H%M%S"), message.from_user.id, message.caption)
        bot.reply_to(message, _append_brand("Saved screenshot. Reply with `#link <signal_id> TP1` or `#link <signal_id> SL`"))
    except Exception:
        logging.exception("photo_handler failed")
        bot.reply_to(message, _append_brand("Failed to save screenshot."))

@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_handler(message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 3:
            bot.reply_to(message, _append_brand("Usage: #link <signal_id> TP1 or SL"))
            return
        sig_id, tag = parts[1], parts[2].upper()
        d = load_data() if load_data else {}
        pnl_item = next((p for p in reversed(d.get("pnl",[])) if p.get("linked") is None and p["from"] == message.from_user.id), None) if isinstance(d, dict) else None
        if not pnl_item:
            bot.reply_to(message, _append_brand("No unlinked screenshot found."))
            return
        pnl_item["linked"] = {"signal_id": sig_id, "result": tag, "linked_by": message.from_user.id}
        # only admin confirmation updates balance
        if message.from_user.id == ADMIN_ID:
            srec = next((s for s in d.get("signals",[]) if s["id"]==sig_id), None)
            if srec:
                risk = srec.get("risk_amt", 0)
                if tag.startswith("TP"):
                    d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) + risk
                    d["challenge"]["wins"] = d["challenge"].get("wins", 0) + 1
                    d["stats"]["wins"] = d["stats"].get("wins",0) + 1
                elif tag == "SL":
                    d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) - risk
                    d["challenge"]["losses"] = d["challenge"].get("losses",0) + 1
                    d["stats"]["losses"] = d["stats"].get("losses",0) + 1
        if save_data and isinstance(d, dict):
            save_data(d)
        bot.reply_to(message, _append_brand(f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance."))
    except Exception:
        logging.exception("link_handler failed")
        bot.reply_to(message, _append_brand("Failed to link screenshot."))

# Callback actions
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        cid = call.message.chat.id
        data = call.data
        bot.answer_callback_query(call.id)
        # Choose pair -> produce inline keyboard of PAIRS
        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"sig_{p}"))
            bot.send_message(cid, _append_brand("Choose pair to analyze:"), reply_markup=kb)
            return

        # individual pair selected
        if data.startswith("sig_"):
            pair = data.split("_",1)[1]
            bot.send_chat_action(cid, "typing")
            sig = _safe_generate_signal(pair, SIGNAL_INTERVAL_DEFAULT)
            if sig.get("error"):
                bot.send_message(cid, _append_brand(f"Error generating signal: {sig['error']}"))
                return
            # show to the requester
            record_signal_and_send(sig, chat_id=cid, user_id=call.from_user.id, auto=False)
            return

        # quick scan top X
        if data == "scan_top4":
            bot.send_message(cid, _append_brand("🔎 Scanning top pairs across exchanges..."))
            for p in PAIRS[:6]:
                try:
                    if can_send_signal(p, SIGNAL_INTERVAL_DEFAULT):
                        sig = _safe_generate_signal(p, SIGNAL_INTERVAL_DEFAULT)
                        if not sig.get("error") and sig.get("signal") in ("LONG","SHORT"):
                            record_signal_and_send(sig, chat_id=cid)
                except Exception:
                    logging.exception("scan_top4 subtask failed")
            return

        if data == "trending":
            bot.send_message(cid, _append_brand("📡 Fetching multi-exchange trending pairs... please wait."))
            try:
                if fetch_trending_pairs_branded:
                    img_buf, caption = fetch_trending_pairs_branded(top_n=8)
                    if img_buf:
                        safe_send_with_image(bot, cid, _append_brand(caption), img_buf)
                    else:
                        bot.send_message(cid, _append_brand(caption))
                elif fetch_trending_pairs_text:
                    bot.send_message(cid, _append_brand(fetch_trending_pairs_text()))
                else:
                    bot.send_message(cid, _append_brand("Trending feature not available (missing market_providers)."))
            except Exception:
                logging.exception("trending handler failed")
                bot.send_message(cid, _append_brand("Failed to fetch trending pairs."))
            return

        if data == "bot_status":
            bot.send_message(cid, _append_brand("⚙️ Bot is running ✅"))
            return

        if data == "market_news":
            bot.send_message(cid, _append_brand("📰 Market news: feature coming soon"))
            return

        if data == "challenge_status":
            d = load_data() if load_data else {}
            bal = d.get("challenge",{}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START
            wins = d.get("challenge",{}).get("wins",0) if isinstance(d, dict) else 0
            losses = d.get("challenge",{}).get("losses",0) if isinstance(d, dict) else 0
            bot.send_message(cid, _append_brand(f"Balance: ${bal:.2f}\nWins: {wins} Losses: {losses}"))
            return

        if data == "ask_ai":
            bot.send_message(cid, _append_brand("🤖 Ask AI: send a message starting with `AI:` followed by your question."))
            return

        if data == "refresh_bot":
            bot.send_message(cid, _append_brand("🔄 Refreshing bot session..."))
            stop_existing_bot_instances()
            time.sleep(2)
            bot.send_message(cid, _append_brand("✅ Refreshed."))
            return

        if data == "start_auto_brief":
            bot.send_message(cid, _append_brand("▶️ Starting background market scanner (auto-send strong signals)."))
            start_background_scanner()
            return

        if data == "stop_auto_brief":
            bot.send_message(cid, _append_brand("⏹ Stopping background market scanner."))
            stop_background_scanner()
            return

        if data.startswith("ai_"):
            sig_id = data.split("_",1)[1]
            d = load_data() if load_data else {}
            rec = next((s for s in d.get("signals",[]) if s["id"]==sig_id), None) if isinstance(d, dict) else None
            if not rec:
                bot.send_message(cid, _append_brand("Signal not found"))
                return
            prompt = f"Provide trade rationale, risk controls and a recommended leverage for this trade:\n{rec['signal']}"
            ai_text = ai_analysis_text(prompt) if ai_analysis_text else "AI feature not available"
            bot.send_message(cid, _append_brand(f"🤖 AI analysis:\n{ai_text}"))
            return

        bot.send_message(cid, _append_brand("Unknown action"))
    except Exception:
        logging.exception("callback_handler failed")
        try:
            bot.answer_callback_query(call.id, "Handler error")
        except Exception:
            pass

# ----- Background scanner (auto-detect strong signals) -----
def _scanner_loop():
    """
    Runs forever until stop event set. Scans multiple timeframes & pairs.
    When it finds a signal >= AUTO_CONFIDENCE_THRESHOLD, sends only to admin (per your request)
    and marks cooldown for that symbol+interval.
    """
    logging.info("[SCANNER] Background scanner started (scanning multiple TFs).")
    # continuous loop
    while not _scanner_stop_event.is_set():
        try:
            # iterate timeframes and pairs
            for interval in SCAN_INTERVALS:
                if _scanner_stop_event.is_set():
                    break
                for pair in PAIRS:
                    if _scanner_stop_event.is_set():
                        break
                    # check cooldown
                    if not can_send_signal(pair, interval):
                        continue

                    sig = _safe_generate_signal(pair, interval)
                    if sig.get("error"):
                        # skip this pair
                        continue

                    # signal type: prefer LONG/SHORT (your engine returns LONG/SHORT/HOLD)
                    s_type = sig.get("signal")
                    conf = float(sig.get("confidence", 0.0)) if sig.get("confidence") is not None else 0.0

                    # check confidence threshold
                    if s_type in ("LONG", "SHORT") and conf >= AUTO_CONFIDENCE_THRESHOLD:
                        # mark and send only to admin
                        try:
                            mark_signal_sent(pair, interval)
                            # send to admin only
                            target = ADMIN_ID if AUTO_SEND_ONLY_ADMIN and ADMIN_ID else None
                            record_signal_and_send(sig, chat_id=target, user_id=ADMIN_ID, auto=True)
                            logging.info("[SCANNER] Auto-sent strong signal for %s %s conf=%.2f", pair, interval, conf)
                        except Exception:
                            logging.exception("Failed to record/send auto signal")
                    # small sleep between pair checks to avoid API rate issues
                    time.sleep(0.6)
            # after a full pass, brief rest to avoid 100% CPU; allow responsive stopping
            # scan continuously — your "all timeframes" requirement: loop without huge pause
            # small sleep to yield, but scanner is effectively continuous
            time.sleep(2.0)
        except Exception:
            logging.exception("Unhandled error in scanner loop")
            time.sleep(1.0)
    logging.info("[SCANNER] Background scanner stopped.")

def start_background_scanner():
    global _scanner_thread, _scanner_stop_event
    if _scanner_thread and _scanner_thread.is_alive():
        logging.info("[SCANNER] Already running.")
        return
    _scanner_stop_event.clear()
    _scanner_thread = threading.Thread(target=_sca
