# bot_runner.py
import os
import time
import threading
import traceback
import requests
import logging
from datetime import datetime
import telebot
from telebot import types

# ----- Branding and global constants -----
BRAND_TAG = "\n\n— <b>Destiny Trading Empire Bot 💎</b>"

# Config from env (with sane defaults)
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT").split(",")
# scan all timeframes per requirement:
SCAN_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
SIGNAL_INTERVAL_DEFAULT = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "1"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "100.0"))
# Auto-send parameters:
AUTO_CONFIDENCE_THRESHOLD = float(os.getenv("AUTO_CONFIDENCE_THRESHOLD", "0.90"))   # 0.90 = 90%
AUTO_SEND_ONLY_ADMIN = True        # send to admin only (as requested)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

# ----- Logging & storage init -----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Import optional project modules (if missing we fallback gracefully)
try:
    from market_providers import (
        fetch_trending_pairs_branded,
        fetch_klines_multi,
        get_session,
        fetch_trending_pairs_text,
        analyze_pair_multi_timeframes,
        detect_strong_signals,
        generate_branded_signal_image
    )
except Exception:
    fetch_trending_pairs_branded = None
    fetch_klines_multi = None
    get_session = None
    fetch_trending_pairs_text = None
    analyze_pair_multi_timeframes = None
    detect_strong_signals = None
    generate_branded_signal_image = None

try:
    from image_utils import build_signal_image, safe_send_with_image, create_brand_image
except Exception:
    build_signal_image = None
    safe_send_with_image = None
    create_brand_image = None

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
    from pro_features import top_gainers_pairs, fear_and_greed_index, futures_leverage_suggestion, quickchart_price_image, ai_market_brief_text
except Exception:
    top_gainers_pairs = None
    fear_and_greed_index = None
    futures_leverage_suggestion = None
    quickchart_price_image = None
    ai_market_brief_text = None

# Scheduler (for auto-briefs)
try:
    from scheduler import start_scheduler, stop_scheduler
except Exception:
    start_scheduler = None
    stop_scheduler = None

# ensure storage directory/data if module available
if ensure_storage:
    try:
        ensure_storage()
    except Exception:
        logging.exception("ensure_storage failed")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_last_signal_time = {}  # dict mapping (symbol|interval) -> datetime of last auto-send
_scanner_thread = None
_scanner_stop_event = threading.Event()

# ----- helpers -----
def _append_brand(text: str) -> str:
    if BRAND_TAG.strip() not in text:
        return text + BRAND_TAG
    return text

def _send_branded(chat_id, text, lines_for_image=None, reply_markup=None):
    """
    Send branded reply. If create_brand_image and safe_send_with_image available,
    create image and send image + caption. Otherwise fallback to text message.
    """
    text = _append_brand(text)
    try:
        if create_brand_image and safe_send_with_image:
            img = create_brand_image(lines_for_image or [text], title="Destiny Trading Empire Bot 💎")
            safe_send_with_image(bot, chat_id, text, img, reply_markup=reply_markup)
        else:
            bot.send_message(chat_id, text, reply_markup=reply_markup)
    except Exception:
        logging.exception("Failed to send branded message; falling back to plain text")
        try:
            bot.send_message(chat_id, text, reply_markup=reply_markup)
        except Exception:
            logging.exception("Second attempt to send message failed")

def stop_existing_bot_instances():
    """Try clear pending getUpdates sessions to reduce 409 conflicts."""
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
    """
    Primary flow:
      1) Try signal_engine.generate_signal (if present)
      2) If missing/errored/insufficient, fallback to analyze_pair_multi_timeframes from market_providers
      3) Normalize and return a consistent signal dict:
         {
           "symbol": "BTCUSDT", "interval": "1h", "signal": "LONG"/"SHORT"/"HOLD",
           "entry": float, "sl": float, "tp1": float, "confidence": float, "reasons": [...], "raw": {...}
         }
    """
    # 1) try the simple engine first
    if generate_signal:
        try:
            g = generate_signal(symbol, interval)
            if isinstance(g, dict) and not g.get("error"):
                # ensure keys exist
                g.setdefault("symbol", symbol.upper())
                g.setdefault("interval", interval)
                # normalize confidence to 0..1
                try:
                    g["confidence"] = float(g.get("confidence", 0.0))
                except Exception:
                    g["confidence"] = 0.0
                # if engine returned insufficient data, fallback
                reasons = g.get("reasons", []) or []
                if "Insufficient data" in (g.get("error") or "") or "insufficient" in " ".join([str(r).lower() for r in reasons]):
                    # fall through to multi-tf analysis
                    pass
                else:
                    return g
            # if g returned error or was insufficient, we'll fallback below
        except Exception:
            logging.exception("signal_engine.generate_signal crashed; falling back to market_providers")

    # 2) fallback: use analyze_pair_multi_timeframes (multi-TF scoring)
    if analyze_pair_multi_timeframes:
        try:
            analysis = analyze_pair_multi_timeframes(symbol, timeframes=SCAN_INTERVALS, exchange="binance")
            if isinstance(analysis, dict) and not analysis.get("error"):
                combined_score = float(analysis.get("combined_score", 0.0))
                combined_signal = analysis.get("combined_signal", "HOLD")
                # pick 1h info if available; fallback to highest-TF available
                tf_info = analysis.get("analysis", {}).get("1h")
                if not tf_info:
                    # pick any tf that has data
                    for tf, info in analysis.get("analysis", {}).items():
                        if isinstance(info, dict) and "close" in info:
                            tf_info = info
                            break
                if not tf_info:
                    return {"error": "insufficient_data_after_analysis"}

                entry = float(tf_info.get("close", 0.0))
                sl = float(tf_info.get("sl", entry * 0.995))
                tp1 = float(tf_info.get("tp1", entry * 1.005))
                confidence = max(0.0, min(1.0, combined_score))
                # normalize combined_signal to LONG/SHORT/HOLD
                sig_map = {
                    "STRONG_LONG": "LONG",
                    "STRONG_SHORT": "SHORT",
                    "LONG": "LONG",
                    "SHORT": "SHORT",
                    "HOLD": "HOLD"
                }
                sig_norm = sig_map.get(combined_signal, "HOLD")
                reasons = []
                # collect top reasons from each tf (best-effort)
                for tf, info in analysis.get("analysis", {}).items():
                    if isinstance(info, dict):
                        reasons.extend(info.get("reasons", []) or [])
                # build normalized dict
                result = {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "signal": sig_norm,
                    "entry": round(entry, 8),
                    "sl": round(sl, 8),
                    "tp1": round(tp1, 8),
                    "confidence": round(confidence, 4),
                    "reasons": reasons,
                    "raw_analysis": analysis
                }
                return result
        except Exception:
            logging.exception("analyze_pair_multi_timeframes fallback failed")

    # 3) final fallback: try detect_strong_signals for this single pair
    if detect_strong_signals:
        try:
            r = detect_strong_signals(pairs=[symbol], timeframes=SCAN_INTERVALS, exchange="binance", min_confidence=0.65)
            if r and isinstance(r, list) and len(r) > 0:
                it = r[0]
                # pick sl/tp if present
                sl = it.get("sl") or 0.0
                tp1 = it.get("tp1") or 0.0
                img = it.get("image")
                # confidence
                conf = float(it.get("combined_score", 0.0))
                # combined_signal -> LONG/SHORT
                combined_signal = it.get("combined_signal", "HOLD")
                signorm = "LONG" if "LONG" in combined_signal else "SHORT" if "SHORT" in combined_signal else "HOLD"
                entry = it.get("analysis", {}).get("1h", {}).get("close") or it.get("analysis", {}).get(next(iter(it.get("analysis", {}))), {}).get("close", 0.0) or 0.0
                return {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "signal": signorm,
                    "entry": round(float(entry), 8),
                    "sl": round(float(sl or 0.0), 8),
                    "tp1": round(float(tp1 or 0.0), 8),
                    "confidence": round(conf, 4),
                    "reasons": it.get("caption_lines", []),
                    "raw_analysis": it
                }
        except Exception:
            logging.exception("detect_strong_signals fallback failed")

    # if all attempts fail, return a clear error
    return {"error": "insufficient_data_all_sources"}

# ----- recording & messaging -----
def record_signal_and_send(sig: dict, chat_id=None, user_id=None, auto=False):
    """Record a signal in storage and send it (image + caption)."""
    # storage read
    try:
        d = load_data() if load_data else {}
    except Exception:
        d = {}

    sig_id = f"S{int(time.time())}"
    balance = d.get("challenge", {}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START

    # risk and pos
    try:
        entry_val = float(sig.get("entry") or sig.get("entry", 0))
        sl_val = float(sig.get("sl") or sig.get("sl", 0))
    except Exception:
        entry_val = 0.0
        sl_val = 0.0
    risk_amt, pos_size = compute_risk_and_size(entry_val, sl_val, balance, RISK_PERCENT)

    rec = {
        "id": sig_id,
        "signal": sig,
        "time": datetime.utcnow().isoformat(),
        "risk_amt": risk_amt,
        "pos_size": pos_size,
        "user": user_id or ADMIN_ID,
        "auto": bool(auto)
    }

    # save record
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

    # Image creation (prefer build_signal_image, else build from market_providers generate_branded_signal_image)
    img = None
    try:
        if build_signal_image:
            img = build_signal_image(sig)
        elif generate_branded_signal_image and isinstance(sig.get("raw_analysis"), dict):
            img, _ = generate_branded_signal_image({
                "symbol": sig.get("symbol"),
                "analysis": sig.get("raw_analysis", {}),
                "caption_lines": [caption]
            })
    except Exception:
        logging.exception("build signal image failed")

    # Keyboard for message
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🔁 Share", switch_inline_query=f"{sig.get('symbol')}"))

    # send (use safe_send_with_image if available)
    try:
        if safe_send_with_image:
            safe_send_with_image(bot, chat_id or ADMIN_ID, caption, img, kb)
        else:
            if img:
                bot.send_photo(chat_id or ADMIN_ID, img, caption=caption, reply_markup=kb)
            else:
                bot.send_message(chat_id or ADMIN_ID, caption, reply_markup=kb)
    except Exception:
        logging.exception("Failed to send signal message")

    # optionally send a quick AI rationale follow-up if available
    try:
        if ai_analysis_text and sig and not sig.get("error"):
            prompt = f"Provide concise trade rationale for this signal:\n{sig}"
            ai_text = ai_analysis_text(prompt)
            if ai_text:
                follow = _append_brand(f"🤖 AI Rationale:\n{ai_text}")
                _send_branded(chat_id or ADMIN_ID, follow, lines_for_image=[f"AI Rationale — {sig.get('symbol')}"])
    except Exception:
        logging.exception("AI rationale follow-up failed")

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
    kb.add(
        types.InlineKeyboardButton("📣 Start Auto Briefs", callback_data="start_auto_brief_scheduler"),
        types.InlineKeyboardButton("⛔ Stop Auto Briefs", callback_data="stop_auto_brief_scheduler")
    )
    return kb

# ----- Telegram handlers -----
@bot.message_handler(commands=['start', 'menu'])
def cmd_start(msg):
    try:
        text = "👋 Welcome Boss Destiny!\n\nThis is your Trading Empire control panel."
        _send_branded(msg.chat.id, text, lines_for_image=["Welcome — Destiny Trading Empire Bot 💎"], reply_markup=main_keyboard())
    except Exception:
        logging.exception("cmd_start failed")

@bot.message_handler(content_types=['photo'])
def photo_handler(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(fi.file_path)
        if record_pnl_screenshot:
            record_pnl_screenshot(data, datetime.utcnow().strftime("%Y%m%d_%H%M%S"), message.from_user.id, message.caption)
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

# ----- Callback actions -----
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
            _send_branded(cid, "Choose pair to analyze:", lines_for_image=["Choose pair to analyze"], reply_markup=kb)
            return

        # individual pair selected
        if data.startswith("sig_"):
            pair = data.split("_",1)[1]
            bot.send_chat_action(cid, "typing")
            sig = _safe_generate_signal(pair, SIGNAL_INTERVAL_DEFAULT)
            if sig.get("error"):
                _send_branded(cid, f"Error generating signal: {sig['error']}", lines_for_image=[f"{pair} - Error"])
                return
            record_signal_and_send(sig, chat_id=cid, user_id=call.from_user.id, auto=False)
            return

        # quick scan top X
        if data == "scan_top4":
            _send_branded(cid, "🔎 Scanning top pairs across exchanges...")
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
            _send_branded(cid, "📡 Fetching multi-exchange trending pairs... please wait.")
            try:
                if fetch_trending_pairs_branded:
                    img_buf, caption = fetch_trending_pairs_branded(limit=8)
                    if img_buf:
                        safe_send_with_image(bot, cid, _append_brand(caption), img_buf)
                    else:
                        _send_branded(cid, caption)
                elif fetch_trending_pairs_text:
                    _send_branded(cid, fetch_trending_pairs_text())
                else:
                    _send_branded(cid, "Trending feature not available (missing market_providers).")
            except Exception:
                logging.exception("trending handler failed")
                _send_branded(cid, "Failed to fetch trending pairs.")
            return

        if data == "bot_status":
            # provide brief health info and whether scanner is running
            scanner_running = _scanner_thread is not None and _scanner_thread.is_alive()
            msg = f"⚙️ Bot is running ✅\nScanner running: {scanner_running}\nAuto confidence threshold: {AUTO_CONFIDENCE_THRESHOLD*100:.0f}%"
            _send_branded(cid, msg, lines_for_image=["Status"])
            return

        if data == "market_news":
            _send_branded(cid, "📰 Market news: feature coming soon")
            return

        if data == "challenge_status":
            d = load_data() if load_data else {}
            bal = d.get("challenge",{}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START
            wins = d.get("challenge",{}).get("wins",0) if isinstance(d, dict) else 0
            losses = d.get("challenge",{}).get("losses",0) if isinstance(d, dict) else 0
            _send_branded(cid, f"Balance: ${bal:.2f}\nWins: {wins} Losses: {losses}", lines_for_image=["Challenge Status"])
            return

        if data == "ask_ai":
            _send_branded(cid, "🤖 Ask AI: send a message starting with `AI:` followed by your question.")
            return

        if data == "refresh_bot":
            _send_branded(cid, "🔄 Refreshing bot session...")
            stop_existing_bot_instances()
            time.sleep(2)
            _send_branded(cid, "✅ Refreshed.")
            return

        if data == "start_auto_brief":
            _send_branded(cid, "▶️ Starting background market scanner (auto-send strong signals).")
            start_background_scanner()
            return

        if data == "stop_auto_brief":
            _send_branded(cid, "⏹ Stopping background market scanner.")
            stop_background_scanner()
            return

        # Scheduler-based auto briefs (text/AI summaries)
        if data == "start_auto_brief_scheduler":
            if start_scheduler:
                _send_branded(cid, "▶️ Scheduler for auto-briefs enabled. You will receive periodic market briefs.")
                try:
                    start_scheduler(bot)
                except Exception:
                    logging.exception("start_scheduler failed")
            else:
                _send_branded(cid, "Scheduler not available (missing scheduler module).")
            return

        if data == "stop_auto_brief_scheduler":
            if stop_scheduler:
                stop_scheduler()
                _send_branded(cid, "⏹ Scheduler for auto-briefs disabled.")
            else:
                _send_branded(cid, "Scheduler not available (missing scheduler module).")
            return

        if data.startswith("ai_"):
            sig_id = data.split("_",1)[1]
            d = load_data() if load_data else {}
            rec = next((s for s in d.get("signals",[]) if s["id"]==sig_id), None) if isinstance(d, dict) else None
            if not rec:
                _send_branded(cid, "Signal not found")
                return
            prompt = f"Provide trade rationale, risk controls and a recommended leverage for this trade:\n{rec['signal']}"
            ai_text = ai_analysis_text(prompt) if ai_analysis_text else "AI feature not available"
            _send_branded(cid, f"🤖 AI analysis:\n{ai_text}", lines_for_image=["AI Analysis"])
            return

        _send_branded(cid, "Unknown action")
    except Exception:
        logging.exception("callback_handler failed")
        try:
            bot.answer_callback_query(call.id, "Handler error")
        except Exception:
            pass

# ----- Background scanner (auto-detect strong signals) -----
def _scanner_loop():
    """
    Runs until stop event set. Scans multiple timeframes & pairs.
    When it finds a strong signal >= AUTO_CONFIDENCE_THRESHOLD, sends only to admin
    and marks cooldown for that symbol+interval.
    """
    logging.info("[SCANNER] Background scanner started (multi-TF).")
    while not _scanner_stop_event.is_set():
        try:
            for interval in SCAN_INTERVALS:
                if _scanner_stop_event.is_set():
                    break
                for pair in PAIRS:
                    if _scanner_stop_event.is_set():
                        break
                    # respect cooldown per pair+TF
                    if not can_send_signal(pair, interval):
                        continue

                    sig = _safe_generate_signal(pair, interval)
                    if sig.get("error"):
                        continue

                    s_type = sig.get("signal")
                    conf = float(sig.get("confidence", 0.0)) if sig.get("confidence") is not None else 0.0

                    if s_type in ("LONG", "SHORT") and conf >= AUTO_CONFIDENCE_THRESHOLD:
                        try:
                            mark_signal_sent(pair, interval)
                            target = ADMIN_ID if AUTO_SEND_ONLY_ADMIN and ADMIN_ID else None
                            record_signal_and_send(sig, chat_id=target, user_id=ADMIN_ID, auto=True)
                            logging.info("[SCANNER] Auto-sent strong signal for %s %s conf=%.2f", pair, interval, conf)
                        except Exception:
                            logging.exception("Failed to record/send auto signal")
                    # mild throttle to avoid rate limits
                    time.sleep(0.6)
            # small pause between full cycles so the system can breathe
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
    _scanner_thread = threading.Thread(target=_scanner_loop, daemon=True)
    _scanner_thread.start()
    logging.info("[SCANNER] Started background scanner thread.")

def stop_background_scanner():
    global _scanner_thread, _scanner_stop_event
    if not _scanner_thread:
        logging.info("[SCANNER] Not running.")
        return
    _scanner_stop_event.set()
    _scanner_thread.join(timeout=5)
    _scanner_thread = None
    logging.info("[SCANNER] Stop requested and thread joined.")

# ----- Start polling safely (exported) -----
def start_bot_polling():
    stop_existing_bot_instances()
    logging.info("[BOT] Starting polling loop...")
    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=60, skip_pending=True)
        except Exception as e:
            logging.error("[BOT] Polling loop exception: %s", e)
            if "409" in str(e):
                logging.warning("[BOT] 409 Conflict - attempting to stop other sessions and retry")
                stop_existing_bot_instances()
                time.sleep(5)
            else:
                time.sleep(5)
                
