# bot_runner.py
import os
import time
import threading
import traceback
import requests
import logging
import inspect
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
    from market_providers import fetch_trending_pairs_branded, fetch_klines_multi, get_session, fetch_trending_pairs_text
except Exception:
    fetch_trending_pairs_branded = None
    fetch_klines_multi = None
    get_session = None
    fetch_trending_pairs_text = None

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

# ---------------------------
# Compatibility wrappers
# ---------------------------
def _call_generate_signal_flexibly(symbol: str, interval: str):
    """
    Support both signal_engine.generate_signal(symbol, interval)
    and generate_signal(df) signatures. Return dict or {error: ...}
    """
    if not generate_signal:
        return {"error": "signal_engine.generate_signal not available"}

    try:
        sig_spec = inspect.signature(generate_signal)
        params = len(sig_spec.parameters)
    except Exception:
        params = None

    # If function expects 1 positional parameter and user provided df-style earlier,
    # we will attempt to fetch klines df and pass that single arg.
    if params == 1:
        # need a DataFrame; try market_providers.fetch_klines_multi
        if fetch_klines_multi:
            try:
                df = fetch_klines_multi(symbol=symbol, interval=interval, limit=300, exchange="binance")
                if df is None:
                    return {"error": "No kline data available to pass to generate_signal(df)"}
                res = generate_signal(df)
                # If generate_signal returned a string or dict, normalize later
                return {"raw": res, "source_df": df}
            except Exception as e:
                logging.error("Error calling generate_signal(df) for %s %s: %s", symbol, interval, e)
                traceback.print_exc()
                return {"error": str(e)}
        else:
            return {"error": "fetch_klines_multi not available to build df for generate_signal(df)"}
    else:
        # call generate_signal(symbol, interval) directly
        try:
            res = generate_signal(symbol, interval)
            return {"raw": res, "source_df": None}
        except Exception as e:
            logging.error("Error calling generate_signal(symbol, interval) for %s %s: %s", symbol, interval, e)
            traceback.print_exc()
            return {"error": str(e)}

def _normalize_signal(raw_result, symbol: str, interval: str, source_df=None):
    """
    Turn whatever generate_signal returned into a standard dict:
    {
      symbol, interval, timestamp, signal, entry, sl, tp1, confidence, reasons: [...]
    }
    - raw_result may be dict, DataFrame, or string.
    - source_df: if available, used to infer price/entry
    """
    # If the wrapper returned an error
    if isinstance(raw_result, dict) and raw_result.get("error"):
        return {"error": raw_result.get("error")}

    raw = raw_result
    # If the wrapper returned {"raw": ..., "source_df": df} pattern
    if isinstance(raw_result, dict) and "raw" in raw_result:
        raw = raw_result["raw"]
        if not source_df:
            source_df = raw_result.get("source_df")

    # If generate_signal returned a dict with expected structure already
    if isinstance(raw, dict):
        # ensure required fields exist
        res = {
            "symbol": raw.get("symbol", symbol).upper(),
            "interval": raw.get("interval", interval),
            "timestamp": raw.get("timestamp", datetime.utcnow().isoformat()),
            "signal": raw.get("signal", "HOLD"),
            "entry": raw.get("entry") or raw.get("price") or None,
            "sl": raw.get("sl") or raw.get("stop") or None,
            "tp1": raw.get("tp1") or raw.get("tp") or None,
            "confidence": float(raw.get("confidence") or raw.get("conf") or 0.5),
            "reasons": raw.get("reasons") or raw.get("notes") or []
        }
        # if numeric entry missing try infer from source_df
        if (res["entry"] is None or res["sl"] is None) and source_df is not None:
            try:
                last_close = float(source_df["close"].iloc[-1])
                if res["entry"] is None:
                    res["entry"] = round(last_close, 8)
                if res["sl"] is None:
                    res["sl"] = round(last_close * 0.995, 8)  # small default SL
                if res["tp1"] is None:
                    res["tp1"] = round(last_close * 1.005, 8)
            except Exception:
                pass
        return res

    # If generate_signal returned a simple string (e.g. "STRONG BUY") or pandas object
    if isinstance(raw, str):
        # make simple dict
        base_entry = None
        base_sl = None
        base_tp = None
        # use source_df if available to infer numbers
        if source_df is not None:
            try:
                base_entry = round(float(source_df["close"].iloc[-1]), 8)
                base_sl = round(base_entry * 0.995, 8)
                base_tp = round(base_entry * 1.005, 8)
            except Exception:
                base_entry = None
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "timestamp": datetime.utcnow().isoformat(),
            "signal": raw,
            "entry": base_entry or 0.0,
            "sl": base_sl or 0.0,
            "tp1": base_tp or 0.0,
            "confidence": 0.5,
            "reasons": []
        }

    # If a DataFrame or other type - fallback
    try:
        if hasattr(raw, "iloc") and source_df is None:
            source_df = raw
        if source_df is not None:
            last_close = float(source_df["close"].iloc[-1])
            return {
                "symbol": symbol.upper(),
                "interval": interval,
                "timestamp": datetime.utcnow().isoformat(),
                "signal": "HOLD",
                "entry": round(last_close, 8),
                "sl": round(last_close * 0.995, 8),
                "tp1": round(last_close * 1.005, 8),
                "confidence": 0.5,
                "reasons": ["inferred from price"]
            }
    except Exception:
        pass

    return {"error": "Unsupported signal result format"}

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
        entry_val = float(sig.get("entry") or 0)
        sl_val = float(sig.get("sl") or 0)
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
                # send as message (no image)
                bot.send_message(chat_id or ADMIN_ID, follow)
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
        # always reply with main keyboard and branding image if available
        text = _append_brand("👋 Welcome Boss Destiny!\n\nThis is your Trading Empire control panel.")
        if create_brand_image:
            img = create_brand_image(["Welcome — Destiny Trading Empire Bot 💎"])
            safe_send_with_image(bot, msg.chat.id, text, img, reply_markup=main_keyboard())
        else:
            bot.send_message(msg.chat.id, text, reply_markup=main_keyboard())
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
            bot.send_message(cid, _append_brand("Choose pair to analyze:"), reply_markup=kb)
            return

        # individual pair selected
        if data.startswith("sig_"):
            pair = data.split("_",1)[1]
            bot.send_chat_action(cid, "typing")
            # ----- FLEXIBLE CALL -----
            wrapper = _call_generate_signal_flexibly(pair, SIGNAL_INTERVAL_DEFAULT)
            if wrapper.get("error"):
                bot.send_message(cid, _append_brand(f"Error generating signal: {wrapper['error']}"))
                return
            # Normalize into dict and handle message
            normalized = _normalize_signal(wrapper, pair, SIGNAL_INTERVAL_DEFAULT)
            if normalized.get("error"):
                bot.send_message(cid, _append_brand(f"Error normalizing signal: {normalized['error']}"))
                return
            record_signal_and_send(normalized, chat_id=cid, user_id=call.from_user.id, auto=False)
            return

        # quick scan top X
        if data == "scan_top4":
            bot.send_message(cid, _append_brand("🔎 Scanning top pairs across exchanges..."))
            for p in PAIRS[:6]:
                try:
                    if can_send_signal(p, SIGNAL_INTERVAL_DEFAULT):
                        wrapper = _call_generate_signal_flexibly(p, SIGNAL_INTERVAL_DEFAULT)
                        if wrapper.get("error"):
                            continue
                        sig = _normalize_signal(wrapper, p, SIGNAL_INTERVAL_DEFAULT)
                        if not sig.get("error") and str(sig.get("signal")).upper() in ("LONG","SHORT","STRONG BUY","STRONG SELL"):
                            record_signal_and_send(sig, chat_id=cid)
                except Exception:
                    logging.exception("scan_top4 subtask failed")
            return

        if data == "trending":
            bot.send_message(cid, _append_brand("📡 Fetching multi-exchange trending pairs... please wait."))
            try:
                if fetch_trending_pairs_branded:
                    img_buf, caption = fetch_trending_pairs_branded(limit=8)
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
            # provide brief health info and whether scanner is running
            scanner_running = _scanner_thread is not None and _scanner_thread.is_alive()
            msg = f"⚙️ Bot is running ✅\nScanner running: {scanner_running}\nAuto confidence threshold: {AUTO_CONFIDENCE_THRESHOLD*100:.0f}%"
            bot.send_message(cid, _append_brand(msg))
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

        # Scheduler-based auto briefs (text/AI summaries)
        if data == "start_auto_brief_scheduler":
            if start_scheduler:
                bot.send_message(cid, _append_brand("▶️ Scheduler for auto-briefs enabled. You will receive periodic market briefs."))
                try:
                    start_scheduler(bot)
                except Exception:
                    logging.exception("start_scheduler failed")
            else:
                bot.send_message(cid, _append_brand("Scheduler not available (missing scheduler module)."))
            return

        if data == "stop_auto_brief_scheduler":
            if stop_scheduler:
                stop_scheduler()
                bot.send_message(cid, _append_brand("⏹ Scheduler for auto-briefs disabled."))
            else:
                bot.send_message(cid, _append_brand("Scheduler not available (missing scheduler module)."))
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

                    wrapper = _call_generate_signal_flexibly(pair, interval)
                    if wrapper.get("error"):
                        # don't spam logs every failure, but record once
                        logging.debug("scanner generate error for %s %s: %s", pair, interval, wrapper.get("error"))
                        continue

                    sig = _normalize_signal(wrapper, pair, interval)
                    if sig.get("error"):
                        continue

                    # determine "strong" by confidence or by text
                    try:
                        conf = float(sig.get("confidence", 0.0))
                    except Exception:
                        conf = 0.0
                    s_type = str(sig.get("signal", "")).upper()

                    strong_text_signals = {"STRONG BUY", "STRONG SELL", "LONG", "SHORT"}
                    if (s_type in strong_text_signals and conf >= AUTO_CONFIDENCE_THRESHOLD) or (conf >= AUTO_CONFIDENCE_THRESHOLD and s_type in ("LONG","SHORT")):
                        try:
                            mark_signal_sent(pair, interval)
                            target = ADMIN_ID if AUTO_SEND_ONLY_ADMIN and ADMIN_ID else None
                            record_signal_and_send(sig, chat_id=target, user_id=ADMIN_ID, auto=True)
                            logging.info("[SCANNER] Auto-sent strong signal for %s %s conf=%.2f", pair, interval, conf)
                        except Exception:
                            logging.exception("Failed to record/send auto signal")
                    # mild throttle to avoid rate limits
                    time.sleep(0.6)
            # small pause between full cycles
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
                
