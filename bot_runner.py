bot_runner.py (edited)

import os import time import threading import traceback import requests import logging from datetime import datetime import telebot from telebot import types

----- Branding and global constants -----

BRAND_TAG = "\n\n— <b>Destiny Trading Empire Bot \U0001f48e</b>"

Config from env (with sane defaults)

BOT_TOKEN = os.getenv("BOT_TOKEN") ADMIN_ID = int(os.getenv("ADMIN_ID", "0")) PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,MATICUSDT,ADAUSDT").split(",")

scan all timeframes per requirement:

SCAN_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"] SIGNAL_INTERVAL_DEFAULT = os.getenv("SIGNAL_INTERVAL", "1h") COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30")) RISK_PERCENT = float(os.getenv("RISK_PERCENT", "1")) CHALLENGE_START = float(os.getenv("CHALLENGE_START", "100.0"))

Auto-send parameters:

AUTO_CONFIDENCE_THRESHOLD = float(os.getenv("AUTO_CONFIDENCE_THRESHOLD", "0.90"))   # 0.90 = 90% AUTO_SEND_ONLY_ADMIN = True        # send to admin only (as requested)

if not BOT_TOKEN: raise RuntimeError("BOT_TOKEN environment variable required")

----- Logging & storage init -----

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s") logger = logging.getLogger("bot_runner")

Import optional project modules (if missing we fallback gracefully)

try: from market_providers import ( fetch_trending_pairs_branded, fetch_klines_multi, get_session, fetch_trending_pairs_text, analyze_pair_multi_timeframes, detect_strong_signals, generate_branded_signal_image ) except Exception: fetch_trending_pairs_branded = None fetch_klines_multi = None get_session = None fetch_trending_pairs_text = None analyze_pair_multi_timeframes = None detect_strong_signals = None generate_branded_signal_image = None logger.exception("market_providers import failed")

try: from image_utils import build_signal_image, safe_send_with_image, create_brand_image except Exception: build_signal_image = None safe_send_with_image = None create_brand_image = None logger.exception("image_utils import failed")

Try to import signal_engine with multiple names (legacy and modern)

try: from signal_engine import ( generate_signal as legacy_generate_signal, generate_signal_multi as generate_signal_multi, detect_strong_signals as se_detect_strong_signals, register_send_callback as register_send_callback, start_auto_scanner as se_start_auto_scanner ) except Exception: # fallback: import what exists, don't fail hard try: from signal_engine import generate_signal as legacy_generate_signal except Exception: legacy_generate_signal = None logger.exception("signal_engine legacy import failed") try: from signal_engine import generate_signal_multi as generate_signal_multi except Exception: generate_signal_multi = None try: from signal_engine import detect_strong_signals as se_detect_strong_signals except Exception: se_detect_strong_signals = None try: from signal_engine import register_send_callback as register_send_callback except Exception: register_send_callback = None try: from signal_engine import start_auto_scanner as se_start_auto_scanner except Exception: se_start_auto_scanner = None logger.exception("signal_engine partial import attempted")

try: from storage import ensure_storage, load_data, save_data, record_pnl_screenshot except Exception: ensure_storage = None load_data = None save_data = None record_pnl_screenshot = None logger.exception("storage import failed")

try: from ai_client import ai_analysis_text except Exception: ai_analysis_text = None logger.exception("ai_client import failed")

try: from pro_features import top_gainers_pairs, fear_and_greed_index, futures_leverage_suggestion, quickchart_price_image, ai_market_brief_text except Exception: top_gainers_pairs = None fear_and_greed_index = None futures_leverage_suggestion = None quickchart_price_image = None ai_market_brief_text = None logger.exception("pro_features import failed")

Scheduler (for auto-briefs)

try: from scheduler import start_scheduler, stop_scheduler except Exception: start_scheduler = None stop_scheduler = None logger.exception("scheduler import failed")

ensure storage directory/data if module available

if ensure_storage: try: ensure_storage() except Exception: logger.exception("ensure_storage failed")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML") _last_signal_time = {}  # dict mapping (symbol|interval) -> datetime of last auto-send _scanner_thread = None _scanner_stop_event = threading.Event()

----- helpers -----

def _append_brand(text: str) -> str: if BRAND_TAG.strip() not in text: return text + BRAND_TAG return text

def _send_branded(chat_id, text, lines_for_image=None, reply_markup=None): """ Always attempt to send a small branded image + caption. If image can't be created, send text only (with brand). lines_for_image is a list of lines used to build a brand image if create_brand_image exists. """ try: caption = _append_brand(text) if create_brand_image and lines_for_image: try: img = create_brand_image(lines_for_image, title="Destiny Trading Empire Bot \U0001f48e") if safe_send_with_image: safe_send_with_image(bot, chat_id, caption, img, reply_markup=reply_markup) return else: # telebot send img.seek(0) bot.send_photo(chat_id, img, caption=caption, reply_markup=reply_markup) return except Exception: logger.exception("create_brand_image failed; falling back to text") # fallback: if there is an image buffer passed directly via safe_send_with_image if safe_send_with_image and isinstance(lines_for_image, (bytes, bytearray)): safe_send_with_image(bot, chat_id, caption, lines_for_image, reply_markup=reply_markup) return # final fallback: send text message bot.send_message(chat_id, caption, reply_markup=reply_markup) except Exception: logger.exception("Failed to _send_branded") try: bot.send_message(chat_id, _append_brand("\u26A0\uFE0F Failed to deliver message.")) except Exception: pass

def stop_existing_bot_instances(): """Try clear pending getUpdates sessions to reduce 409 conflicts.""" try: url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset=-1" requests.get(url, timeout=5) logger.info("[BOT] Attempted to stop other bot sessions (getUpdates offset -1).") except Exception as e: logger.warning(f"[BOT] Could not call Telegram getUpdates clear: {e}")

def can_send_signal(symbol: str, interval: str) -> bool: """Respect cooldown per symbol+interval (auto-sends).""" key = f"{symbol}|{interval}" last = _last_signal_time.get(key) if not last: return True return (datetime.utcnow() - last).total_seconds() > COOLDOWN_MIN * 60

def mark_signal_sent(symbol: str, interval: str): key = f"{symbol}|{interval}" _last_signal_time[key] = datetime.utcnow()

def compute_risk_and_size(entry: float, sl: float, balance: float, risk_percent: float): risk_amount = (balance * risk_percent) / 100.0 diff = abs(entry - sl) if diff <= 1e-12: return round(risk_amount, 8), 0.0 pos_size = risk_amount / diff return round(risk_amount, 8), round(pos_size, 8)

----- signal generation integration -----

def _safe_generate_signal(symbol: str, interval: str): """ Primary path: use analyze_pair_multi_timeframes() from market_providers to get a multi-TF analysis. If that fails, fallback to fetching klines and legacy signal_engine. Returns a standardized dict: { symbol, interval, signal (LONG/SHORT/HOLD), entry, sl, tp1, confidence (0..1), reasons: [] } """ try: # 1) Try the multi-TF analyzer (best) if analyze_pair_multi_timeframes: try: res = analyze_pair_multi_timeframes(symbol, timeframes=[interval] + [tf for tf in SCAN_INTERVALS if tf != interval]) # res should contain combined_signal and combined_score if isinstance(res, dict) and not res.get("error"): combined = res.get("combined_signal", "HOLD") score = float(res.get("combined_score", 0.0)) # pick interval close if available, else any close entry = None try: entry = res["analysis"].get(interval, {}).get("close") or next(iter(res["analysis"].values())).get("close") except Exception: entry = None # use sl/tp1 from 1h if exists else from interval analysis sl = None; tp1 = None try: sl = res["analysis"].get("1h", {}).get("sl") or res["analysis"].get(interval, {}).get("sl") tp1 = res["analysis"].get("1h", {}).get("tp1") or res["analysis"].get(interval, {}).get("tp1") except Exception: pass reasons = [] # compile reasons across timeframes (top few) try: for tf, info in res.get("analysis", {}).items(): if isinstance(info, dict): reasons.extend(info.get("reasons", [])) except Exception: pass

# Optional AI augmentation: if ai_analysis_text exists, ask AI for extra rationale and trust modifier
                if ai_analysis_text and score > 0.4:
    try:
        # Convert res to string safely in case it's a dict
        res_text = str(res) if not isinstance(res, str) else res

        # Corrected f-string: use double braces {{}} for literal {}
        prompt = (
            f"Given the following multi-timeframe analysis for {symbol} on {interval}:\n"
            f"{res_text}\n"
            "Provide concise trade rationale and, if appropriate, a trust modifier between -0.05 and +0.1 to adjust confidence. "
            "Reply JSON: {{}}"
        )

        ai_resp = ai_analysis_text(prompt)

        # Attempt to parse a numeric modifier out of AI response (best-effort)
        import re
        m = re.search(r"([+-]?[0-9]*\.?[0-9]+)", str(ai_resp))
        if m:
            mod = float(m.group(1))
            score = max(0.0, min(1.0, score + mod))
            reasons.append("ai_adj")
    except Exception:
        logger.exception("AI augmentation failed")

                return {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "signal": "LONG" if "LONG" in combined or "STRONG_LONG" in combined else ("SHORT" if "SHORT" in combined or "STRONG_SHORT" in combined else "HOLD"),
                    "entry": float(entry) if entry else None,
                    "sl": float(sl) if sl else None,
                    "tp1": float(tp1) if tp1 else None,
                    "confidence": float(score),
                    "reasons": list(dict.fromkeys(reasons))  # dedupe
                }
        except Exception:
            logger.exception("analyze_pair_multi_timeframes failed; falling back")

    # 2) Fallback: try to fetch klines for exchange choices and run legacy generate_signal
    # Try multiple exchanges to get usable data
    exchanges_to_try = ["binance", "bybit", "kucoin", "okx"]
    if fetch_klines_multi:
        for ex in exchanges_to_try:
            try:
                df = fetch_klines_multi(symbol, interval, limit=200, exchange=ex)
                if df is None or df.empty or "close" not in df:
                    continue
                # If legacy_generate_signal exists, try it
                if legacy_generate_signal:
                    try:
                        # legacy_generate_signal sometimes expects just df and returns string or dict
                        out = legacy_generate_signal(df, pair=symbol) if callable(legacy_generate_signal) else legacy_generate_signal
                        sig_text = str(out)
                        if "BUY" in sig_text.upper() or "STRONG BUY" in sig_text.upper():
                            sig = "LONG"
                        elif "SELL" in sig_text.upper():
                            sig = "SHORT"
                        else:
                            sig = "HOLD"
                        # basic SL/TP using ATR
                        try:
                            highs = df["high"].astype(float)
                            lows = df["low"].astype(float)
                            closes = df["close"].astype(float)
                            atr_val = (highs - lows).rolling(14).mean().iloc[-1]
                            last = float(closes.iloc[-1])
                            if sig == "LONG":
                                sl = last - (atr_val * 1.5)
                                tp1 = last + (atr_val * 1.5)
                            elif sig == "SHORT":
                                sl = last + (atr_val * 1.5)
                                tp1 = last - (atr_val * 1.5)
                            else:
                                sl = last * 0.995
                                tp1 = last * 1.005
                        except Exception:
                            sl = None; tp1 = None
                        return {
                            "symbol": symbol.upper(),
                            "interval": interval,
                            "signal": sig,
                            "entry": float(closes.iloc[-1]),
                            "sl": float(sl) if sl else None,
                            "tp1": float(tp1) if tp1 else None,
                            "confidence": 0.3,
                            "reasons": [sig_text]
                        }
                    except Exception:
                        logger.exception("legacy_generate_signal failed on df")
                        continue
        # if no df succeeded, return error
        return {"error": "no_data_on_exchanges"}
    else:
        return {"error": "no_fetch_klines_available"}
except Exception as exc:
    logger.exception("_safe_generate_signal unexpected error")
    return {"error": str(exc)}

----- recording & messaging -----

def record_signal_and_send(sig: dict, chat_id=None, user_id=None, auto=False): """Record a signal in storage and send it (image + caption).""" # storage read try: d = load_data() if load_data else {} except Exception: d = {}

sig_id = f"S{int(time.time())}"
balance = d.get("challenge", {}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START

# risk and pos
risk_amt, pos_size = compute_risk_and_size(
    sig.get("entry") or 0.0,
    sig.get("sl") or 0.0,
    balance, RISK_PERCENT
)

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
    logger.exception("Failed to save signal record")

# Build caption
try:
    confidence_pct = int(sig.get("confidence", 0) * 100)
except Exception:
    confidence_pct = 0

caption = (
    f"\uDD25 <b>Destiny Trading Empire — Signal</b>\n"
    f"ID: {sig_id}\nPair: {sig.get('symbol')} | TF: {sig.get('interval')}\n"
    f"Signal: <b>{sig.get('signal')}</b>\nEntry: {sig.get('entry') or 'N/A'} | SL: {sig.get('sl') or 'N/A'} | TP1: {sig.get('tp1') or 'N/A'}\n"
    f"Confidence: {confidence_pct}% | Risk (USD): {risk_amt}\n"
    f"Reasons: {', '.join(sig.get('reasons', []) or ['None'])}\n"
)
caption = _append_brand(caption)

# Image creation
img = None
try:
    # if market_providers gave a branded image generator or pro_features quickchart exists use them
    if generate_branded_signal_image and isinstance(sig, dict):
        try:
            img_buf, _ = generate_branded_signal_image({
                "symbol": sig.get("symbol"),
                "analysis": None,
                "combined_score": sig.get("confidence", 0),
                "combined_signal": sig.get("signal"),
                "sl": sig.get("sl"),
                "tp1": sig.get("tp1"),
                "image": None,
                "caption_lines": [f"{sig.get('symbol')} | {sig.get('interval')} | {sig.get('signal')}"]
            })
            if img_buf:
                img = img_buf
        except Exception:
            logger.exception("generate_branded_signal_image failed")
    # fallback to simple builder
    if not img and build_signal_image:
        img = build_signal_image(sig)
except Exception:
    logger.exception("build_signal_image failed")

# Keyboard for message
kb = types.InlineKeyboardMarkup(row_width=2)
kb.add(types.InlineKeyboardButton("\U0001F4F7 Link PnL", callback_data=f"link_{sig_id}"))
kb.add(types.InlineKeyboardButton("\U0001F916 AI Details", callback_data=f"ai_{sig_id}"))
kb.add(types.InlineKeyboardButton("\U0001F501 Share", switch_inline_query=f"{sig.get('symbol')}"))

# send (use safe_send_with_image if available)
try:
    if safe_send_with_image:
        safe_send_with_image(bot, chat_id or ADMIN_ID, caption, img, kb)
    else:
        if img:
            # telebot expects file-like for send_photo
            try:
                img.seek(0)
            except Exception:
                pass
            bot.send_photo(chat_id or ADMIN_ID, img, caption=caption, reply_markup=kb)
        else:
            bot.send_message(chat_id or ADMIN_ID, caption, reply_markup=kb)
except Exception:
    logger.exception("Failed to send signal message")

# optionally send a quick AI rationale follow-up if available
try:
    if ai_analysis_text and sig and not sig.get("error"):
        prompt = f"Provide concise trade rationale for this signal:\n{sig}"
        ai_text = ai_analysis_text(prompt)
        if ai_text:
            follow = _append_brand(f"\U0001F916 AI Rationale:\n{ai_text}")
            bot.send_message(chat_id or ADMIN_ID, follow)
except Exception:
    logger.exception("AI rationale follow-up failed")

return sig_id

----- keyboard UI -----

def main_keyboard(): kb = types.InlineKeyboardMarkup(row_width=2) kb.add( types.InlineKeyboardButton("\U0001F4C8 Get Signals", callback_data="get_signal"), types.InlineKeyboardButton("\U0001F50E Scan Top 4", callback_data="scan_top4") ) kb.add( types.InlineKeyboardButton("\u2699\ufe0f Bot Status", callback_data="bot_status"), types.InlineKeyboardButton("\U0001F680 Trending Pairs", callback_data="trending") ) kb.add( types.InlineKeyboardButton("\U0001F4F0 Market News", callback_data="market_news"), types.InlineKeyboardButton("\U0001F4C3 My Challenge", callback_data="challenge_status") ) kb.add( types.InlineKeyboardButton("\U0001F4F7 Upload PnL", callback_data="pnl_upload"), types.InlineKeyboardButton("\U0001F4CB History", callback_data="history") ) kb.add( types.InlineKeyboardButton("\U0001F916 AI Market Brief", callback_data="ask_ai"), types.InlineKeyboardButton("\U0001F504 Refresh Bot", callback_data="refresh_bot") ) kb.add( types.InlineKeyboardButton("\u25B6 Start Auto Scanner", callback_data="start_auto_brief"), types.InlineKeyboardButton("\23F9 Stop Auto Scanner", callback_data="stop_auto_brief") ) kb.add( types.InlineKeyboardButton("\U0001F4E3 Start Auto Briefs", callback_data="start_auto_brief_scheduler"), types.InlineKeyboardButton("\26D4 Stop Auto Briefs", callback_data="stop_auto_brief_scheduler") ) return kb

----- Telegram handlers -----
@bot.message_handler(commands=['start', 'menu'])
def cmd_start(msg):
    try:
        # always reply with main keyboard and branding image if available
        text = "\U0001F44B Welcome Boss Destiny!\n\nThis is your Trading Empire control panel."
        # prepare lines for image
        lines = ["Welcome — Destiny Trading Empire Bot \U0001f48e", "Use the buttons to get signals, start scanners, view trending pairs."]
        if create_brand_image:
            try:
                img = create_brand_image(lines, title="Destiny Trading Empire Bot \U0001f48e")
                if safe_send_with_image:
                    safe_send_with_image(bot, msg.chat.id, _append_brand(text), img, reply_markup=main_keyboard())
                    return
                else:
                    img.seek(0)
                    bot.send_photo(msg.chat.id, img, caption=_append_brand(text), reply_markup=main_keyboard())
                    return
            except Exception:
                logger.exception("Failed creating welcome brand image")
        bot.send_message(msg.chat.id, _append_brand(text), reply_markup=main_keyboard())
    except Exception:
        logger.exception("cmd_start failed")

@bot.message_handler(content_types=['photo'])
def photo_handler(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(fi.file_path)
        if record_pnl_screenshot:
            record_pnl_screenshot(data, datetime.utcnow().strftime("%Y%m%d_%H%M%S"), message.from_user.id, message.caption)
        bot.reply_to(message, _append_brand("Saved screenshot. Reply with `#link <signal_id> TP1` or `#link <signal_id> SL`"))
    except Exception:
        logger.exception("photo_handler failed")
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
        logger.exception("link_handler failed")
        bot.reply_to(message, _append_brand("Failed to link screenshot."))

# ----- Callback actions -----
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        cid = call.message.chat.id
        data = call.data
        # make sure to answer callback quickly to avoid spinner
        try:
            bot.answer_callback_query(call.id)
        except Exception:
            pass

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
                # if no data try again with a different exchange/timeframe message
                err = sig.get("error")
                bot.send_message(cid, _append_brand(f"Error generating signal: {err}\nTry another timeframe or check pair symbol."))
                return
            # Standardize missing fields
            sig.setdefault("interval", SIGNAL_INTERVAL_DEFAULT)
            sig.setdefault("symbol", pair.upper())
            record_signal_and_send(sig, chat_id=cid, user_id=call.from_user.id, auto=False)
            return

        # quick scan top X
        if data == "scan_top4":
            bot.send_message(cid, _append_brand("\U0001F50E Scanning top pairs across exchanges..."))
            for p in PAIRS[:6]:
                try:
                    if can_send_signal(p, SIGNAL_INTERVAL_DEFAULT):
                        sig = _safe_generate_signal(p, SIGNAL_INTERVAL_DEFAULT)
                        if not sig.get("error") and sig.get("signal") in ("LONG","SHORT"):
                            record_signal_and_send(sig, chat_id=cid)
                except Exception:
                    logger.exception("scan_top4 subtask failed")
            return

        if data == "trending":
            bot.send_message(cid, _append_brand("\U0001F4E1 Fetching multi-exchange trending pairs... please wait."))
            try:
                # support multi-exchange aggregation if detect_strong_signals available
                if fetch_trending_pairs_branded:
                    img_buf, caption = fetch_trending_pairs_branded(limit=8)
                    if img_buf:
                        if safe_send_with_image:
                            safe_send_with_image(bot, cid, _append_brand(caption), img_buf)
                        else:
                            img_buf.seek(0)
                            bot.send_photo(cid, img_buf, caption=_append_brand(caption))
                    else:
                        bot.send_message(cid, _append_brand(caption))
                elif fetch_trending_pairs_text:
                    bot.send_message(cid, _append_brand(fetch_trending_pairs_text()))
                else:
                    bot.send_message(cid, _append_brand("Trending feature not available (missing market_providers)."))
            except Exception:
                logger.exception("trending handler failed")
                bot.send_message(cid, _append_brand("Failed to fetch trending pairs."))
            return

        if data == "bot_status":
            # provide brief health info and whether scanner is running
            scanner_running = _scanner_thread is not None and _scanner_thread.is_alive()
            msg = f"\u2699\ufe0f Bot is running \u2705\nScanner running: {scanner_running}\nAuto confidence threshold: {AUTO_CONFIDENCE_THRESHOLD*100:.0f}%"
            bot.send_message(cid, _append_brand(msg))
            return

        if data == "market_news":
            bot.send_message(cid, _append_brand("\U0001F4F0 Market news: feature coming soon"))
            return

        if data == "challenge_status":
            d = load_data() if load_data else {}
            bal = d.get("challenge",{}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START
            wins = d.get("challenge",{}).get("wins",0) if isinstance(d, dict) else 0
            losses = d.get("challenge",{}).get("losses",0) if isinstance(d, dict) else 0
            bot.send_message(cid, _append_brand(f"Balance: ${bal:.2f}\nWins: {wins} Losses: {losses}"))
            return

        if data == "ask_ai":
            bot.send_message(cid, _append_brand("\U0001F916 Ask AI: send a message starting with `AI:` followed by your question."))
            return

        if data == "refresh_bot":
            bot.send_message(cid, _append_brand("\U0001F504 Refreshing bot session..."))
            stop_existing_bot_instances()
            time.sleep(2)
            bot.send_message(cid, _append_brand("\u2705 Refreshed."))
            return

        if data == "start_auto_brief":
            bot.send_message(cid, _append_brand("\u25B6 Starting background market scanner (auto-send strong signals)."))
            start_background_scanner()
            return

        if data == "stop_auto_brief":
            bot.send_message(cid, _append_brand("\u23F9 Stopping background market scanner."))
            stop_background_scanner()
            return

        # Scheduler-based auto briefs (text/AI summaries)
        if data == "start_auto_brief_scheduler":
            if start_scheduler:
                bot.send_message(cid, _append_brand("\u25B6 Scheduler for auto-briefs enabled. You will receive periodic market briefs."))
                try:
                    start_scheduler(bot)
                except Exception:
                    logger.exception("start_scheduler failed")
            else:
                bot.send_message(cid, _append_brand("Scheduler not available (missing scheduler module)."))
            return

        if data == "stop_auto_brief_scheduler":
            if stop_scheduler:
                stop_scheduler()
                bot.send_message(cid, _append_brand("\u23F9 Scheduler for auto-briefs disabled."))
            else:
                bot.send_message(cid, _append_brand("Scheduler not available (missing scheduler module)."))
            return

        if data.startswith("ai_"):
            sig_id = data.split("_",1)[1]
            d = load_data() if load_data else {}
            rec = next((s for s in d.get("signals",[]) if s["id")==sig_id), None) if isinstance(d, dict) else None
            if not rec:
                bot.send_message(cid, _append_brand("Signal not found"))
                return
            prompt = f"Provide trade rationale, risk controls and a recommended leverage for this trade:\n{rec['signal']}"
            ai_text = ai_analysis_text(prompt) if ai_analysis_text else "AI feature not available"
            bot.send_message(cid, _append_brand(f"\U0001F916 AI analysis:\n{ai_text}"))
            return

        bot.send_message(cid, _append_brand("Unknown action"))
    except Exception:
        logger.exception("callback_handler failed")
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
    logger.info("[SCANNER] Background scanner started (multi-TF).")
    exchanges_list = os.getenv("SCAN_EXCHANGES", "binance,bybit,kucoin,okx").split(",")
    while not _scanner_stop_event.is_set():
        try:
            # Use detect_strong_signals if available (market_providers)
            if detect_strong_signals or se_detect_strong_signals:
                try:
                    # aggregate candidates across exchanges
                    aggregated = []
                    detectors = []
                    if detect_strong_signals:
                        detectors.append((detect_strong_signals, "market_providers"))
                    if se_detect_strong_signals:
                        detectors.append((se_detect_strong_signals, "signal_engine"))
                    for det, src in detectors:
                        for ex in exchanges_list:
                            try:
                                cands = det(pairs=PAIRS, timeframes=SCAN_INTERVALS, exchange=ex, min_confidence=AUTO_CONFIDENCE_THRESHOLD)
                                if isinstance(cands, list):
                                    for c in cands:
                                        c['_src'] = src
                                        c['_exchange'] = ex
                                        aggregated.append(c)
                            except Exception:
                                logger.exception("detector %s failed for exchange %s", src, ex)
                    # dedupe by symbol and pick best confidence
                    best = {}
                    for c in aggregated:
                        sym = c.get('symbol')
                        sc = float(c.get('combined_score', 0.0))
                        if sym not in best or sc > float(best[sym].get('combined_score', 0.0)):
                            best[sym] = c
                    for sym, cand in best.items():
                        if _scanner_stop_event.is_set():
                            break
                        pair = cand.get('symbol')
                        conf = float(cand.get('combined_score', 0.0))
                        interval = SIGNAL_INTERVAL_DEFAULT
                        if conf >= AUTO_CONFIDENCE_THRESHOLD and can_send_signal(pair, interval):
                            try:
                                mark_signal_sent(pair, interval)
                                target = ADMIN_ID if AUTO_SEND_ONLY_ADMIN and ADMIN_ID else None
                                img = cand.get('image')
                                cap = "\n".join(cand.get('caption_lines', [])) if cand.get('caption_lines') else f"Auto strong signal {pair}"
                                cap = _append_brand(cap)
                                if img and safe_send_with_image:
                                    safe_send_with_image(bot, target or ADMIN_ID, cap, img)
                                else:
                                    # prepare signal dict and record/send
                                    sig = {
                                        "symbol": pair,
                                        "interval": interval,
                                        "signal": cand.get('combined_signal', 'HOLD'),
                                        "entry": cand.get('analysis', {}).get('1h', {}).get('close') if cand.get('analysis') else None,
                                        "sl": cand.get('sl'),
                                        "tp1": cand.get('tp1'),
                                        "confidence": cand.get('combined_score', 0.0),
                                        "reasons": []
                                    }
                                    record_signal_and_send(sig, chat_id=target, user_id=ADMIN_ID, auto=True)
                                logger.info("[SCANNER] Auto-sent %s conf=%.3f", pair, conf)
                            except Exception:
                                logger.exception("Failed to send candidate from detectors")
                    time.sleep(5.0)
                    continue
                except Exception:
                    logger.exception("detect_strong_signals detectors failed; fallback scanning below")

            # fallback scanning: loop each pair/timeframe via _safe_generate_signal
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
                            logger.info("[SCANNER] Auto-sent strong signal for %s %s conf=%.2f", pair, interval, conf)
                        except Exception:
                            logger.exception("Failed to record/send auto signal")
                    time.sleep(0.6)
            time.sleep(2.0)
        except Exception:
            logger.exception("Unhandled error in scanner loop")
            time.sleep(1.0)
    logger.info("[SCANNER] Background scanner stopped.")


def start_background_scanner():
    global _scanner_thread, _scanner_stop_event
    if _scanner_thread and _scanner_thread.is_alive():
        logger.info("[SCANNER] Already running.")
        return
    _scanner_stop_event.clear()
    _scanner_thread = threading.Thread(target=_scanner_loop, daemon=True)
    _scanner_thread.start()
    logger.info("[SCANNER] Started background scanner thread.")


def stop_background_scanner():
    global _scanner_thread, _scanner_stop_event
    if not _scanner_thread:
        logger.info("[SCANNER] Not running.")
        return
    _scanner_stop_event.set()
    _scanner_thread.join(timeout=5)
    _scanner_thread = None
    logger.info("[SCANNER] Stop requested and thread joined.")


# ----- signal send callback registration (optional) -----
def _signal_build_caption(signal_dict: dict) -> str:
    try:
        reasons = signal_dict.get("reasons") or []
        reasons_text = ", ".join(reasons) if isinstance(reasons, (list, tuple)) else str(reasons)
        conf_pct = float(signal_dict.get("confidence", 0.0)) * 100.0
        caption = (
            f"<b>{signal_dict.get('symbol')} | {signal_dict.get('interval')}</b>\n"
            f"Signal: <b>{signal_dict.get('signal')}</b>\n"
            f"Entry: {signal_dict.get('entry')}  SL: {signal_dict.get('sl')}  TP1: {signal_dict.get('tp1')}\n"
            f"Confidence: {conf_pct:.1f}%\n"
            f"Source: {signal_dict.get('source_exchange')}\n"
            f"Reasons: {reasons_text}\n"
            f"<i>{signal_dict.get('timestamp')}</i>"
        )
        return caption
    except Exception:
        return str(signal_dict)


def _signal_send_callback(signal_dict: dict):
    try:
        chat_id = ADMIN_ID if AUTO_SEND_ONLY_ADMIN and ADMIN_ID else None
        if not chat_id:
            logger.warning("No ADMIN_ID configured; skipping auto-signal send")
            return
        caption = _signal_build_caption(signal_dict)
        # try to send branded image if present
        try:
            lines = signal_dict.get('caption_lines') or caption.split('\n')[:12]
            chart = signal_dict.get('chart_bytes') or signal_dict.get('chart_img') if isinstance(signal_dict.get('chart_img'), (bytes, bytearray)) else None
            if create_brand_image:
                try:
                    img_buf = create_brand_image(lines, chart_img_bytes=chart)
                except TypeError:
                    img_buf = create_brand_image(lines)
                if img_buf and safe_send_with_image:
                    safe_send_with_image(bot, chat_id, caption, img_buf)
                    return
                elif img_buf:
                    try:
                        img_buf.seek(0)
                    except Exception:
                 pass
                    bot.send_photo(chat_id, img_buf, caption=caption)
                    return
        except Exception:
            logger.exception("create_brand_image/send image failed; falling back to text")
        # fallback to text
        bot.send_message(chat_id, caption, parse_mode="HTML")
    except Exception:
        logger.exception("Failed to send auto-signal callback")

# register callback if the signal engine exposes register_send_callback
try:
    if register_send_callback:
        try:
            register_send_callback(_signal_send_callback)
            logger.info("Registered external signal send callback with signal_engine")
        except Exception:
            logger.exception("register_send_callback call failed")
    else:
        logger.debug("No register_send_callback available in signal_engine")
except Exception:
    logger.exception("Error while attempting to register signal callback")

# Optionally start signal engine provided auto scanner if available
try:
    if se_start_auto_scanner:
        try:
            # se_start_auto_scanner may accept different signature; try common one
            se_start_auto_scanner(pairs=PAIRS, interval=SIGNAL_INTERVAL_DEFAULT, exchanges=os.getenv("SCAN_EXCHANGES", "binance,bybit,kucoin,okx").split(","), min_confidence=AUTO_CONFIDENCE_THRESHOLD, poll_seconds=int(os.getenv("SCAN_POLL_SEC", 300)))
            logger.info("Started external signal_engine auto scanner")
        except Exception:
            logger.exception("signal_engine.start_auto_scanner failed to start")
except Exception:
    pass


# ----- Start polling safely (exported) -----
def start_bot_polling():
    stop_existing_bot_instances()
    logger.info("[BOT] Starting polling loop...")
    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=60, skip_pending=True)
        except Exception as e:
            logger.error("[BOT] Polling loop exception: %s", e)
            if "409" in str(e):
                logger.warning("[BOT] 409 Conflict - attempting to stop other sessions and retry")
                stop_existing_bot_instances()
                time.sleep(5)
            else:
                time.sleep(5)
