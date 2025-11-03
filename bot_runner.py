# bot_runner.py
import os
import time
import threading
import traceback
import logging
from datetime import datetime
import telebot
from telebot import types
import requests

# Branding
BRAND_TAG = "\n\n— <b>Destiny Trading Empire Bot 💎</b>"

# Config
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN required in env")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT").split(",")
SCAN_INTERVALS = os.getenv("SCAN_INTERVALS", "1m,5m,15m,30m,1h,4h,1d").split(",")
SIGNAL_INTERVAL_DEFAULT = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "1"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "100.0"))
AUTO_CONFIDENCE_THRESHOLD = float(os.getenv("AUTO_CONFIDENCE_THRESHOLD", "0.90"))
AUTO_SEND_ONLY_ADMIN = True

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# imports of modules we created
from market_providers import fetch_trending_pairs_branded, fetch_klines_multi, detect_strong_signals, analyze_pair_multi_timeframes, fetch_trending_pairs_text
from image_utils import build_signal_image, safe_send_with_image, create_brand_image
from signal_engine import generate_signal
from storage import ensure_storage, load_data, save_data, record_pnl_screenshot
from ai_client import ai_analysis_text
from pro_features import quickchart_price_image, ai_market_brief_text
from scheduler import start_scheduler, stop_scheduler

ensure_storage()

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_last_signal_time = {}  # key: symbol|interval -> datetime
_scanner_thread = None
_scanner_stop_event = threading.Event()

def _append_brand(text: str) -> str:
    if BRAND_TAG.strip() not in text:
        return text + BRAND_TAG
    return text

def stop_existing_bot_instances():
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset=-1"
        requests.get(url, timeout=5)
    except Exception:
        logging.exception("stop_existing_bot_instances failed")

def can_send_signal(symbol: str, interval: str) -> bool:
    key = f"{symbol}|{interval}"
    last = _last_signal_time.get(key)
    if not last: return True
    return (datetime.utcnow() - last).total_seconds() > COOLDOWN_MIN * 60

def mark_signal_sent(symbol: str, interval: str):
    _last_signal_time[f"{symbol}|{interval}"] = datetime.utcnow()

def compute_risk(entry, sl, balance, risk_percent):
    try:
        entry = float(entry); sl = float(sl)
        risk_amount = (balance * risk_percent)/100.0
        diff = abs(entry - sl)
        if diff <= 1e-12: return round(risk_amount,8), 0.0
        pos_size = risk_amount / diff
        return round(risk_amount,8), round(pos_size,8)
    except Exception:
        return 0.0, 0.0

def _safe_generate_signal(symbol, interval):
    try:
        return generate_signal(symbol, interval)
    except Exception:
        traceback.print_exc()
        return {"error":"generate_signal_failed"}

def record_signal_and_send(sig: dict, chat_id=None, user_id=None, auto=False):
    # ensure storage
    try:
        data = load_data()
    except Exception:
        data = {}
    sig_id = f"S{int(time.time())}"
    balance = data.get("challenge", {}).get("balance", CHALLENGE_START)
    risk_amt, pos_size = compute_risk(sig.get("entry") or 0, sig.get("sl") or 0, balance, RISK_PERCENT)
    rec = {"id":sig_id,"signal":sig,"time":datetime.utcnow().isoformat(),"risk_amt":risk_amt,"pos_size":pos_size,"user": user_id or ADMIN_ID,"auto":bool(auto)}
    try:
        data.setdefault("signals", []).append(rec)
        data.setdefault("stats", {})
        data["stats"]["total_signals"] = data["stats"].get("total_signals",0)+1
        save_data(data)
    except Exception:
        logging.exception("save_data failed")

    # caption
    conf = int((sig.get("confidence") or 0)*100)
    reasons = ", ".join(sig.get("reasons") or ["None"])
    caption = (f"🔥 <b>Destiny Trading Empire — Signal</b>\n"
               f"ID: {sig_id}\nPair: {sig.get('symbol')} | TF: {sig.get('interval')}\n"
               f"Signal: <b>{sig.get('signal')}</b>\nEntry: {sig.get('entry')} | SL: {sig.get('sl')} | TP1: {sig.get('tp1')}\n"
               f"Confidence: {conf}% | Risk (USD): {risk_amt}\nReasons: {reasons}\n")
    caption = _append_brand(caption)

    # image
    img = None
    try:
        img = build_signal_image(sig)
    except Exception:
        logging.exception("build image failed")

    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🔁 Share", switch_inline_query=f"{sig.get('symbol')}"))

    try:
        if safe_send_with_image:
            safe_send_with_image(bot, chat_id or ADMIN_ID, caption, img, kb)
        else:
            if img:
                bot.send_photo(chat_id or ADMIN_ID, img, caption=caption, reply_markup=kb)
            else:
                bot.send_message(chat_id or ADMIN_ID, caption, reply_markup=kb)
    except Exception:
        logging.exception("send message failed")

    # AI follow-up
    try:
        if ai_analysis_text and sig and not sig.get("error"):
            prompt = f"Provide trade rationale, risk controls and suggested leverage for this trade:\n{sig}"
            ai_text = ai_analysis_text(prompt)
            if ai_text:
                bot.send_message(chat_id or ADMIN_ID, _append_brand("🤖 AI analysis:\n" + ai_text))
    except Exception:
        logging.exception("AI follow-up failed")

    return sig_id

# keyboard
def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📈 Get Signals", callback_data="get_signal"),
           types.InlineKeyboardButton("🔎 Scan Top 6", callback_data="scan_top4"))
    kb.add(types.InlineKeyboardButton("⚙️ Bot Status", callback_data="bot_status"),
           types.InlineKeyboardButton("🚀 Trending Pairs", callback_data="trending"))
    kb.add(types.InlineKeyboardButton("📰 Market News", callback_data="market_news"),
           types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status"))
    kb.add(types.InlineKeyboardButton("📸 Upload PnL", callback_data="pnl_upload"),
           types.InlineKeyboardButton("🧾 History", callback_data="history"))
    kb.add(types.InlineKeyboardButton("🤖 AI Market Brief", callback_data="ask_ai"),
           types.InlineKeyboardButton("🔄 Refresh Bot", callback_data="refresh_bot"))
    kb.add(types.InlineKeyboardButton("▶️ Start Auto Scanner", callback_data="start_auto_brief"),
           types.InlineKeyboardButton("⏹ Stop Auto Scanner", callback_data="stop_auto_brief"))
    kb.add(types.InlineKeyboardButton("📣 Start Auto Briefs", callback_data="start_auto_brief_scheduler"),
           types.InlineKeyboardButton("⛔ Stop Auto Briefs", callback_data="stop_auto_brief_scheduler"))
    return kb

# handlers
@bot.message_handler(commands=['start','menu'])
def cmd_start(msg):
    try:
        text = _append_brand("👋 Welcome Boss Destiny!\n\nThis is your Trading Empire control panel.")
        if create_brand_image:
            img = create_brand_image(["Welcome — Destiny Trading Empire Bot 💎"])
            safe_send_with_image(bot, msg.chat.id, text, img, reply_markup=main_keyboard())
        else:
            bot.send_message(msg.chat.id, text, reply_markup=main_keyboard())
    except Exception:
        logging.exception("cmd_start")

@bot.message_handler(content_types=['photo'])
def photo_handler(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(fi.file_path)
        record_pnl_screenshot(data, datetime.utcnow().strftime("%Y%m%d_%H%M%S"), message.from_user.id, message.caption)
        bot.reply_to(message, _append_brand("Saved screenshot. Reply with `#link <signal_id> TP1` or `#link <signal_id> SL`"))
    except Exception:
        logging.exception("photo_handler failed")
        bot.reply_to(message, _append_brand("Failed to save screenshot."))

@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_handler(message):
    try:
        parts = message.text.strip().split()
        if len(parts)<3:
            bot.reply_to(message, _append_brand("Usage: #link <signal_id> TP1 or SL"))
            return
        sig_id, tag = parts[1], parts[2].upper()
        d = load_data()
        pnl_item = next((p for p in reversed(d.get("pnl",[])) if p.get("linked") is None and p["from"]==message.from_user.id), None)
        if not pnl_item:
            bot.reply_to(message, _append_brand("No unlinked screenshot found."))
            return
        pnl_item["linked"] = {"signal_id":sig_id, "result":tag, "linked_by":message.from_user.id}
        # admin update balance
        if message.from_user.id == ADMIN_ID:
            srec = next((s for s in d.get("signals",[]) if s["id"]==sig_id), None)
            if srec:
                risk = srec.get("risk_amt",0)
                if tag.startswith("TP"):
                    d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) + risk
                    d["challenge"]["wins"] = d["challenge"].get("wins",0)+1
                    d["stats"]["wins"] = d["stats"].get("wins",0)+1
                elif tag == "SL":
                    d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) - risk
                    d["challenge"]["losses"] = d["challenge"].get("losses",0)+1
                    d["stats"]["losses"] = d["stats"].get("losses",0)+1
        save_data(d)
        bot.reply_to(message, _append_brand(f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance."))
    except Exception:
        logging.exception("link_handler")
        bot.reply_to(message, _append_brand("Failed to link screenshot."))

# callbacks
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        cid = call.message.chat.id
        data = call.data
        bot.answer_callback_query(call.id)
        # get_signal -> show pairs
        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"sig_{p}"))
            bot.send_message(cid, _append_brand("Choose pair to analyze:"), reply_markup=kb)
            return
        if data.startswith("sig_"):
            pair = data.split("_",1)[1]
            bot.send_chat_action(cid, "typing")
            sig = _safe_generate_signal(pair, SIGNAL_INTERVAL_DEFAULT)
            if sig.get("error"):
                bot.send_message(cid, _append_brand(f"Error generating signal: {sig['error']}"))
                return
            record_signal_and_send(sig, chat_id=cid, user_id=call.from_user.id, auto=False)
            return
        if data == "scan_top4":
            bot.send_message(cid, _append_brand("🔎 Scanning top pairs..."))
            for p in PAIRS[:6]:
                try:
                    if can_send_signal(p, SIGNAL_INTERVAL_DEFAULT):
                        sig = _safe_generate_signal(p, SIGNAL_INTERVAL_DEFAULT)
                        if not sig.get("error") and sig.get("signal") in ("LONG","SHORT"):
                            record_signal_and_send(sig, chat_id=cid)
                except Exception:
                    logging.exception("scan_top4 error")
            return
        if data == "trending":
            bot.send_message(cid, _append_brand("📡 Fetching trending..."))
            try:
                img_buf, caption = fetch_trending_pairs_branded(limit=8)
                if img_buf:
                    safe_send_with_image(bot, cid, _append_brand(caption), img_buf)
                else:
                    bot.send_message(cid, _append_brand(caption))
            except Exception:
                logging.exception("trending")
                bot.send_message(cid, _append_brand("Failed to fetch trending"))
            return
        if data == "bot_status":
            scanner_running = _scanner_thread is not None and _scanner_thread.is_alive()
            bot.send_message(cid, _append_brand(f"⚙️ Bot running ✅\nScanner running: {scanner_running}\nAuto threshold: {AUTO_CONFIDENCE_THRESHOLD*100:.0f}%"))
            return
        if data == "challenge_status":
            d = load_data(); bal = d.get("challenge",{}).get("balance", CHALLENGE_START)
            wins = d.get("challenge",{}).get("wins",0); losses = d.get("challenge",{}).get("losses",0)
            bot.send_message(cid, _append_brand(f"Balance: ${bal:.2f}\nWins: {wins} Losses: {losses}"))
            return
        if data == "ask_ai":
            bot.send_message(cid, _append_brand("🤖 Ask AI: send a message starting with `AI:` followed by your question."))
            return
        if data == "refresh_bot":
            bot.send_message(cid, _append_brand("🔄 Refreshing..."))
            stop_existing_bot_instances(); time.sleep(2); bot.send_message(cid, _append_brand("✅ Refreshed."))
            return
        if data == "start_auto_brief":
            bot.send_message(cid, _append_brand("▶️ Starting background scanner."))
            start_background_scanner()
            return
        if data == "stop_auto_brief":
            bot.send_message(cid, _append_brand("⏹ Stopping background scanner."))
            stop_background_scanner()
            return
        if data == "start_auto_brief_scheduler":
            if start_scheduler:
                bot.send_message(cid, _append_brand("▶️ Scheduler enabled."))
                try:
                    start_scheduler(bot)
                except Exception:
                    logging.exception("start_scheduler")
            else:
                bot.send_message(cid, _append_brand("Scheduler not available."))
            return
        if data == "stop_auto_brief_scheduler":
            if stop_scheduler:
                stop_scheduler(); bot.send_message(cid, _append_brand("⏹ Scheduler disabled."))
            else:
                bot.send_message(cid, _append_brand("Scheduler not available."))
            return
        if data.startswith("ai_"):
            sig_id = data.split("_",1)[1]
            d = load_data()
            rec = next((s for s in d.get("signals",[]) if s["id"]==sig_id), None)
            if not rec:
                bot.send_message(cid, _append_brand("Signal not found"))
                return
            prompt = f"Provide trade rationale, risk controls, leverage for: {rec['signal']}"
            ai_text = ai_analysis_text(prompt)
            bot.send_message(cid, _append_brand("🤖 AI analysis:\n" + ai_text))
            return
        bot.send_message(cid, _append_brand("Unknown action"))
    except Exception:
        logging.exception("callback_handler")

# scanner
def _scanner_loop():
    logging.info("[SCANNER] started")
    while not _scanner_stop_event.is_set():
        try:
            for interval in SCAN_INTERVALS:
                if _scanner_stop_event.is_set(): break
                for pair in PAIRS:
                    if _scanner_stop_event.is_set(): break
                    if not can_send_signal(pair, interval): continue
                    sig = _safe_generate_signal(pair, interval)
                    if sig.get("error"): continue
                    s_type = sig.get("signal"); conf = float(sig.get("confidence",0.0) or 0.0)
                    if s_type in ("LONG","SHORT") and conf >= AUTO_CONFIDENCE_THRESHOLD:
                        try:
                            mark_signal_sent(pair, interval)
                            target = ADMIN_ID if AUTO_SEND_ONLY_ADMIN and ADMIN_ID else None
                            record_signal_and_send(sig, chat_id=target, user_id=ADMIN_ID, auto=True)
                            logging.info("[SCANNER] Auto-sent %s %s conf=%.2f", pair, interval, conf)
                        except Exception:
                            logging.exception("auto-send fail")
                    time.sleep(0.6)
            time.sleep(2.0)
        except Exception:
            logging.exception("scanner loop error"); time.sleep(1.0)
    logging.info("[SCANNER] stopped")

def start_background_scanner():
    global _scanner_thread, _scanner_stop_event
    if _scanner_thread and _scanner_thread.is_alive():
        logging.info("scanner already running")
        return
    _scanner_stop_event.clear()
    _scanner_thread = threading.Thread(target=_scanner_loop, daemon=True)
    _scanner_thread.start()
    logging.info("scanner started")

def stop_background_scanner():
    global _scanner_thread, _scanner_stop_event
    if not _scanner_thread:
        logging.info("scanner not running")
        return
    _scanner_stop_event.set()
    _scanner_thread.join(timeout=5)
    _scanner_thread = None
    logging.info("scanner stopped")

def start_bot_polling():
    stop_existing_bot_instances()
    logging.info("[BOT] starting polling")
    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=60, skip_pending=True)
        except Exception as e:
            logging.exception("polling crashed, restarting")
            time.sleep(3)
