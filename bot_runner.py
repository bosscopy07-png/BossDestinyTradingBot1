# bot_runner.py
import os
import time
import traceback
import requests
import logging
from datetime import datetime
import telebot
from telebot import types

# Branding tag added to every text reply
BRAND_TAG = "\n\n— <b>Boss Destiny Trading Empire</b>"

# Feature modules (light-weight, provided below)
from market_providers import fetch_klines_df, fetch_trending_pairs
from signal_engine import generate_signal_for
from storage import ensure_storage, load_data, save_data, record_pnl_screenshot
from ai_client import ai_analysis_text
from image_utils import build_signal_image, safe_send_with_image
from pro_features import top_gainers_pairs, fear_and_greed_index, futures_leverage_suggestion

# Config
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT").split(",")
SIGNAL_INTERVAL = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "5"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "10"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required in environment variables")

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
ensure_storage()
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_last_signal_time = {}  # cooldown tracker

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

def can_send_signal(symbol: str) -> bool:
    last = _last_signal_time.get(symbol)
    if not last: return True
    return (datetime.utcnow() - last).total_seconds() > COOLDOWN_MIN * 60

def compute_risk_and_size(entry: float, sl: float, balance: float, risk_percent: float):
    risk_amount = (balance * risk_percent) / 100.0
    diff = abs(entry - sl)
    pos_size = 0.0
    if diff > 1e-12:
        pos_size = risk_amount / diff
    return round(risk_amount, 8), round(pos_size, 8)

def record_signal_and_send(sig: dict, chat_id=None, user_id=None):
    data = load_data()
    sig_id = f"S{int(time.time())}"
    balance = data.get("challenge", {}).get("balance", CHALLENGE_START)
    risk_amt, pos_size = compute_risk_and_size(sig["entry"], sig["sl"], balance, RISK_PERCENT)

    rec = {
        "id": sig_id,
        "signal": sig,
        "time": datetime.utcnow().isoformat(),
        "risk_amt": risk_amt,
        "pos_size": pos_size,
        "user": user_id or ADMIN_ID
    }
    data.setdefault("signals", []).append(rec)
    data.setdefault("stats", {})
    data["stats"]["total_signals"] = data["stats"].get("total_signals", 0) + 1
    save_data(data)
    _last_signal_time[sig["symbol"]] = datetime.utcnow()

    # build caption and send image
    caption = (
        f"🔥 <b>Boss Destiny Signal</b>\n"
        f"ID: {sig_id}\nPair: {sig['symbol']} | TF: {sig['interval']}\n"
        f"Signal: <b>{sig['signal']}</b>\nEntry: {sig['entry']} | SL: {sig['sl']} | TP1: {sig['tp1']}\n"
        f"Risk: ${risk_amt:.4f} | Pos size: {pos_size}\n"
        f"Confidence: {int(sig.get('confidence',0)*100)}%\n"
        f"Reasons: {', '.join(sig.get('reasons',[]) or ['None'])}\n\n"
    )
    caption = _append_brand(caption)
    img = build_signal_image(sig)
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
    safe_send_with_image(bot, chat_id or ADMIN_ID, caption, img, kb)
    return sig_id

# Keyboard
def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📈 Get Signal", callback_data="get_signal"),
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
        types.InlineKeyboardButton("🤖 Ask AI", callback_data="ask_ai"),
        types.InlineKeyboardButton("🔄 Refresh Bot", callback_data="refresh_bot")
    )
    return kb

# Handlers
@bot.message_handler(commands=['start', 'menu'])
def cmd_start(msg):
    try:
        bot.send_message(msg.chat.id, f"Welcome — Boss Destiny Trading Empire\nChoose:" , reply_markup=main_keyboard())
    except Exception:
        traceback.print_exc()

@bot.message_handler(content_types=['photo'])
def photo_handler(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(fi.file_path)
        fname = record_pnl_screenshot(data, datetime.utcnow().strftime("%Y%m%d_%H%M%S"), message.from_user.id, message.caption)
        bot.reply_to(message, _append_brand("Saved screenshot. Reply with `#link <signal_id> TP1` or `#link <signal_id> SL`"))
    except Exception:
        traceback.print_exc()
        bot.reply_to(message, _append_brand("Failed to save screenshot."))

@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_handler(message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 3:
            bot.reply_to(message, _append_brand("Usage: #link <signal_id> TP1 or SL"))
            return
        sig_id, tag = parts[1], parts[2].upper()
        d = load_data()
        pnl_item = next((p for p in reversed(d.get("pnl",[])) if p.get("linked") is None and p["from"] == message.from_user.id), None)
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
        save_data(d)
        bot.reply_to(message, _append_brand(f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance."))
    except Exception:
        traceback.print_exc()
        bot.reply_to(message, _append_brand("Failed to link screenshot."))

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        cid = call.message.chat.id
        data = call.data
        bot.answer_callback_query(call.id)
        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"sig_{p}"))
            bot.send_message(cid, _append_brand("Choose pair:"), reply_markup=kb)
            return

        if data.startswith("sig_"):
            pair = data.split("_",1)[1]
            bot.send_chat_action(cid, "typing")
            sig = generate_signal_for(pair, SIGNAL_INTERVAL)
            if sig.get("error"):
                bot.send_message(cid, _append_brand(f"Error generating signal: {sig['error']}"))
                return
            record_signal_and_send(sig, chat_id=cid, user_id=call.from_user.id)
            return

        if data == "scan_top4":
            bot.send_message(cid, _append_brand("🔎 Scanning top pairs..."))
            # generate quick signals for PAIRS top 4
            for p in PAIRS[:4]:
                try:
                    sig = generate_signal_for(p, SIGNAL_INTERVAL)
                    if not sig.get("error") and sig["signal"] in ("BUY","SELL"):
                        record_signal_and_send(sig, chat_id=cid)
                except Exception as e:
                    logging.error("scan_top4 error: %s", e)
            return

        if data == "trending":
            txt = fetch_trending_pairs()
            bot.send_message(cid, _append_brand(txt))
            return

        if data == "bot_status":
            bot.send_message(cid, _append_brand("⚙️ Bot is running ✅"))
            return

        if data == "market_news":
            bot.send_message(cid, _append_brand("📰 Market news: feature coming soon"))
            return

        if data == "challenge_status":
            d = load_data()
            bal = d.get("challenge",{}).get("balance", CHALLENGE_START)
            wins = d.get("challenge",{}).get("wins",0)
            losses = d.get("challenge",{}).get("losses",0)
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

        # AI details callback like ai_S12345
        if data.startswith("ai_"):
            sig_id = data.split("_",1)[1]
            d = load_data()
            rec = next((s for s in d.get("signals",[]) if s["id"]==sig_id), None)
            if not rec:
                bot.send_message(cid, _append_brand("Signal not found"))
                return
            prompt = f"Provide trade rationale, risk controls and a recommended leverage for this trade:\n{rec['signal']}"
            ai_text = ai_analysis_text(prompt)
            bot.send_message(cid, _append_brand(f"🤖 AI analysis:\n{ai_text}"))
            return

        bot.send_message(cid, _append_brand("Unknown action"))
    except Exception:
        traceback.print_exc()
        try:
            bot.answer_callback_query(call.id, "Handler error")
        except:
            pass

# Start polling safely
def start_bot_polling():
    # try to stop other sessions once
    stop_existing_bot_instances()
    logging.info("[BOT] Starting polling loop...")
    while True:
        try:
            # skip_pending helps avoid mass-processing older messages on restart
            bot.infinity_polling(timeout=60, long_polling_timeout=60, skip_pending=True)
        except Exception as e:
            logging.error("[BOT] Polling loop exception: %s", e)
            if "409" in str(e):
                logging.warning("[BOT] 409 Conflict - attempting to stop other sessions and retry")
                stop_existing_bot_instances()
                time.sleep(5)
            else:
                time.sleep(5)
