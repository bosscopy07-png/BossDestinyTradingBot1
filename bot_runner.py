# bot_runner.py
import os
import time
import traceback
import requests
import logging
from datetime import datetime
import telebot
from telebot import types

BRAND_TAG = "\n\n— <b>Boss Destiny Trading Empire</b>"

def send_with_brand(chat_id, text, **kwargs):
    """Send text messages with brand tag automatically."""
    if BRAND_TAG.strip() not in text:
        text += BRAND_TAG
    bot.send_message(chat_id, text, **kwargs)

# ===== IMPORT FEATURES =====
from pro_features import (
    top_gainers_pairs,
    fear_and_greed_index,
    quickchart_price_image,
    futures_leverage_suggestion,
    ai_market_brief_text
)
from market_providers import fetch_klines_multi as fetch_klines_df, fetch_trending_pairs
from signal_engine import generate_signal
from ai_client import ai_analysis_text
from scheduler import start_scheduler, stop_scheduler
from storage import ensure_storage, load_data, save_data, record_pnl_screenshot
from image_utils import build_signal_image, safe_send_with_image

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT").split(",")
SIGNAL_INTERVAL = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", 30))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", 5))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

# ===== SETUP =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
ensure_storage()

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_last_signal_time = {}

# ===== UTILITIES =====
def can_send_signal(symbol):
    last = _last_signal_time.get(symbol)
    if not last:
        return True
    return (datetime.utcnow() - last).total_seconds() > COOLDOWN_MIN * 60


def record_signal_and_send(sig_record, chat_id=None, user_id=None):
    data = load_data()
    sig_id = f"S{int(time.time())}"
    balance = data.get("challenge", {}).get("balance", float(os.getenv("CHALLENGE_START", 10)))
    risk_usd = sig_record.get("suggested_risk_usd", round(balance * RISK_PERCENT / 100, 8))

    rec = {
        "id": sig_id,
        "signal": sig_record,
        "time": datetime.utcnow().isoformat(),
        "risk_amt": risk_usd,
        "result": None,
        "posted_by": user_id or ADMIN_ID
    }

    data["signals"].append(rec)
    data["stats"]["total_signals"] = data["stats"].get("total_signals", 0) + 1
    save_data(data)
    _last_signal_time[sig_record["symbol"]] = datetime.utcnow()

    caption = (
        f"🔥 <b>Boss Destiny Trading Empire — Signal</b>\n"
        f"ID: {sig_id}\nPair: {sig_record['symbol']} | TF: {sig_record['interval']}\n"
        f"Signal: <b>{sig_record['signal']}</b>\nEntry: {sig_record['entry']} | SL: {sig_record['sl']} | TP1: {sig_record.get('tp1')}\n"
        f"Confidence: {int(sig_record.get('confidence', 0) * 100)}% | Risk (USD): {risk_usd}\n"
        f"Reasons: {', '.join(sig_record.get('reasons', [])) if sig_record.get('reasons') else 'None'}\n\n"
        f"— Boss Destiny Trading Empire"
    )

    img = build_signal_image(sig_record)
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))

    safe_send_with_image(bot, chat_id or ADMIN_ID, caption, img, kb)
    return sig_id


# ===== KEYBOARD =====
def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📈 Get Signal", callback_data="get_signal"),
        types.InlineKeyboardButton("🔎 Scan Top 4", callback_data="scan_top4"),
        types.InlineKeyboardButton("⚙️ Bot Status", callback_data="bot_status"),
        types.InlineKeyboardButton("🚀 Trending Pairs", callback_data="trending"),
        types.InlineKeyboardButton("📰 Market News", callback_data="market_news"),
        types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status"),
        types.InlineKeyboardButton("📸 Upload PnL", callback_data="pnl_upload"),
        types.InlineKeyboardButton("🧾 History", callback_data="history"),
        types.InlineKeyboardButton("🤖 Ask AI", callback_data="ask_ai"),
        types.InlineKeyboardButton("🔄 Refresh Bot", callback_data="refresh_bot"),
        types.InlineKeyboardButton("🚀 Top Movers", callback_data="top_gainers"),
        types.InlineKeyboardButton("📈 Fear & Greed", callback_data="fear_greed"),
        types.InlineKeyboardButton("🖼️ Quick Chart", callback_data="open_chart_menu"),
        types.InlineKeyboardButton("⚖️ Futures Suggest", callback_data="open_fut_menu"),
        types.InlineKeyboardButton("▶️ Start Auto Brief", callback_data="start_auto_brief"),
        types.InlineKeyboardButton("⏹ Stop Auto Brief", callback_data="stop_auto_brief")
    )
    return kb

# ===== HANDLERS =====

@bot.message_handler(commands=["start"])
def start_cmd(message):
    bot.send_message(
        message.chat.id,
        f"👋 Welcome Boss Destiny!\n\nThis is your <b>Trading Empire</b> control panel.",
        reply_markup=main_keyboard()
    )

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        bot.answer_callback_query(call.id)

        if call.data == "get_signal":
            bot.send_message(call.message.chat.id, "📈 Generating new trading signal...")
            # you can call: signal = generate_signal() or similar

        elif call.data == "scan_top4":
            bot.send_message(call.message.chat.id, "🔎 Scanning top 4 market pairs...")

        elif call.data == "bot_status":
            bot.send_message(call.message.chat.id, "⚙️ Bot is running smoothly ✅")

        elif call.data == "trending":
            pairs = fetch_trending_pairs()
            bot.send_message(call.message.chat.id, f"🚀 Trending pairs:\n{pairs}")

        elif call.data == "market_news":
            bot.send_message(call.message.chat.id, "📰 Latest market news coming soon...")

        elif call.data == "challenge_status":
            data = load_data()
            balance = data.get("challenge", {}).get("balance", "N/A")
            bot.send_message(call.message.chat.id, f"📊 Current Challenge Balance: {balance}")

        elif call.data == "pnl_upload":
            bot.send_message(call.message.chat.id, "📸 Please send your PnL screenshot...")

        elif call.data == "history":
            bot.send_message(call.message.chat.id, "🧾 Fetching signal history...")

        elif call.data == "ask_ai":
            bot.send_message(call.message.chat.id, "🤖 What would you like to ask the AI?")

        elif call.data == "refresh_bot":
            bot.send_message(call.message.chat.id, "🔄 Refreshing bot session...")
            stop_existing_bot_instances()
            time.sleep(3)
            bot.send_message(call.message.chat.id, "✅ Bot refreshed successfully!")

        elif call.data == "top_gainers":
            gainers = top_gainers_pairs()
            bot.send_message(call.message.chat.id, f"🚀 Top Gainers:\n{gainers}")

        elif call.data == "fear_greed":
            fg = fear_and_greed_index()
            bot.send_message(call.message.chat.id, f"📈 Fear & Greed Index:\n{fg}")

        elif call.data == "open_chart_menu":
            bot.send_message(call.message.chat.id, "🖼️ Opening quick chart menu...")

        elif call.data == "open_fut_menu":
            bot.send_message(call.message.chat.id, "⚖️ Fetching futures suggestions...")
            fut = futures_leverage_suggestion()
            bot.send_message(call.message.chat.id, fut)

        elif call.data == "start_auto_brief":
            bot.send_message(call.message.chat.id, "▶️ Starting AI auto brief...")
            start_scheduler(bot)

        elif call.data == "stop_auto_brief":
            bot.send_message(call.message.chat.id, "⏹ Stopping AI auto brief...")
            stop_scheduler()

        else:
            bot.send_message(call.message.chat.id, f"⚠️ Unknown action: {call.data}")

    except Exception as e:
        logging.error(f"[CALLBACK ERROR] {e}")
        traceback.print_exc()

# ===== SAFE START =====
def stop_existing_bot_instances():
    """Stop any active getUpdates sessions to prevent 409 conflict."""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset=-1"
        requests.get(url, timeout=5)
        logging.info("[BOT] Previous polling sessions stopped successfully.")
    except Exception as e:
        logging.warning(f"[BOT] Could not stop existing sessions: {e}")


def start_bot_polling():
    """Safely start polling — only one active session."""
    stop_existing_bot_instances()
    logging.info("[BOT] Starting Boss Destiny Trading Empire polling...")

    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=60, skip_pending=True)
    except Exception as e:
        if "409" in str(e):
            logging.warning("[BOT] 409 Conflict detected — stopping other sessions and retrying.")
            stop_existing_bot_instances()
            time.sleep(5)
            start_bot_polling()  # retry recursively
        else:
            logging.error(f"[BOT] Unexpected error: {e}")
            traceback.print_exc()
            time.sleep(10)
            start_bot_polling()
