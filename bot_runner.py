# bot_runner.py
import os
import time
import traceback
from datetime import datetime

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

import telebot
from telebot import types

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT").split(",")
SIGNAL_INTERVAL = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", 30))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", 5))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

# ---------------- SETUP ----------------
ensure_storage()
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
try:
    bot.remove_webhook()
except Exception:
    pass

_last_signal_time = {}  # cooldown tracker

# ---------------- UTILITIES ----------------
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

    caption = (f"🔥 <b>Boss Destiny Trading Empire — Signal</b>\n"
               f"ID: {sig_id}\nPair: {sig_record['symbol']} | TF: {sig_record['interval']}\n"
               f"Signal: <b>{sig_record['signal']}</b>\nEntry: {sig_record['entry']} | SL: {sig_record['sl']} | TP1: {sig_record.get('tp1')}\n"
               f"Confidence: {int(sig_record.get('confidence', 0) * 100)}% | Risk (USD): {risk_usd}\n"
               f"Reasons: {', '.join(sig_record.get('reasons', [])) if sig_record.get('reasons') else 'None'}\n\n"
               f"— Boss Destiny Trading Empire")
    img = build_signal_image(sig_record)
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
    safe_send_with_image(bot, chat_id or ADMIN_ID, caption, img, kb)
    return sig_id

# ---------------- KEYBOARD ----------------
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
    kb.row(
        types.InlineKeyboardButton("🚀 Top Movers", callback_data="top_gainers"),
        types.InlineKeyboardButton("📈 Fear & Greed", callback_data="fear_greed")
    )
    kb.row(
        types.InlineKeyboardButton("🖼️ Quick Chart", callback_data="open_chart_menu"),
        types.InlineKeyboardButton("⚖️ Futures Suggest", callback_data="open_fut_menu")
    )
    kb.row(
        types.InlineKeyboardButton("▶️ Start Auto Brief", callback_data="start_auto_brief"),
        types.InlineKeyboardButton("⏹ Stop Auto Brief", callback_data="stop_auto_brief")
    )
    return kb

# ---------------- START & STOP ----------------
def stop_existing_bot_instances():
    try:
        bot.remove_webhook()
    except Exception:
        pass

def start_bot_polling():
    try:
        stop_existing_bot_instances()
        print("Starting polling...")
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception:
        traceback.print_exc()
        time.sleep(5)
        start_bot_polling()
