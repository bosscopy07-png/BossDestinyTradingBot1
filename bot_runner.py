# bot_runner.py
# Destiny Trading Empire Signal Bot - Production-ready version (improved 2025)

import os
import sys
import time
import threading
import logging
import signal
from datetime import datetime, timedelta
from collections import defaultdict
import json
import re

import telebot
from telebot import types

# ────────────────────────────────────────
#  Environment & Constants
# ────────────────────────────────────────

BRAND_TAG = "\n\n— <b>Destiny Trading Empire Bot 💎</b>"

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is required")

ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
ADMIN_IDS = {ADMIN_ID}  # can become list/set later

PAIRS = [
    p.strip().upper()
    for p in os.getenv(
        "PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,MATICUSDT,ADAUSDT"
    ).split(",")
    if p.strip()
]

SIGNAL_INTERVAL_DEFAULT = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MINUTES      = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
RISK_PERCENT          = float(os.getenv("RISK_PERCENT", "1.0"))
CHALLENGE_START       = float(os.getenv("CHALLENGE_START", "100.0"))
AUTO_CONFIDENCE_THRESHOLD = float(os.getenv("AUTO_CONFIDENCE_THRESHOLD", "0.90"))

AUTO_PUBLISH_ONLY_TO_ADMIN = os.getenv("AUTO_SEND_ONLY_ADMIN", "true").lower() in (
    "true", "1", "yes", "on"
)

# ────────────────────────────────────────
# Logging
# ────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("destiny-bot")

# ────────────────────────────────────────
# Lazy / Optional module imports
# ────────────────────────────────────────

market_providers = None
fetch_trending_pairs_text = fetch_klines_multi = analyze_pair_multi_timeframes = None
generate_branded_signal_image = None

signal_engine = None
se_start_auto_scanner = se_stop_auto_scanner = None

storage = None
load_data = save_data = ensure_storage = record_pnl_screenshot = None

ai_client = None
ai_analysis_text = None

image_utils = None
create_brand_image = safe_send_with_image = quickchart_image_bytes = None

scheduler = None
start_scheduler = stop_scheduler = None

try:
    import market_providers
    from market_providers import (
        fetch_trending_pairs_text, fetch_klines_multi,
        analyze_pair_multi_timeframes, generate_branded_signal_image
    )
except Exception:
    logger.exception("market_providers import failed")

try:
    import signal_engine
    from signal_engine import start_auto_scanner, stop_auto_scanner
except Exception:
    logger.exception("signal_engine import failed")

try:
    import storage
    from storage import ensure_storage, load_data, save_data, record_pnl_screenshot
except Exception:
    logger.exception("storage import failed")

try:
    import ai_client
    from ai_client import ai_analysis_text
except Exception:
    logger.exception("ai_client import failed")

try:
    import image_utils
    from image_utils import create_brand_image, safe_send_with_image, quickchart_image_bytes
except Exception:
    logger.exception("image_utils import failed")

try:
    import scheduler
    from scheduler import start_scheduler, stop_scheduler
except Exception:
    logger.exception("scheduler import failed")

# Ensure storage folder / files
if ensure_storage:
    try:
        ensure_storage()
    except Exception:
        logger.exception("ensure_storage failed")

# ────────────────────────────────────────
# Bot instance
# ────────────────────────────────────────

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

# ────────────────────────────────────────
# Global runtime state
# ────────────────────────────────────────

_last_signal_time = {}                    # "SYMBOL|TF" → datetime
_last_callback_time = defaultdict(lambda: datetime(2000,1,1))
_scanner_thread = None
_bot_polling_thread = None

# ────────────────────────────────────────
# Helpers
# ────────────────────────────────────────

def append_brand(text: str) -> str:
    return text if BRAND_TAG.strip() in text else text + BRAND_TAG


def send_message(chat_id: int, text: str, **kwargs):
    try:
        bot.send_message(chat_id, append_brand(text), **kwargs)
    except Exception:
        logger.exception(f"send_message failed to {chat_id}")


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def rate_limit_ok(call) -> bool:
    now = datetime.utcnow()
    key = f"{call.from_user.id}:{call.data}"
    if (now - _last_callback_time[key]).total_seconds() < 1.4:
        bot.answer_callback_query(call.id, "⏳ Please wait...", show_alert=False)
        return False
    _last_callback_time[key] = now
    return True


def module_available(name: str, obj) -> bool:
    if obj is None:
        logger.warning(f"Feature unavailable: {name}")
        return False
    return True


def can_send_signal(symbol: str, interval: str) -> bool:
    key = f"{symbol.upper()}|{interval}"
    last = _last_signal_time.get(key)
    if last is None:
        return True
    return (datetime.utcnow() - last).total_seconds() > COOLDOWN_MINUTES * 60


def mark_signal_sent(symbol: str, interval: str):
    key = f"{symbol.upper()}|{interval}"
    _last_signal_time[key] = datetime.utcnow()


def should_auto_publish(sig: dict) -> bool:
    if sig.get("signal", "HOLD") == "HOLD":
        return False
    if sig.get("confidence", 0) < AUTO_CONFIDENCE_THRESHOLD:
        return False
    if AUTO_PUBLISH_ONLY_TO_ADMIN:
        return False  # for now — can be extended later with user context
    return can_send_signal(sig["symbol"], sig["interval"])


# ────────────────────────────────────────
# Main menu keyboard
# ────────────────────────────────────────

def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    buttons = [
        ("📈 Get Signals",           "get_signal"),
        ("🔎 Scan Top 4",            "scan_top4"),
        ("🚀 Trending Pairs",        "trending"),
        ("⚙️ Bot Status",           "bot_status"),
        ("📰 Market News",          "market_news"),
        ("📃 My Challenge",          "challenge_status"),
        ("📷 Upload PnL",            "pnl_upload"),
        ("📋 History",               "history"),
        ("🤖 AI Market Brief",       "ask_ai"),
        ("🔄 Refresh Bot",           "refresh_bot"),
        ("▶ Start Auto Scanner",     "start_auto_scanner"),
        ("⏹ Stop Auto Scanner",      "stop_auto_scanner"),
    ]
    for text, cb in buttons:
        kb.add(types.InlineKeyboardButton(text, callback_data=cb))
    return kb


# ────────────────────────────────────────
# Command handlers
# ────────────────────────────────────────

@bot.message_handler(commands=["start", "menu"])
def cmd_start(message):
    text = "👋 Welcome Boss Destiny!\n\nControl panel for signals, scanners & market intel."
    try:
        if module_available("image_utils - create_brand_image", create_brand_image):
            img = create_brand_image(
                ["Destiny Trading Empire", "Signal & Automation Bot"],
                title="Destiny Trading Empire Bot 💎"
            )
            if module_available("image_utils - safe_send", safe_send_with_image):
                safe_send_with_image(bot, message.chat.id, append_brand(text), img, reply_markup=main_keyboard())
                return
            bot.send_photo(message.chat.id, img, caption=append_brand(text), reply_markup=main_keyboard())
            return
    except Exception:
        logger.exception("welcome image creation failed")
    bot.send_message(message.chat.id, append_brand(text), reply_markup=main_keyboard())


@bot.message_handler(commands=["help"])
def cmd_help(message):
    text = (
        "<b>Commands & features</b>\n\n"
        "/start /menu   — main menu\n"
        "/help           — this message\n"
        "/status         — bot & scanner status\n\n"
        "<b>Main actions via buttons:</b>\n"
        "• Get Signals\n"
        "• Scan Top 4 pairs\n"
        "• Trending pairs\n"
        "• Auto scanner start/stop\n"
        "• Upload PnL screenshots\n"
        "• View signal history\n"
        "• AI market brief\n"
    )
    send_message(message.chat.id, text)


@bot.message_handler(commands=["status"])
def cmd_status(message):
    scanner_alive = _scanner_thread is not None and _scanner_thread.is_alive()
    lines = [
        f"Bot polling   : {'🟢 running' if _bot_polling_thread and _bot_polling_thread.is_alive() else '🔴 stopped'}",
        f"Auto scanner  : {'🟢 running' if scanner_alive else '⏹ stopped'}",
        f"Pairs watched : {len(PAIRS)}",
        f"Default TF    : {SIGNAL_INTERVAL_DEFAULT}",
        f"Cooldown      : {COOLDOWN_MINUTES} min",
        f"Risk / trade  : {RISK_PERCENT}%",
        f"Auto threshold: {AUTO_CONFIDENCE_THRESHOLD:.0%}",
        f"Auto only admin: {'yes' if AUTO_PUBLISH_ONLY_TO_ADMIN else 'no'}",
    ]
    if module_available("storage", load_data):
        try:
            d = load_data() or {}
            bal = d.get("challenge", {}).get("balance", CHALLENGE_START)
            lines.append(f"Challenge bal : ${bal:,.2f}")
        except:
            pass
    send_message(message.chat.id, "\n".join(lines))


# ────────────────────────────────────────
# Placeholder for the big callback handler
# (you can keep expanding this part)
# ────────────────────────────────────────

@bot.callback_query_handler(func=lambda call: True)
def callback_router(call):
    if not rate_limit_ok(call):
        return

    chat_id = call.message.chat.id
    user_id = call.from_user.id
    data = call.data

    logger.info(f"Callback  {data:24}  user={user_id:10}  chat={chat_id}")

    if data in {"start_auto_scanner", "stop_auto_scanner", "refresh_bot"} and not is_admin(user_id):
        bot.answer_callback_query(call.id, "🔒 Admin only", show_alert=True)
        return

    # ── Add your existing callback logic here ───────────────────────
    # For now just a few examples:

    if data == "bot_status":
        cmd_status(call.message)   # reuse command logic
        bot.answer_callback_query(call.id, "Status checked")

    elif data == "trending":
        if module_available("market_providers", fetch_trending_pairs_text):
            text = fetch_trending_pairs_text(PAIRS)
            send_message(chat_id, f"<b>Trending pairs:</b>\n{text}")
        else:
            send_message(chat_id, "Trending module not available")
        bot.answer_callback_query(call.id)

    # ... add get_signal, scan_top4, ask_ai, history, etc.

    else:
        send_message(chat_id, "Action not implemented yet")
        bot.answer_callback_query(call.id, "🚧 Under construction")


# ────────────────────────────────────────
# Graceful shutdown
# ────────────────────────────────────────

def shutdown_handler(signum=None, frame=None):
    logger.info("Shutdown signal received...")
    try:
        if module_available("signal_engine", se_stop_auto_scanner):
            se_stop_auto_scanner()
        if module_available("scheduler", stop_scheduler):
            stop_scheduler()
        bot.stop_polling()
    except Exception:
        logger.exception("Cleanup during shutdown failed")
    finally:
        logger.info("Bot shutdown complete.")
        sys.exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# ────────────────────────────────────────
# Polling control
# ────────────────────────────────────────

def start_polling():
    global _bot_polling_thread
    if _bot_polling_thread and _bot_polling_thread.is_alive():
        return
    def poll():
        try:
            bot.infinity_polling(timeout=65, long_polling_timeout=65)
        except Exception:
            logger.exception("Polling loop crashed")
    _bot_polling_thread = threading.Thread(target=poll, daemon=True, name="TgPolling")
    _bot_polling_thread.start()


def stop_polling():
    try:
        bot.stop_polling()
    except Exception:
        logger.exception("stop_polling failed")


# ────────────────────────────────────────
# Entry point
# ────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Destiny Trading Empire Bot starting...")
    try:
        stop_polling()          # clear stuck sessions if any
        time.sleep(0.4)
        start_polling()
        logger.info("Bot polling started")
        while True:
            time.sleep(3600)    # keep main thread alive
    except KeyboardInterrupt:
        shutdown_handler()
    except Exception:
        logger.critical("Main loop crashed", exc_info=True)
        shutdown_handler()
