# bot_runner.py (production-ready, orchestration layer only)
import os
import time
import threading
import logging
from datetime import datetime

import telebot
from telebot import types

# ----- Configuration -----
BRAND_TAG = "\n\n— <b>Destiny Trading Empire Bot 💎</b>"

BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PAIRS = os.getenv(
    "PAIRS",
    "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,MATICUSDT,ADAUSDT",
).split(",")

SIGNAL_INTERVAL_DEFAULT = os.getenv("SIGNAL_INTERVAL", "1h")
AUTO_SEND_ONLY_ADMIN = os.getenv("AUTO_SEND_ONLY_ADMIN", "True").lower() in ("1", "true", "yes")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

# ----- Logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("bot_runner")

# ----- Import Services (thin layer) -----
# These modules handle their own logic; bot_runner just orchestrates
try:
    import signal_engine
    from signal_engine import (
        generate_and_send_signal,
        start_auto_scanner,
        stop_auto_scanner,
    )
except Exception as e:
    signal_engine = None
    generate_and_send_signal = None
    start_auto_scanner = None
    stop_auto_scanner = None
    logger.warning(f"signal_engine unavailable: {e}")

try:
    import storage
    from storage import record_pnl_screenshot, get_signal_history
except Exception as e:
    storage = None
    record_pnl_screenshot = None
    get_signal_history = None
    logger.warning(f"storage unavailable: {e}")

try:
    import ai_client
    from ai_client import get_market_brief
except Exception as e:
    ai_client = None
    get_market_brief = None
    logger.warning(f"ai_client unavailable: {e}")

try:
    import market_providers
    from market_providers import get_trending_pairs_text
except Exception as e:
    market_providers = None
    get_trending_pairs_text = None
    logger.warning(f"market_providers unavailable: {e}")

try:
    import scheduler
    from scheduler import start_scheduler, stop_scheduler
except Exception as e:
    scheduler = None
    start_scheduler = None
    stop_scheduler = None
    logger.warning(f"scheduler unavailable: {e}")

try:
    from image_utils import create_welcome_image, send_image_or_text
except Exception as e:
    create_welcome_image = None
    send_image_or_text = None
    logger.warning(f"image_utils unavailable: {e}")

# ----- Bot State -----
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_bot_polling_thread = None
_polling_stop_event = threading.Event()

# ----- Core Helpers -----
def _append_brand(text: str) -> str:
    """Append brand tag if not present."""
    if BRAND_TAG.strip() not in text:
        return text + BRAND_TAG
    return text

def send_plain_message(chat_id: int, text: str) -> None:
    """Send simple text message with branding."""
    try:
        bot.send_message(chat_id, _append_brand(text))
    except Exception:
        logger.exception(f"Failed to send message to {chat_id}")

def send_rich_message(chat_id: int, text: str, image_bytes=None, reply_markup=None) -> None:
    """Send message with image fallback."""
    try:
        if image_bytes and send_image_or_text:
            send_image_or_text(bot, chat_id, _append_brand(text), image_bytes, reply_markup)
        else:
            bot.send_message(chat_id, _append_brand(text), reply_markup=reply_markup)
    except Exception:
        logger.exception("Failed to send rich message")
        # Fallback to plain text
        try:
            bot.send_message(chat_id, _append_brand(text), reply_markup=reply_markup)
        except Exception:
            logger.exception("Fallback message failed")

# ----- Keyboard Factory -----
def main_keyboard():
    """Create main control panel keyboard."""
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
        types.InlineKeyboardButton("📃 My Challenge", callback_data="challenge_status")
    )
    kb.add(
        types.InlineKeyboardButton("📷 Upload PnL", callback_data="pnl_upload"),
        types.InlineKeyboardButton("📋 History", callback_data="history")
    )
    kb.add(
        types.InlineKeyboardButton("🤖 AI Market Brief", callback_data="ask_ai"),
        types.InlineKeyboardButton("🔄 Refresh Bot", callback_data="refresh_bot")
    )
    kb.add(
        types.InlineKeyboardButton("▶ Start Auto Scanner", callback_data="start_auto"),
        types.InlineKeyboardButton("⏹ Stop Auto Scanner", callback_data="stop_auto")
    )
    kb.add(
        types.InlineKeyboardButton("📣 Start Auto Briefs", callback_data="start_scheduler"),
        types.InlineKeyboardButton("⛔ Stop Auto Briefs", callback_data="stop_scheduler")
    )
    return kb

# ----- Command Handlers -----
@bot.message_handler(commands=["start", "menu"])
def cmd_start(msg):
    """Handle /start and /menu commands."""
    text = "👋 Welcome Boss Destiny!\n\nThis is your Trading Empire control panel."
    lines = [
        "Welcome — Destiny Trading Empire Bot 💎",
        "Use the buttons to get signals, start scanners, view trending pairs."
    ]
    
    # Try to send branded image welcome
    if create_welcome_image:
        try:
            img = create_welcome_image(lines, title="Destiny Trading Empire Bot 💎")
            send_rich_message(msg.chat.id, text, img, main_keyboard())
            return
        except Exception:
            logger.exception("Welcome image failed, falling back to text")
    
    send_plain_message(msg.chat.id, text)

@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    """Handle PnL screenshot uploads."""
    try:
        # Download photo
        file_info = bot.get_file(message.photo[-1].file_id)
        image_data = bot.download_file(file_info.file_path)
        
        # Store via storage module
        if record_pnl_screenshot:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            record_pnl_screenshot(
                image_data, 
                timestamp, 
                message.from_user.id, 
                message.caption
            )
            bot.reply_to(
                message, 
                _append_brand("✅ Screenshot saved. Reply with `#link <signal_id> TP1` or `#link <signal_id> SL`")
            )
        else:
            bot.reply_to(message, _append_brand("⚠️ Storage unavailable"))
            
    except Exception:
        logger.exception("Photo handler failed")
        bot.reply_to(message, _append_brand("❌ Failed to save screenshot"))

# ----- Callback Router -----
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    """Route all callback queries to appropriate handlers."""
    chat_id = call.message.chat.id
    data = call.data
    logger.info(f"Callback: {data} from {chat_id}")
    
    try:
        # Route to specific handler based on action prefix
        if data == "get_signal":
            _handle_get_signal(chat_id)
        elif data == "scan_top4":
            _handle_scan_top4(chat_id)
        elif data == "trending":
            _handle_trending(chat_id)
        elif data == "bot_status":
            _handle_bot_status(chat_id)
        elif data == "ask_ai":
            _handle_ai_brief(chat_id)
        elif data == "history":
            _handle_history(chat_id)
        elif data == "pnl_upload":
            _handle_pnl_upload(chat_id)
        elif data == "start_auto":
            _handle_start_auto(chat_id)
        elif data == "stop_auto":
            _handle_stop_auto(chat_id)
        elif data == "start_scheduler":
            _handle_start_scheduler(chat_id)
        elif data == "stop_scheduler":
            _handle_stop_scheduler(chat_id)
        elif data == "refresh_bot":
            _handle_refresh(chat_id)
        elif data.startswith("link_"):
            _handle_link_pnl(chat_id, data)
        else:
            send_plain_message(chat_id, "⚠️ Unknown command")
            
    except Exception:
        logger.exception(f"Callback handler failed for {data}")
        send_plain_message(chat_id, "⚠️ An error occurred")

# ----- Callback Action Handlers -----
def _handle_get_signal(chat_id: int):
    """Generate signals for all pairs."""
    if not generate_and_send_signal:
        send_plain_message(chat_id, "⚠️ Signal engine unavailable")
        return
    
    for pair in PAIRS:
        try:
            generate_and_send_signal(bot, pair, SIGNAL_INTERVAL_DEFAULT, chat_id)
        except Exception:
            logger.exception(f"Signal generation failed for {pair}")
            send_plain_message(chat_id, f"⚠️ Failed to generate signal for {pair}")

def _handle_scan_top4(chat_id: int):
    """Scan top 4 pairs only."""
    if not generate_and_send_signal:
        send_plain_message(chat_id, "⚠️ Signal engine unavailable")
        return
    
    for pair in PAIRS[:4]:
        try:
            generate_and_send_signal(bot, pair, SIGNAL_INTERVAL_DEFAULT, chat_id)
        except Exception:
            logger.exception(f"Scan failed for {pair}")

def _handle_trending(chat_id: int):
    """Show trending pairs."""
    if get_trending_pairs_text:
        text = get_trending_pairs_text(PAIRS)
        send_plain_message(chat_id, f"🚀 Trending Pairs:\n{text}")
    else:
        send_plain_message(chat_id, "⚠️ Trending data unavailable")

def _handle_bot_status(chat_id: int):
    """Check bot health."""
    polling_ok = _bot_polling_thread and _bot_polling_thread.is_alive()
    status = "🟢 Running" if polling_ok else "🔴 Stopped"
    send_plain_message(chat_id, f"Bot Status: {status}")

def _handle_ai_brief(chat_id: int):
    """Generate AI market analysis."""
    if not get_market_brief:
        send_plain_message(chat_id, "⚠️ AI service unavailable")
        return
    
    try:
        brief = get_market_brief(PAIRS)
        send_plain_message(chat_id, f"🤖 AI Market Brief:\n{brief}")
    except Exception:
        logger.exception("AI brief failed")
        send_plain_message(chat_id, "⚠️ Failed to generate brief")

def _handle_history(chat_id: int):
    """Show signal history."""
    if not get_signal_history:
        send_plain_message(chat_id, "⚠️ History unavailable")
        return
    
    try:
        signals = get_signal_history(limit=10)
        if signals:
            text = "\n".join([
                f"{s['symbol']} | {s['signal']} | {s['time']}" 
                for s in signals
            ])
        else:
            text = "No signals yet"
        send_plain_message(chat_id, f"📋 Last 10 Signals:\n{text}")
    except Exception:
        logger.exception("History fetch failed")
        send_plain_message(chat_id, "⚠️ Failed to load history")

def _handle_pnl_upload(chat_id: int):
    """Prompt for PnL upload."""
    send_plain_message(chat_id, "📷 Send me a photo of your PnL screenshot")

def _handle_start_auto(chat_id: int):
    """Start auto scanner."""
    if not start_auto_scanner:
        send_plain_message(chat_id, "⚠️ Auto scanner unavailable")
        return
    
    try:
        start_auto_scanner(bot, PAIRS, SIGNAL_INTERVAL_DEFAULT)
        send_plain_message(chat_id, "▶ Auto Scanner Started")
    except Exception:
        logger.exception("Auto scanner start failed")
        send_plain_message(chat_id, "⚠️ Failed to start scanner")

def _handle_stop_auto(chat_id: int):
    """Stop auto scanner."""
    if not stop_auto_scanner:
        send_plain_message(chat_id, "⚠️ Auto scanner unavailable")
        return
    
    try:
        stop_auto_scanner()
        send_plain_message(chat_id, "⏹ Auto Scanner Stopped")
    except Exception:
        logger.exception("Auto scanner stop failed")

def _handle_start_scheduler(chat_id: int):
    """Start scheduled briefs."""
    if not start_scheduler:
        send_plain_message(chat_id, "⚠️ Scheduler unavailable")
        return
    
    try:
        start_scheduler(bot, PAIRS)
        send_plain_message(chat_id, "📣 Auto Brief Scheduler Started")
    except Exception:
        logger.exception("Scheduler start failed")

def _handle_stop_scheduler(chat_id: int):
    """Stop scheduled briefs."""
    if not stop_scheduler:
        send_plain_message(chat_id, "⚠️ Scheduler unavailable")
        return
    
    try:
        stop_scheduler()
        send_plain_message(chat_id, "⛔ Auto Brief Scheduler Stopped")
    except Exception:
        logger.exception("Scheduler stop failed")

def _handle_refresh(chat_id: int):
    """Restart bot polling."""
    send_plain_message(chat_id, "🔄 Refreshing...")
    stop_bot_polling()
    time.sleep(1)
    start_bot_polling()
    send_plain_message(chat_id, "✅ Bot refreshed")

def _handle_link_pnl(chat_id: int, data: str):
    """Handle PnL linking callback."""
    sig_id = data.split("_", 1)[1]
    msg = f"🔗 Link PnL for signal `{sig_id}`\nReply with: `#link {sig_id} TP1` or `#link {sig_id} SL`"
    send_plain_message(chat_id, msg)

# ----- Lifecycle Management -----
def start_bot_polling():
    """Start bot polling in daemon thread."""
    global _bot_polling_thread
    
    if _bot_polling_thread and _bot_polling_thread.is_alive():
        logger.info("Polling already active")
        return True
    
    _polling_stop_event.clear()
    
    def _poll_loop():
        try:
            bot.infinity_polling(
                timeout=60,
                long_polling_timeout=60,
                none_stop=True
            )
        except Exception:
            logger.exception("Polling loop terminated")
    
    _bot_polling_thread = threading.Thread(target=_poll_loop, daemon=True)
    _bot_polling_thread.start()
    logger.info("Bot polling started")
    return True

def stop_bot_polling():
    """Stop bot polling gracefully."""
    try:
        bot.stop_polling()
        logger.info("Bot polling stopped")
    except Exception:
        logger.exception("Stop polling error")

def clear_pending_updates():
    """Clear old updates to prevent flood on restart."""
    try:
        import requests
        requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset=-1",
            timeout=5
        )
        logger.info("Cleared pending updates")
    except Exception as e:
        logger.warning(f"Could not clear updates: {e}")

# ----- Entry Point -----
if __name__ == "__main__":
    logger.info("=== Destiny Trading Empire Bot Starting ===")
    clear_pending_updates()
    start_bot_polling()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        stop_bot_polling()
        
