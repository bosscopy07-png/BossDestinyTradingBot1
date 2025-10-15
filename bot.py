import os
import threading
import time
import traceback
import logging
from flask import Flask, jsonify
import telebot

# ---------------- SETUP LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------- TELEGRAM BOT SETUP ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

# Example start command
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.reply_to(message, "🔥 Boss Destiny Trading Empire is online!")

# ---------------- BOT POLLING FUNCTION ----------------
def start_bot_polling():
    """
    Start the Telegram bot polling loop.
    """
    try:
        logging.info("[BOT] Starting polling...")
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception as e:
        logging.error(f"[BOT] Polling crashed: {e}")
        traceback.print_exc()
        time.sleep(5)
        start_bot_polling()  # auto-restart if it crashes

# ---------------- FLASK HEALTH SERVER ----------------
app = Flask("boss_destiny_health")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "service": "boss_destiny_trading_empire_bot",
        "status": "running",
        "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    })

# ---------------- BOT THREAD ----------------
def run_bot():
    """
    Start the bot in a separate thread and restart if it fails.
    """
    while True:
        try:
            start_bot_polling()
        except Exception:
            logging.error("[BOT] Crashed! Restarting in 5 seconds...")
            traceback.print_exc()
            time.sleep(5)

# ---------------- STOP EXISTING INSTANCES ----------------
def stop_existing_bot_instances():
    """
    Attempts to stop any other running instance (optional).
    """
    try:
        import requests
        port = int(os.getenv("PORT", 8080))
        url = f"http://127.0.0.1:{port}/stop_bot"
        requests.get(url, timeout=5)
    except Exception as e:
        logging.warning(f"Could not stop existing bot instances: {e}")

# ---------------- MAIN ENTRY POINT ----------------
if __name__ == "__main__":
    # Start Telegram bot in background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # Start Flask web server (keeps Render alive)
    port = int(os.getenv("PORT", 8080))
    logging.info(f"[SERVER] Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port)
