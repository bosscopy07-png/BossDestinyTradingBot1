import threading
import os
import telebot
from flask import Flask

BOT_TOKEN = os.getenv("BOT_TOKEN")  # your bot token from env
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 BossDestiny Trading Empire Bot is running!"

def start_flask_app():
    port = int(os.environ.get("PORT", 8080))
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port)).start()

def start_bot_polling():
    print("🚀 Bot polling started...")
    threading.Thread(target=lambda: bot.infinity_polling(timeout=60, long_polling_timeout=10)).start()

def stop_existing_bot_instances():
    print("🧹 Cleaning old bot instances (if any)...")
    # placeholder, if you want to add PID cleanup logic
    pass
