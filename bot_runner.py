import threading
import telebot
from flask import Flask

BOT_TOKEN = "your_bot_token_here"
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

def start_flask_app():
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

def start_bot_polling():
    threading.Thread(target=lambda: bot.infinity_polling(timeout=60, long_polling_timeout=10)).start()

def stop_existing_bot_instances():
    # Optional cleanup logic
    pass
