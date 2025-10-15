from flask import Flask
import threading
from bot import start_bot_polling

app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Boss Destiny Bot is Running Smoothly on Render!"

def run_bot():
    start_bot_polling()

if __name__ == "__main__":
    threading.Thread(target=run_bot, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
