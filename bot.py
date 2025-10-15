import os
import threading
import logging
import time
from flask import Flask, jsonify

# Import the main bot logic
import bot_runner  # 👈 make sure this matches your filename (bot.runner.py → bot_runner.py)

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------- FLASK APP ----------------
app = Flask("boss_destiny_health")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "service": "boss_destiny_trading_empire_bot",
        "status": "running",
        "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    })

# ---------------- RUN BOT ----------------
def run_bot():
    while True:
        try:
            logging.info("[BOT] Starting Boss Destiny Trading Empire...")
            bot_runner.start_bot_polling()  # 👈 call function from your bot.runner file
        except Exception as e:
            logging.error(f"[BOT] Crashed: {e}")
            time.sleep(5)

# ---------------- MAIN ENTRY POINT ----------------
if __name__ == "__main__":
    # Start Telegram bot thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # Keep Flask alive for Render
    port = int(os.getenv("PORT", 8080))
    logging.info(f"[SERVER] Flask running on port {port}")
    app.run(host="0.0.0.0", port=port)
