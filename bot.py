# bot.py
import os
import threading
import logging
import time
from flask import Flask, jsonify
import bot_runner  # your bot file (imports will occur)

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
            bot_runner.start_bot_polling()  # blocking call inside bot_runner
            logging.info("[BOT] Polling function exited unexpectedly")
        except Exception as e:
            logging.error(f"[BOT] Crashed: {e}")
            time.sleep(5)

# ---------------- MAIN ENTRY POINT ----------------
if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    port = int(os.getenv("PORT", 10000))
    logging.info(f"[SERVER] Flask running on port {port}")
    app.run(host="0.0.0.0", port=port, use_reloader=False)
