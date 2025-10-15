import os
import threading
import logging
import time
from flask import Flask, jsonify

# ✅ Import your main bot logic here
import bot_runner  # make sure your file name is bot_runner.py, not bot.runner.py

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
    """Start the bot safely in a loop."""
    while True:
        try:
            logging.info("[BOT] Starting Boss Destiny Trading Empire...")
            bot_runner.start_bot_polling()
        except Exception as e:
            logging.error(f"[BOT] Crashed: {e}")
            time.sleep(5)

# ---------------- MAIN ENTRY POINT ----------------
if __name__ == "__main__":
    # ✅ Prevent Flask auto-reloader from launching a second process
    if os.getenv("RUN_MAIN") != "true":
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()

    # ✅ Run Flask only once (Render healthcheck)
    port = int(os.getenv("PORT", 8080))
    logging.info(f"[SERVER] Flask running on port {port}")
    app.run(host="0.0.0.0", port=port, use_reloader=False)
