import os
import threading
import time
import traceback
from flask import Flask, jsonify
from bot import start_bot_polling  # OK, because bot.py no longer imports bot_runner

if __name__ == "__main__":
    start_bot_polling()

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
    Start the bot in a separate thread. If it crashes, restart it after 5 seconds.
    """
    while True:
        try:
            print("[BOT] Starting Telegram bot polling...")
            start_bot_polling()
        except Exception:
            print("[BOT] Bot crashed! Restarting in 5 seconds...")
            traceback.print_exc()
            time.sleep(5)

# ---------------- PUBLIC FUNCTIONS ----------------
def stop_existing_bot_instances():
    """
    Placeholder to remove webhook or stop other bot instances if needed.
    """
    try:
        import requests
        port = int(os.getenv("PORT", 8080))
        url = f"http://127.0.0.1:{port}/stop_bot"
        requests.get(url, timeout=5)
    except Exception as e:
        import logging
        logging.warning(f"Could not stop existing bot instances: {e}")

# ---------------- MAIN ENTRY ----------------
if __name__ == "__main__":
    # start bot in background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # start Flask web server
    port = int(os.getenv("PORT", 8080))
    print(f"[SERVER] Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port)
