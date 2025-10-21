# app.py
import os
import threading
import logging
import time
from flask import Flask, jsonify
import bot_runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask("boss_destiny_health")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "service": "boss_destiny_trading_empire_bot",
        "status": "running",
        "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    })

def run_bot():
    bot_runner.start_bot_polling()

if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    port = int(os.getenv("PORT", "10000"))
    logging.info(f"[SERVER] Flask running on port {port}")
    app.run(host="0.0.0.0", port=port, use_reloader=False)
