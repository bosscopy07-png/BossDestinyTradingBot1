# bot.py
import os, threading, logging, time
from flask import Flask, jsonify
import bot_runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask("destiny_health")

@app.route("/", methods=["GET"])
def health():
    return jsonify({"service":"destiny_trading_empire_bot","status":"running","time":time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())})

def run_bot():
    while True:
        try:
            logging.info("[BOT] starting")
            bot_runner.start_bot_polling()
        except Exception:
            logging.exception("bot crashed; restarting")
            time.sleep(3)

if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True); t.start()
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
