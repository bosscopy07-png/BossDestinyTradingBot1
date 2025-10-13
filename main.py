# main.py
"""
Entry point for Boss Destiny Trading Bot.
Run as a Background Worker (recommended) with:
    python main.py

If you want to run as a Web Service (not recommended for bots), set RUN_MODE=web
and PORT env var; the code will start a minimal Flask health endpoint.

This file ensures webhook removal to avoid Telegram 409 conflicts and starts the
telegram polling in a safe way.
"""

import os
import threading
import time
import traceback
from datetime import datetime

# Import the bot starter function from bot_process (we'll supply next)
# Keep imports lazy so this file remains small; other files will be separate.
try:
    from bot_process import start_bot_polling, stop_existing_bot_instances, start_health_server
except Exception as e:
    # if bot_process not present yet, print error but keep file usable
    print("Warning: bot_process module not found. Make sure bot_process.py is uploaded next.")
    start_bot_polling = None
    stop_existing_bot_instances = None
    start_health_server = None

RUN_MODE = os.getenv("RUN_MODE", "background").lower()  # "background" or "web"
PORT = int(os.getenv("PORT", "8080"))

def main():
    print(f"[{datetime.utcnow().isoformat()}] Starting main. RUN_MODE={RUN_MODE}")
    # Remove any webhook / stop previous instances (if function available)
    try:
        if stop_existing_bot_instances:
            stop_existing_bot_instances()
    except Exception:
        traceback.print_exc()

    if RUN_MODE == "web":
        # start polling in a separate thread and a small health server to bind a port
        if not start_bot_polling:
            print("Error: bot_process.start_bot_polling not available.")
            return
        poll_thread = threading.Thread(target=start_bot_polling, daemon=True)
        poll_thread.start()
        print("Started polling in background thread; starting health server (web mode).")
        if start_health_server:
            start_health_server(port=PORT)
        else:
            print("Health server not available. Exiting.")
    else:
        # background worker - blocking polling
        if not start_bot_polling:
            print("Error: bot_process.start_bot_polling not available. Upload bot_process.py next.")
            return
        print("Starting bot polling (background worker).")
        start_bot_polling()

if __name__ == "__main__":
    main()
