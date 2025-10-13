import threading
from bot import app

import sys, os
sys.dont_write_bytecode = True  # Prevents .pyc caching issues
sys.path.append(os.path.dirname(__file__))  # Ensures correct import path

if __name__ == "__main__":
    # If your bot.py contains the Flask app + polling thread, this will launch both
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080}).start()
