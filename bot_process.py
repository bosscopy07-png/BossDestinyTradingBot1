import threading
from bot import app

if __name__ == "__main__":
    # If your bot.py contains the Flask app + polling thread, this will launch both
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080}).start()
