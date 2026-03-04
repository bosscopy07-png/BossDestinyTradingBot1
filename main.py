# render_app.py - Optimized for Render.com deployment
import os
import sys
import signal
import logging
import threading
import time
from datetime import datetime
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("render_app")

# Create Flask app
app = Flask(__name__)

# Global state
_bot_running = False
_bot_thread = None
_start_time = datetime.utcnow()


@app.route('/')
def home():
    """Simple health check for Render."""
    return "🚀 Boss Destiny Bot is Running Smoothly on Render!"


@app.route('/health')
def health():
    """Detailed health check."""
    return jsonify({
        "status": "healthy" if _bot_running else "starting",
        "uptime_seconds": (datetime.utcnow() - _start_time).total_seconds(),
        "bot_thread_alive": _bot_thread.is_alive() if _bot_thread else False,
        "timestamp": datetime.utcnow().isoformat()
    })


def run_bot_safe():
    """Run bot with crash recovery."""
    global _bot_running
    
    while True:
        try:
            logger.info("[BOT] Starting polling...")
            _bot_running = True
            
            # Import here to catch import errors
            from bot import start_bot_polling
            start_bot_polling()
            
            # If polling returns normally, log it
            logger.warning("[BOT] Polling returned unexpectedly")
            
        except Exception as e:
            _bot_running = False
            logger.exception(f"[BOT] Error: {e}")
            time.sleep(5)  # Wait before restart
        
        # Check if we should exit (for graceful shutdown)
        if hasattr(run_bot_safe, '_stop_requested'):
            logger.info("[BOT] Stop requested, exiting")
            break


def signal_handler(signum, frame):
    """Handle shutdown signals from Render."""
    logger.info(f"Received signal {signum}, shutting down...")
    
    # Request bot stop
    run_bot_safe._stop_requested = True
    
    # Give bot time to clean up
    time.sleep(1)
    sys.exit(0)


# Register signal handlers for Render's container lifecycle
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    # Get port from Render's environment variable
    port = int(os.environ.get("PORT", 10000))
    
    # Start bot in background thread
    _bot_thread = threading.Thread(target=run_bot_safe, daemon=True)
    _bot_thread.start()
    logger.info(f"[MAIN] Bot thread started on port {port}")
    
    # Start Flask (use_reloader=False is critical for threads)
    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False,
        threaded=True,
        debug=False
    )
    
