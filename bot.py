# app.py - Production-ready Flask wrapper for Destiny Trading Empire Bot
import os
import sys
import signal
import logging
import time
import threading
from typing import Optional, Dict, Any
from datetime import datetime
from flask import Flask, jsonify, request

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("destiny_app")

# Import bot runner with error handling
try:
    import bot_runner
    HAS_BOT_RUNNER = True
except ImportError as e:
    logger.error(f"Failed to import bot_runner: {e}")
    bot_runner = None
    HAS_BOT_RUNNER = False

# Global state
app = Flask("destiny_health")
_bot_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_bot_status: Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "last_error": None,
    "restart_count": 0
}


def get_bot_health() -> Dict[str, Any]:
    """Get actual bot health status, not just Flask status."""
    status = _bot_status.copy()
    status["flask_alive"] = True
    status["thread_alive"] = _bot_thread.is_alive() if _bot_thread else False
    
    # Try to get deeper bot status if available
    if HAS_BOT_RUNNER and hasattr(bot_runner, '_bot_polling_thread'):
        status["polling_alive"] = (
            bot_runner._bot_polling_thread.is_alive() 
            if bot_runner._bot_polling_thread else False
        )
    
    return status


@app.route("/", methods=["GET"])
def health():
    """Basic health check endpoint."""
    return jsonify({
        "service": "destiny_trading_empire_bot",
        "status": "healthy" if _bot_status["running"] else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": os.getenv("BOT_VERSION", "unknown")
    })


@app.route("/health/detailed", methods=["GET"])
def detailed_health():
    """Detailed health check with bot internals."""
    health_data = get_bot_health()
    return jsonify(health_data)


@app.route("/stop", methods=["POST"])
def stop_bot():
    """Graceful shutdown endpoint (admin only)."""
    # Simple auth check via header
    admin_token = os.getenv("ADMIN_TOKEN")
    if admin_token and request.headers.get("X-Admin-Token") != admin_token:
        return jsonify({"error": "unauthorized"}), 401
    
    logger.info("Shutdown requested via API")
    _stop_event.set()
    
    # Try to stop bot gracefully
    if HAS_BOT_RUNNER and hasattr(bot_runner, 'stop_bot_polling'):
        try:
            bot_runner.stop_bot_polling()
        except Exception as e:
            logger.warning(f"Error stopping bot: {e}")
    
    return jsonify({"status": "shutdown_initiated"})


def run_bot_with_monitoring():
    """
    Run bot with proper lifecycle management and restart logic.
    """
    global _bot_status
    
    while not _stop_event.is_set():
        try:
            logger.info("[BOT] Starting polling...")
            _bot_status["running"] = True
            _bot_status["started_at"] = datetime.utcnow().isoformat()
            _bot_status["last_error"] = None
            
            if not HAS_BOT_RUNNER:
                logger.error("bot_runner not available, cannot start")
                _bot_status["running"] = False
                time.sleep(10)
                continue
            
            # Start the bot - this blocks until stopped or crashed
            bot_runner.start_bot_polling()
            
            # If we get here, polling returned (shouldn't happen with infinity_polling)
            logger.warning("[BOT] Polling returned unexpectedly")
            _bot_status["restart_count"] += 1
            
        except Exception as e:
            _bot_status["running"] = False
            _bot_status["last_error"] = str(e)
            _bot_status["restart_count"] += 1
            logger.exception(f"[BOT] Crashed (restart #{_bot_status['restart_count']})")
            
            # Exponential backoff for restarts
            backoff = min(30, 3 * (2 ** min(_bot_status["restart_count"] - 1, 4)))
            logger.info(f"[BOT] Restarting in {backoff}s...")
            
            # Wait with stop check
            if _stop_event.wait(timeout=backoff):
                break
    
    logger.info("[BOT] Monitor loop stopped")
    _bot_status["running"] = False


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    _stop_event.set()
    
    if HAS_BOT_RUNNER and hasattr(bot_runner, 'stop_bot_polling'):
        try:
            bot_runner.stop_bot_polling()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
    
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main entry point with production server support."""
    global _bot_thread
    
    # Start bot in background thread
    _bot_thread = threading.Thread(
        target=run_bot_with_monitoring,
        name="BotMonitor",
        daemon=True
    )
    _bot_thread.start()
    logger.info("[MAIN] Bot monitor thread started")
    
    # Get port from environment
    port = int(os.getenv("PORT", 10000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    
    # Production check
    if os.getenv("PRODUCTION", "false").lower() == "true":
        logger.info("[MAIN] Running in production mode")
        # In production, use gunicorn externally
        # This block shouldn't be reached if using gunicorn
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    else:
        logger.info(f"[MAIN] Starting Flask on port {port} (debug={debug})")
        app.run(
            host="0.0.0.0",
            port=port,
            debug=debug,
            use_reloader=False,  # Critical: reloader kills threads
            threaded=True
        )


if __name__ == "__main__":
    main()
