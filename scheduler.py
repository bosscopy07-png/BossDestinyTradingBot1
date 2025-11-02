# scheduler.py
import threading
import time
import logging
from datetime import datetime
from pro_features import top_gainers_pairs, ai_market_brief_text
from storage import load_data, save_data

_logger = logging.getLogger("scheduler")
_scheduler_thread = None
_stop_event = threading.Event()

def _worker(bot, interval_sec=300):
    _logger.info("Auto-brief worker starting...")
    while not _stop_event.is_set():
        try:
            # get top gainers text and broadcast to admin for review
            txt = top_gainers_pairs(limit=5)
            bot.send_message(int(bot.chat_id) if hasattr(bot, "chat_id") else bot.get_me().id, f"Auto-brief snapshot:\n{txt}")
        except Exception as e:
            _logger.exception("Auto-brief error: %s", e)
        _stop_event.wait(interval_sec)
    _logger.info("Auto-brief worker stopping...")

def start_scheduler(bot, interval_sec=600):
    global _scheduler_thread, _stop_event
    if _scheduler_thread and _scheduler_thread.is_alive():
        _logger.info("Scheduler already running")
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_worker, args=(bot, interval_sec), daemon=True)
    _scheduler_thread.start()
    _logger.info("Scheduler started")

def stop_scheduler():
    global _stop_event
    _stop_event.set()
    _logger.info("Scheduler stop requested")
