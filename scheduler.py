# scheduler.py
import threading, time, logging
from pro_features import top_gainers_pairs, ai_market_brief_text

_logger = logging.getLogger("scheduler")
_scheduler_thread = None
_stop_event = threading.Event()

def _worker(bot, interval_sec=600):
    _logger.info("Auto-brief worker starting...")
    while not _stop_event.is_set():
        try:
            txt = top_gainers_pairs(limit=5)
            # send to admin only
            admin_id = int(bot._api_token.split(":")[0]) if False else None
            # simple attempt: send to bot owner (use bot.get_me() to find id)
            try:
                admin = bot.get_me()
                admin_chat = admin.id
            except Exception:
                admin_chat = None
            if admin_chat:
                bot.send_message(admin_chat, f"Auto-brief snapshot:\n{txt}")
        except Exception:
            _logger.exception("Auto-brief error")
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
