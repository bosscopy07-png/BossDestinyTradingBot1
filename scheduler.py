# scheduler.py
import threading, time, traceback
from datetime import datetime, timedelta
from pro_features import ai_market_brief_text
from storage import load_data, save_data
from ai_client import ai_analysis_text

# internal state
_scheduler_thread = None
_stop_event = threading.Event()

def _run_periodic_job(chat_id, interval_hours=4):
    print(f"[scheduler] periodic job started; every {interval_hours}h, posting to chat {chat_id}")
    while not _stop_event.is_set():
        try:
            now = datetime.utcnow()
            brief = ai_market_brief_text()
            # send brief via bot. We import telebot lazily to avoid circular import
            import telebot, os
            bot_token = os.getenv("BOT_TOKEN")
            bot = telebot.TeleBot(bot_token, parse_mode="HTML")
            bot.send_message(chat_id, f"🤖 Boss Destiny AI Market Brief ({now.isoformat()}):\n\n{brief}")
        except Exception:
            traceback.print_exc()
        # sleep until next
        for _ in range(int(interval_hours*3600)):
            if _stop_event.is_set(): break
            time.sleep(1)
    print("[scheduler] periodic job stopped.")

def start_scheduler(chat_id, interval_hours=4):
    global _scheduler_thread, _stop_event
    if _scheduler_thread and _scheduler_thread.is_alive():
        return "Scheduler already running."
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_run_periodic_job, args=(chat_id, interval_hours), daemon=True)
    _scheduler_thread.start()
    return "Scheduler started."

def stop_scheduler():
    global _stop_event
    _stop_event.set()
    return "Scheduler stop signal sent."
