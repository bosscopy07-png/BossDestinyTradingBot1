import threading
import time
import traceback
from datetime import datetime
from pro_features import ai_market_brief_text
from image_utils import build_signal_image, safe_send_with_image
import os

# internal state
_scheduler_threads = {}  # chat_id -> thread
_stop_events = {}        # chat_id -> threading.Event

def _run_periodic_job(chat_id, interval_hours=4, use_image=True):
    """
    Periodic job to fetch AI market brief and send to Telegram.
    """
    interval_sec = interval_hours * 3600
    print(f"[Scheduler] Started for chat {chat_id}, every {interval_hours}h")
    
    stop_event = _stop_events[chat_id]

    while not stop_event.is_set():
        try:
            now = datetime.utcnow()
            brief_text = ai_market_brief_text()
            if use_image:
                img_buf = build_signal_image({"symbol":"Market Brief","interval":"--","signal":"--","entry":"--",
                                              "SL":"--","TP1":"--","confidence":1,"risk_usd":0,"reasons":[brief_text]})
            else:
                img_buf = None

            # Send via Telegram
            import telebot
            bot_token = os.getenv("BOT_TOKEN")
            bot = telebot.TeleBot(bot_token, parse_mode="HTML")
            safe_send_with_image(bot, chat_id, f"🤖 Boss Destiny AI Market Brief ({now.isoformat()})", image_buf=img_buf)
        except Exception as e:
            print(f"[Scheduler] Error sending market brief: {e}")
            traceback.print_exc()

        # Sleep in small intervals to respond quickly to stop
        for _ in range(int(interval_sec)):
            if stop_event.is_set():
                break
            time.sleep(1)

    print(f"[Scheduler] Stopped for chat {chat_id}")

def start_scheduler(chat_id, interval_hours=4, use_image=True):
    """
    Start a scheduler job for a specific chat.
    """
    if chat_id in _scheduler_threads and _scheduler_threads[chat_id].is_alive():
        return f"[Scheduler] Already running for chat {chat_id}"
    
    stop_event = threading.Event()
    _stop_events[chat_id] = stop_event
    thread = threading.Thread(target=_run_periodic_job, args=(chat_id, interval_hours, use_image), daemon=True)
    _scheduler_threads[chat_id] = thread
    thread.start()
    return f"[Scheduler] Started for chat {chat_id}, interval: {interval_hours}h"

def stop_scheduler(chat_id=None):
    """
    Stop scheduler job(s). If chat_id is None, stop all.
    """
    if chat_id:
        if chat_id in _stop_events:
            _stop_events[chat_id].set()
            return f"[Scheduler] Stop signal sent for chat {chat_id}"
        return f"[Scheduler] No scheduler running for chat {chat_id}"
    else:
        for cid, event in _stop_events.items():
            event.set()
        return "[Scheduler] Stop signal sent for all chat jobs"

def list_running_schedulers():
    """
    List all currently running scheduler chat IDs.
    """
    running = [cid for cid, t in _scheduler_threads.items() if t.is_alive()]
    return running
