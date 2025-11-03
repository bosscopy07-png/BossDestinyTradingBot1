# scheduler.py
import threading
import time
import logging
from datetime import datetime

from pro_features import top_gainers_pairs, ai_market_brief_text
from market_providers import get_market_momentum_summary
from signal_engine import summarize_candle_patterns
from storage import load_data, save_data

_logger = logging.getLogger("scheduler")
_scheduler_thread = None
_stop_event = threading.Event()


def _generate_auto_brief():
    """Generates a full automated market brief combining AI + real data."""
    try:
        # Step 1: Fetch top gainers
        gainers_text = top_gainers_pairs(limit=5)

        # Step 2: Get market momentum overview from all timeframes
        momentum_text = get_market_momentum_summary(timeframes=["1m", "5m", "15m", "1h", "4h", "1d"])

        # Step 3: Get candlestick pattern summary (for short + long term)
        candle_summary = summarize_candle_patterns(limit=10)

        # Step 4: Generate AI-based overall brief
        ai_brief = ai_market_brief_text()

        # Combine all data into one message
        full_brief = (
            f"📊 **AUTO MARKET BRIEF ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})**\n\n"
            f"🔥 **Top Gainers:**\n{gainers_text}\n\n"
            f"📈 **Momentum Summary:**\n{momentum_text}\n\n"
            f"🕯️ **Candle Pattern Summary:**\n{candle_summary}\n\n"
            f"🤖 **AI Market Insight:**\n{ai_brief}"
        )

        # Save to local storage for record/history
        history = load_data("market_briefs") or []
        history.append({
            "time": datetime.utcnow().isoformat(),
            "gainers": gainers_text,
            "momentum": momentum_text,
            "candle_patterns": candle_summary,
            "ai_brief": ai_brief
        })
        save_data("market_briefs", history)

        return full_brief

    except Exception as e:
        _logger.exception("Error generating auto brief: %s", e)
        return f"⚠️ Error generating brief: {e}"


def _worker(bot, interval_sec=600):
    """Background worker that auto-sends market briefs periodically."""
    _logger.info("Auto-brief worker started...")
    while not _stop_event.is_set():
        try:
            brief = _generate_auto_brief()
            # Send to admin chat (or replace with your group/channel ID)
            if hasattr(bot, "chat_id"):
                chat_id = int(bot.chat_id)
            else:
                chat_id = bot.get_me().id

            bot.send_message(chat_id, brief, parse_mode="Markdown")

        except Exception as e:
            _logger.exception("Auto-brief error: %s", e)

        _stop_event.wait(interval_sec)

    _logger.info("Auto-brief worker stopped.")


def start_scheduler(bot, interval_sec=600):
    """Start the background scheduler thread."""
    global _scheduler_thread, _stop_event
    if _scheduler_thread and _scheduler_thread.is_alive():
        _logger.info("Scheduler already running")
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_worker, args=(bot, interval_sec), daemon=True)
    _scheduler_thread.start()
    _logger.info(f"Scheduler started with interval {interval_sec}s")


def stop_scheduler():
    """Stop the scheduler thread."""
    global _stop_event
    _stop_event.set()
    _logger.info("Scheduler stop requested")
