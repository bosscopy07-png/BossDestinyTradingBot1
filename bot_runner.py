# bot_runner.py (edited & extended with fallbacks)
import os
import time
import threading
import traceback
import requests
import logging
import json as _json
from datetime import datetime
import telebot
from telebot import types
import re

# ----- Branding and global constants -----
BRAND_TAG = "\n\n— <b>Destiny Trading Empire Bot 💎</b>"

# Config from env (with sane defaults)
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PAIRS = os.getenv(
    "PAIRS",
    "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,MATICUSDT,ADAUSDT",
).split(",")

# scan all timeframes per requirement
SCAN_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
SIGNAL_INTERVAL_DEFAULT = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "1"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "100.0"))

# Auto-send parameters
AUTO_CONFIDENCE_THRESHOLD = float(os.getenv("AUTO_CONFIDENCE_THRESHOLD", "0.90"))  # 0.90 = 90%
AUTO_SEND_ONLY_ADMIN = os.getenv("AUTO_SEND_ONLY_ADMIN", "True").lower() in ("1", "true", "yes")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

# ----- Logging & storage init -----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bot_runner")

# Import optional project modules (if missing we fallback gracefully)
try:
    from market_providers import (
        fetch_trending_pairs_branded,
        fetch_klines_multi,
        get_session,
        fetch_trending_pairs_text,
        analyze_pair_multi_timeframes,
        detect_strong_signals,
        generate_branded_signal_image,
    )
except Exception:
    fetch_trending_pairs_branded = None
    fetch_klines_multi = None
    get_session = None
    fetch_trending_pairs_text = None
    analyze_pair_multi_timeframes = None
    detect_strong_signals = None
    generate_branded_signal_image = None
    logger.exception("market_providers import failed")

try:
    from image_utils import build_signal_image, safe_send_with_image, create_brand_image
except Exception:
    build_signal_image = None
    safe_send_with_image = None
    create_brand_image = None
    logger.exception("image_utils import failed")

# Try to import signal_engine with multiple names (legacy and modern)
try:
    from signal_engine import (
        generate_signal as legacy_generate_signal,
        generate_signal_multi as generate_signal_multi,
        detect_strong_signals as se_detect_strong_signals,
        register_send_callback as register_send_callback,
        start_auto_scanner as se_start_auto_scanner,
        stop_auto_scanner as se_stop_auto_scanner,
    )
except Exception:
    # fallback: import what exists, don't fail hard
    try:
        from signal_engine import generate_signal as legacy_generate_signal
    except Exception:
        legacy_generate_signal = None
    try:
        from signal_engine import generate_signal_multi as generate_signal_multi
    except Exception:
        generate_signal_multi = None
    try:
        from signal_engine import detect_strong_signals as se_detect_strong_signals
    except Exception:
        se_detect_strong_signals = None
    try:
        from signal_engine import register_send_callback as register_send_callback
    except Exception:
        register_send_callback = None
    try:
        from signal_engine import start_auto_scanner as se_start_auto_scanner
    except Exception:
        se_start_auto_scanner = None
    try:
        from signal_engine import stop_auto_scanner as se_stop_auto_scanner
    except Exception:
        se_stop_auto_scanner = None
    logger.exception("signal_engine partial import attempted")

try:
    from storage import ensure_storage, load_data, save_data, record_pnl_screenshot
except Exception:
    ensure_storage = None
    load_data = None
    save_data = None
    record_pnl_screenshot = None
    logger.exception("storage import failed")

# Safe import of ai_client (may expose multiple helpers)
try:
    from ai_client import (
        ai_analysis_text,
        ExchangeStreamer,
        ImageAnalyzer,
        SignalGenerator,
        analyze_image_and_signal,
    )
except Exception:
    ai_analysis_text = None
    ExchangeStreamer = None
    ImageAnalyzer = None
    SignalGenerator = None
    analyze_image_and_signal = None
    logger.exception("ai_client import failed")

try:
    from pro_features import (
        top_gainers_pairs,
        fear_and_greed_index,
        futures_leverage_suggestion,
        quickchart_price_image,
        ai_market_brief_text,
        momentum_and_candle_analysis,
        pro_market_report,
        get_multi_exchange_snapshot,
    )
except Exception:
    top_gainers_pairs = None
    fear_and_greed_index = None
    futures_leverage_suggestion = None
    quickchart_price_image = None
    ai_market_brief_text = None
    momentum_and_candle_analysis = None
    pro_market_report = None
    get_multi_exchange_snapshot = None
    logger.exception("pro_features import failed")

# Scheduler (for auto-briefs)
try:
    from scheduler import start_scheduler, stop_scheduler
except Exception:
    start_scheduler = None
    stop_scheduler = None
    logger.exception("scheduler import failed")

# ------------------------------
# Local fallback storage (when storage.py missing)
# ------------------------------
DATA_PATH = os.getenv("BOT_DATA_PATH", "./bot_data.json")
PNL_DIR = os.getenv("BOT_PNL_DIR", "./pnl_screenshots")


def _ensure_local_storage():
    try:
        os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
        os.makedirs(PNL_DIR, exist_ok=True)
        if not os.path.exists(DATA_PATH):
            with open(DATA_PATH, "w") as fh:
                fh.write(_json.dumps({"signals": [], "pnl": [], "challenge": {"balance": CHALLENGE_START}}))
        return True
    except Exception:
        logger.exception("Local storage ensure failed")
        return False


def _load_local_data():
    try:
        if not os.path.exists(DATA_PATH):
            _ensure_local_storage()
        with open(DATA_PATH, "r") as fh:
            return _json.load(fh)
    except Exception:
        logger.exception("load_data fallback failed")
        return {"signals": [], "pnl": [], "challenge": {"balance": CHALLENGE_START}}


def _save_local_data(d):
    try:
        with open(DATA_PATH, "w") as fh:
            fh.write(_json.dumps(d, indent=2, default=str))
        return True
    except Exception:
        logger.exception("save_data fallback failed")
        return False


def _record_pnl_local(data_bytes, filename_prefix, user_id, caption=None):
    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"{filename_prefix}_{user_id}_{ts}.jpg"
        path = os.path.join(PNL_DIR, fname)
        with open(path, "wb") as fh:
            fh.write(data_bytes)
        # update bot data
        d = load_data() if load_data else _load_local_data()
        entry = {"from": user_id, "file": path, "caption": caption, "time": datetime.utcnow().isoformat(), "linked": None}
        d.setdefault("pnl", []).append(entry)
        if save_data:
            try:
                save_data(d)
            except Exception:
                _save_local_data(d)
        else:
            _save_local_data(d)
        return path
    except Exception:
        logger.exception("record_pnl_local failed")
        return None


# Wire fallback functions if storage module missing
if ensure_storage is None or load_data is None or save_data is None or record_pnl_screenshot is None:
    try:
        # Replace only if missing so original storage module (if present) still used
        if ensure_storage is None:
            ensure_storage = _ensure_local_storage
        if load_data is None:
            load_data = _load_local_data
        if save_data is None:
            save_data = _save_local_data
        if record_pnl_screenshot is None:
            record_pnl_screenshot = _record_pnl_local
        # ensure dirs exist
        _ensure_local_storage()
    except Exception:
        logger.exception("Failed to wire local storage fallbacks")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_last_signal_time = {}  # dict mapping (symbol|interval) -> datetime of last auto-send
_scanner_thread = None
_scanner_stop_event = threading.Event()

# ----- helpers -----
def _append_brand(text: str) -> str:
    if BRAND_TAG.strip() not in text:
        return text + BRAND_TAG
    return text


def _send_branded(chat_id, text, lines_for_image=None, reply_markup=None):
    """
    Always attempt to send a small branded image + caption. If image can't be created, send text only (with brand).
    """
    try:
        caption = _append_brand(text)

        # If create_brand_image available and we have lines, build an image
        if create_brand_image and isinstance(lines_for_image, (list, tuple)) and lines_for_image:
            try:
                img = create_brand_image(lines_for_image, title="Destiny Trading Empire Bot 💎")
                if safe_send_with_image:
                    safe_send_with_image(bot, chat_id, caption, img, reply_markup=reply_markup)
                    return
                else:
                    try:
                        img.seek(0)
                    except Exception:
                        pass
                    bot.send_photo(chat_id, img, caption=caption, reply_markup=reply_markup)
                    return
            except Exception:
                logger.exception("create_brand_image failed; falling back to text")

        # fallback: if there is an image buffer passed directly via safe_send_with_image
        if safe_send_with_image and isinstance(lines_for_image, (bytes, bytearray)):
            safe_send_with_image(bot, chat_id, caption, lines_for_image, reply_markup=reply_markup)
            return

        # final fallback: send text message
        bot.send_message(chat_id, caption, reply_markup=reply_markup)
    except Exception:
        logger.exception("Failed to _send_branded")
        try:
            bot.send_message(chat_id, _append_brand("⚠️ Failed to deliver message."))
        except Exception:
            pass


def stop_existing_bot_instances():
    """Try clear pending getUpdates sessions to reduce 409 conflicts."""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset=-1"
        requests.get(url, timeout=5)
        logger.info("[BOT] Attempted to stop other bot sessions (getUpdates offset -1).")
    except Exception as e:
        logger.warning(f"[BOT] Could not call Telegram getUpdates clear: {e}")


def can_send_signal(symbol: str, interval: str) -> bool:
    """Respect cooldown per symbol+interval (auto-sends)."""
    key = f"{symbol}|{interval}"
    last = _last_signal_time.get(key)
    if not last:
        return True
    return (datetime.utcnow() - last).total_seconds() > COOLDOWN_MIN * 60


def mark_signal_sent(symbol: str, interval: str):
    key = f"{symbol}|{interval}"
    _last_signal_time[key] = datetime.utcnow()


def compute_risk_and_size(entry: float, sl: float, balance: float, risk_percent: float):
    risk_amount = (balance * risk_percent) / 100.0
    diff = None
    try:
        diff = abs(entry - sl)
    except Exception:
        diff = None
    if not diff or diff <= 1e-12:
        return round(risk_amount, 8), 0.0
    pos_size = risk_amount / diff
    return round(risk_amount, 8), round(pos_size, 8)


# ----- small background scanner wrappers -----
def _internal_scanner_loop(pairs, interval, exchanges, min_confidence, poll_seconds, use_ai):
    """
    Fallback scanner used only when se_start_auto_scanner is not available.
    """
    logger.info("Fallback scanner started for pairs=%s interval=%s", pairs, interval)
    while not _scanner_stop_event.is_set():
        try:
            for p in pairs:
                if _scanner_stop_event.is_set():
                    break
                try:
                    # try to use generate_signal_multi if available
                    if generate_signal_multi:
                        res = generate_signal_multi(p, interval, exchanges=exchanges, min_confidence_for_signal=min_confidence, use_ai_explain=use_ai)
                    else:
                        # fallback to _safe_generate_signal
                        res = _safe_generate_signal(p, interval)
                    if not isinstance(res, dict):
                        continue
                    conf = float(res.get("confidence", 0.0))
                    is_strong = (conf >= min_confidence) and (res.get("signal") in ("LONG", "SHORT"))
                    consensus = bool(res.get("consensus")) if isinstance(res.get("consensus"), bool) else False
                    if is_strong and (consensus or conf >= min_confidence):
                        logger.info("Fallback scanner: strong signal %s %s %.3f", p, res.get("signal"), conf)
                        # Try to send
                        try:
                            if AUTO_SEND_ONLY_ADMIN and ADMIN_ID:
                                record_signal_and_send(res, chat_id=ADMIN_ID, user_id=ADMIN_ID, auto=True)
                            elif not AUTO_SEND_ONLY_ADMIN:
                                record_signal_and_send(res, chat_id=ADMIN_ID, user_id=ADMIN_ID, auto=True)
                        except Exception:
                            logger.exception("Fallback scanner: failed to deliver signal")
                except Exception:
                    logger.exception("Fallback scanner: per-pair error")
            # sleep with interruption checks
            for _ in range(int(poll_seconds)):
                if _scanner_stop_event.is_set():
                    break
                time.sleep(1)
        except Exception:
            logger.exception("Fallback scanner loop exception, continuing")
    logger.info("Fallback scanner stopped.")


def start_background_scanner(pairs=None, interval: str = SIGNAL_INTERVAL_DEFAULT, exchanges=None, min_confidence: float = AUTO_CONFIDENCE_THRESHOLD, poll_seconds: int = 300, use_ai: bool = False):
    """
    Start a background scanner. Prefer signal_engine.start_auto_scanner if present, otherwise use fallback.
    """
    global _scanner_thread, _scanner_stop_event
    if _scanner_thread and _scanner_thread.is_alive():
        logger.warning("Scanner already running")
        return False

    _scanner_stop_event.clear()
    exs = exchanges or ["binance", "bybit", "kucoin", "okx"]
    ps = pairs or PAIRS

    if se_start_auto_scanner:
        try:
            ok = se_start_auto_scanner(ps, interval, exchanges=exs, min_confidence=min_confidence, poll_seconds=poll_seconds, use_ai=use_ai)
            logger.info("signal_engine scanner started: %s", ok)
            return ok
        except Exception:
            logger.exception("se_start_auto_scanner failed; falling back to internal scanner")

    _scanner_thread = threading.Thread(target=_internal_scanner_loop, args=(ps, interval, exs, min_confidence, poll_seconds, use_ai), daemon=True)
    _scanner_thread.start()
    logger.info("Fallback scanner thread started")
    return True


def stop_background_scanner(timeout: int = 5):
    """
    Stop background scanner. Prefer signal_engine.stop_auto_scanner if present.
    """
    global _scanner_thread, _scanner_stop_event
    if se_stop_auto_scanner:
        try:
            ok = se_stop_auto_scanner(timeout=timeout)
            logger.info("signal_engine.stop_auto_scanner returned: %s", ok)
            return ok
        except Exception:
            logger.exception("se_stop_auto_scanner failed; falling back to internal stop")

    if not _scanner_thread or not _scanner_thread.is_alive():
        return False
    _scanner_stop_event.set()
    _scanner_thread.join(timeout)
    if _scanner_thread.is_alive():
        logger.warning("Fallback scanner thread did not stop within timeout")
        return False
    logger.info("Fallback scanner stopped cleanly")
    return True


# ----- signal generation integration -----
def _safe_generate_signal(symbol: str, interval: str):
    """
    Primary path: use analyze_pair_multi_timeframes() from market_providers to get a multi-TF analysis.
    If that fails, fallback to fetching klines and legacy signal_engine.
    Returns a standardized dict:
      { symbol, interval, signal, entry, sl, tp1, confidence, reasons }
    """
    try:
        # 1) Try the multi-TF analyzer (best)
        if analyze_pair_multi_timeframes:
            try:
                res = analyze_pair_multi_timeframes(
                    symbol, timeframes=[interval] + [tf for tf in SCAN_INTERVALS if tf != interval]
                )
                if isinstance(res, dict) and not res.get("error"):
                    combined = res.get("combined_signal", "HOLD")
                    score = float(res.get("combined_score", 0.0))

                    # pick entry/sl/tp1 from analysis if available
                    entry = None
                    try:
                        entry = (
                            res["analysis"].get(interval, {}).get("close")
                            or next(iter(res["analysis"].values())).get("close")
                        )
                    except Exception:
                        entry = None

                    sl = None
                    tp1 = None
                    try:
                        sl = (
                            res["analysis"].get("1h", {}).get("sl")
                            or res["analysis"].get(interval, {}).get("sl")
                        )
                        tp1 = (
                            res["analysis"].get("1h", {}).get("tp1")
                            or res["analysis"].get(interval, {}).get("tp1")
                        )
                    except Exception:
                        pass

                    reasons = []
                    try:
                        for tf, info in res.get("analysis", {}).items():
                            if isinstance(info, dict):
                                reasons.extend(info.get("reasons", []))
                    except Exception:
                        pass

                    # Optional AI augmentation
                    if ai_analysis_text and score > 0.4 and callable(ai_analysis_text):
                        try:
                            res_text = _json.dumps(res, indent=2) if not isinstance(res, str) else res
                            prompt = (
                                "Given the following multi-timeframe analysis for "
                                f"{symbol} on {interval}:\n\n{res_text}\n\n"
                                "Provide concise trade rationale and, if appropriate, a trust modifier between -0.05 and +0.1 to adjust confidence. "
                                "Reply with a short text containing a numeric modifier if needed."
                            )
                            ai_resp = ai_analysis_text(prompt)
                            # ai_analysis_text may return dict with 'analysis' or a string
                            ai_text = ""
                            if isinstance(ai_resp, dict):
                                ai_text = ai_resp.get("analysis") or ""
                            else:
                                ai_text = str(ai_resp)
                            m = re.search(r"([+-]?[0-9]*\.?[0-9]+)", ai_text)
                            if m:
                                mod = float(m.group(1))
                                score = max(0.0, min(1.0, score + mod))
                                reasons.append("ai_adj")
                        except Exception:
                            logger.exception("AI augmentation failed")

                    return {
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "signal": "LONG"
                        if "LONG" in combined or "STRONG_LONG" in combined
                        else ("SHORT" if "SHORT" in combined or "STRONG_SHORT" in combined else "HOLD"),
                        "entry": float(entry) if entry else None,
                        "sl": float(sl) if sl else None,
                        "tp1": float(tp1) if tp1 else None,
                        "confidence": float(score),
                        "reasons": list(dict.fromkeys(reasons)),
                    }
            except Exception:
                logger.exception("analyze_pair_multi_timeframes failed; falling back")

        ## 2) Fallback: try to fetch klines for exchange choices and run legacy generate_signal
        exchanges_to_try = ["binance", "bybit", "kucoin", "okx"]
        if fetch_klines_multi:
            for ex in exchanges_to_try:
                try:
                    df = fetch_klines_multi(symbol, interval, limit=200, exchange=ex)
                    if df is None:
                        continue
                    if hasattr(df, "empty") and df.empty:
                        continue
                    if "close" not in df:
                        continue

                    if legacy_generate_signal:
                        try:
                            out = (
                                legacy_generate_signal(df, pair=symbol)
                                if callable(legacy_generate_signal)
                                else legacy_generate_signal
                            )
                            sig_text = str(out)
                            if "BUY" in sig_text.upper() or "STRONG BUY" in sig_text.upper():
                                sig = "LONG"
                            elif "SELL" in sig_text.upper():
                                sig = "SHORT"
                            else:
                                sig = "HOLD"

                            # basic SL/TP using ATR-like estimation
                            try:
                                highs = df["high"].astype(float)
                                lows = df["low"].astype(float)
                                closes = df["close"].astype(float)
                                atr_val = (highs - lows).rolling(14).mean().iloc[-1]
                                last = float(closes.iloc[-1])
                                if sig == "LONG":
                                    sl = last - (atr_val * 1.5)
                                    tp1 = last + (atr_val * 1.5)
                                elif sig == "SHORT":
                                    sl = last + (atr_val * 1.5)
                                    tp1 = last - (atr_val * 1.5)
                                else:
                                    sl = last * 0.995
                                    tp1 = last * 1.005
                            except Exception:
                                sl = None
                                tp1 = None

                            return {
                                "symbol": symbol.upper(),
                                "interval": interval,
                                "signal": sig,
                                "entry": float(closes.iloc[-1]),
                                "sl": float(sl) if sl else None,
                                "tp1": float(tp1) if tp1 else None,
                                "confidence": 0.3,
                                "reasons": [sig_text],
                            }
                        except Exception:
                            logger.exception("legacy_generate_signal failed on df")
                            continue
                except Exception:
                    logger.exception("fetch_klines_multi attempt failed")
                    continue
            return {"error": "no_data_on_exchanges"}
        else:
            return {"error": "no_fetch_klines_available"}
    except Exception as exc:
        logger.exception("_safe_generate_signal unexpected error")
        return {"error": str(exc)}


# ----- recording & messaging -----
def record_signal_and_send(sig: dict, chat_id=None, user_id=None, auto=False):
    """Record a signal in storage and send it (image + caption)."""
    # storage read
    try:
        d = load_data() if load_data else {}
    except Exception:
        d = {}

    sig_id = f"S{int(time.time())}"
    balance = d.get("challenge", {}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START

    # risk and pos
    risk_amt, pos_size = compute_risk_and_size(sig.get("entry") or 0.0, sig.get("sl") or 0.0, balance, RISK_PERCENT)

    rec = {
        "id": sig_id,
        "signal": sig,
        "time": datetime.utcnow().isoformat(),
        "risk_amt": risk_amt,
        "pos_size": pos_size,
        "user": user_id or ADMIN_ID,
        "auto": bool(auto),
    }

    # save record
    try:
        if isinstance(d, dict):
            d.setdefault("signals", []).append(rec)
            d.setdefault("stats", {})
            d["stats"]["total_signals"] = d["stats"].get("total_signals", 0) + 1
            try:
                if save_data:
                    save_data(d)
                else:
                    _save_local_data(d)
            except Exception:
                _save_local_data(d)
    except Exception:
        logger.exception("Failed to save signal record")

    # Build caption
    try:
        confidence_pct = int(sig.get("confidence", 0) * 100)
    except Exception:
        confidence_pct = 0

    caption = (
        "🔔 <b>Destiny Trading Empire — Signal</b>\n\n"
        f"ID: {sig_id} Pair: {sig.get('symbol')} | TF: {sig.get('interval')}\n"
        f"Signal: <b>{sig.get('signal')}</b>\n"
        f"Entry: {sig.get('entry') or 'N/A'} | SL: {sig.get('sl') or 'N/A'} | TP1: {sig.get('tp1') or 'N/A'}\n"
        f"Confidence: {confidence_pct}% | Risk (USD): {risk_amt}\n"
        f"Reasons: {', '.join(sig.get('reasons', []) or ['None'])}"
    )
    caption = _append_brand(caption)

    # Image creation
    img = None
    try:
        if generate_branded_signal_image and isinstance(sig, dict):
            try:
                img_buf, _ = generate_branded_signal_image(
                    {
                        "symbol": sig.get("symbol"),
                        "analysis": None,
                        "combined_score": sig.get("confidence", 0),
                        "combined_signal": sig.get("signal"),
                        "sl": sig.get("sl"),
                        "tp1": sig.get("tp1"),
                        "image": None,
                        "caption_lines": [f"{sig.get('symbol')} | {sig.get('interval')} | {sig.get('signal')}"],
                    }
                )
                if img_buf:
                    img = img_buf
            except Exception:
                logger.exception("generate_branded_signal_image failed")

        if not img and build_signal_image:
            try:
                img = build_signal_image(sig)
            except Exception:
                logger.exception("build_signal_image failed")
    except Exception:
        logger.exception("build_signal_image wrapper failed")

    # Keyboard for message
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📷 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🔁 Share", switch_inline_query=f"{sig.get('symbol')}"))

    # send (use safe_send_with_image if available)
    try:
        if safe_send_with_image:
            safe_send_with_image(bot, chat_id or ADMIN_ID, caption, img, kb)
        else:
            if img:
                try:
                    img.seek(0)
                except Exception:
                    pass
                bot.send_photo(chat_id or ADMIN_ID, img, caption=caption, reply_markup=kb)
            else:
                bot.send_message(chat_id or ADMIN_ID, caption, reply_markup=kb)
    except Exception:
        logger.exception("Failed to send signal message")

    # optionally send a quick AI rationale follow-up if available
    try:
        if ai_analysis_text and sig and not sig.get("error"):
            prompt = f"Provide concise trade rationale for this signal:\n\n{_json.dumps(sig, default=str)}"
            ai_text = None
            try:
                resp = ai_analysis_text(prompt) if callable(ai_analysis_text) else None
                if isinstance(resp, dict):
                    ai_text = resp.get("analysis") or str(resp)
                else:
                    ai_text = str(resp)
            except Exception:
                ai_text = None
            if ai_text:
                follow = _append_brand(f"🤖 AI Rationale: {ai_text}")
                try:
                    bot.send_message(chat_id or ADMIN_ID, follow)
                except Exception:
                    logger.exception("Failed to send AI follow-up")
    except Exception:
        logger.exception("AI rationale follow-up failed")

    # mark as sent for cooldown rules
    try:
        mark_signal_sent(sig.get("symbol") or "UNKNOWN", sig.get("interval") or SIGNAL_INTERVAL_DEFAULT)
    except Exception:
        pass

    return sig_id


# ----- keyboard UI -----
def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📈 Get Signals", callback_data="get_signal"),
           types.InlineKeyboardButton("🔎 Scan Top 4", callback_data="scan_top4"))
    kb.add(types.InlineKeyboardButton("⚙️ Bot Status", callback_data="bot_status"),
           types.InlineKeyboardButton("🚀 Trending Pairs", callback_data="trending"))
    kb.add(types.InlineKeyboardButton("📰 Market News", callback_data="market_news"),
           types.InlineKeyboardButton("📃 My Challenge", callback_data="challenge_status"))
    kb.add(types.InlineKeyboardButton("📷 Upload PnL", callback_data="pnl_upload"),
           types.InlineKeyboardButton("📋 History", callback_data="history"))
    kb.add(types.InlineKeyboardButton("🤖 AI Market Brief", callback_data="ask_ai"),
           types.InlineKeyboardButton("🔄 Refresh Bot", callback_data="refresh_bot"))
    kb.add(types.InlineKeyboardButton("▶ Start Auto Scanner", callback_data="start_auto_brief"),
           types.InlineKeyboardButton("⏹ Stop Auto Scanner", callback_data="stop_auto_brief"))
    kb.add(types.InlineKeyboardButton("📣 Start Auto Briefs", callback_data="start_auto_brief_scheduler"),
           types.InlineKeyboardButton("⛔ Stop Auto Briefs", callback_data="stop_auto_brief_scheduler"))
    return kb


# --- Telegram handlers ---
@bot.message_handler(commands=["start", "menu"])
def cmd_start(msg):
    try:
        # always reply with main keyboard and branding image if available
        text = "👋 Welcome Boss Destiny!\n\nThis is your Trading Empire control panel."
        lines = [
            "Welcome — Destiny Trading Empire Bot 💎",
            "Use the buttons to get signals, start scanners, view trending pairs.",
        ]
        if create_brand_image:
            try:
                img = create_brand_image(lines, title="Destiny Trading Empire Bot 💎")
                if safe_send_with_image:
                    safe_send_with_image(bot, msg.chat.id, _append_brand(text), img, reply_markup=main_keyboard())
                    return
                else:
                    try:
                        img.seek(0)
                    except Exception:
                        pass
                    bot.send_photo(msg.chat.id, img, caption=_append_brand(text), reply_markup=main_keyboard())
                    return
            except Exception:
                logger.exception("Failed creating welcome brand image")
        bot.send_message(msg.chat.id, _append_brand(text), reply_markup=main_keyboard())
    except Exception:
        logger.exception("cmd_start failed")


@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(fi.file_path)
        # try to use configured record_pnl_screenshot or fallback
        saved_path = None
        try:
            if record_pnl_screenshot and callable(record_pnl_screenshot):
                # some recorders expect (bytes, name, user, caption)
                saved = record_pnl_screenshot(data, datetime.utcnow().strftime("%Y%m%d_%H%M%S"), message.from_user.id, message.caption)
                saved_path = saved if isinstance(saved, str) else None
            else:
                # fallback: write locally
                saved_path = _record_pnl_local(data, "pnl", message.from_user.id, caption=message.caption)
        except Exception:
            logger.exception("photo save fallback failed")
            saved_path = _record_pnl_local(data, "pnl", message.from_user.id, caption=message.caption)

        bot.reply_to(message, _append_brand(f"Saved screenshot. File: {saved_path or 'unknown'}. Reply with `#link <signal_id> TP1` or `#link <signal_id> SL`"))
    except Exception:
        logger.exception("photo_handler failed")
        bot.reply_to(message, _append_brand("Failed to save screenshot."))


@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_handler(message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 3:
            bot.reply_to(message, _append_brand("Usage: #link <signal_id> TP1 or SL"))
            return
        sig_id, tag = parts[1], parts[2].upper()
        d = load_data() if load_data else {}
        pnl_item = (
            next(
                (p for p in reversed(d.get("pnl", [])) if p.get("linked") is None and p["from"] == message.from_user.id),
                None,
            )
            if isinstance(d, dict)
            else None
        )
        if not pnl_item:
            bot.reply_to(message, _append_brand("No unlinked screenshot found."))
            return
        pnl_item["linked"] = {"signal_id": sig_id, "result": tag, "linked_by": message.from_user.id}
        # only admin confirmation updates balance
        if message.from_user.id == ADMIN_ID:
            srec = next((s for s in d.get("signals", []) if s["id"] == sig_id), None)
            if srec:
                risk = srec.get("risk_amt", 0)
                if tag.startswith("TP"):
                    d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) + risk
                    d["challenge"]["wins"] = d["challenge"].get("wins", 0) + 1
                    d["stats"]["wins"] = d["stats"].get("wins", 0) + 1
                elif tag == "SL":
                    d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) - risk
                    d["challenge"]["losses"] = d["challenge"].get("losses", 0) + 1
                    d["stats"]["losses"] = d["stats"].get("losses", 0) + 1
        try:
            if save_data:
                save_data(d)
            else:
                _save_local_data(d)
        except Exception:
            _save_local_data(d)
        bot.reply_to(message, _append_brand(f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance."))
    except Exception:
        logger.exception("link_handler failed")
        bot.reply_to(message, _append_brand("Failed to link screenshot."))


# ----- Callback actions -----
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        cid = call.message.chat.id
        data = call.data
        # make sure to answer callback quickly to avoid spinner
        try:
            bot.answer_callback_query(call.id)
        except Exception:
            pass

        # Choose pair -> produce inline keyboard of PAIRS
        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"sig_{p}"))
            bot.send_message(cid, _append_brand("Choose pair to analyze:"), reply_markup=kb)
            return

        # individual pair selected
        if data.startswith("sig_"):
            pair = data.split("_", 1)[1]
            bot.send_chat_action(cid, "typing")
            sig = _safe_generate_signal(pair, SIGNAL_INTERVAL_DEFAULT)
            if sig.get("error"):
                # if no data try again with a different exchange/timeframe message
                err = sig.get("error")
                bot.send_message(cid, _append_brand(f"Error generating signal: {err}\nTry another timeframe or check pair symbol."))
                return
            # Standardize missing fields
            sig.setdefault("interval", SIGNAL_INTERVAL_DEFAULT)
            sig.setdefault("symbol", pair.upper())
            record_signal_and_send(sig, chat_id=cid, user_id=call.from_user.id, auto=False)
            return

        # quick scan top X
        if data == "scan_top4":
            bot.send_message(cid, _append_brand("🔎 Scanning top pairs across exchanges..."))
            for p in PAIRS[:6]:
                try:
                    if can_send_signal(p, SIGNAL_INTERVAL_DEFAULT):
                        sig = _safe_generate_signal(p, SIGNAL_INTERVAL_DEFAULT)
                        if not sig.get("error") and sig.get("signal") in ("LONG", "SHORT"):
                            record_signal_and_send(sig, chat_id=cid)
                except Exception:
                    logger.exception("scan_top4 subtask failed")
            return

        if data == "trending":
            bot.send_message(cid, _append_brand("📡 Fetching multi-exchange trending pairs... please wait."))
            try:
                # support multi-exchange aggregation if detect_strong_signals available
                if fetch_trending_pairs_branded:
                    img_buf, caption = fetch_trending_pairs_branded(limit=8)
                    if img_buf:
                        if safe_send_with_image:
                            safe_send_with_image(bot, cid, _append_brand(caption), img_buf)
                        else:
                            try:
                                img_buf.seek(0)
                            except Exception:
                                pass
                            bot.send_photo(cid, img_buf, caption=_append_brand(caption))
                    else:
                        bot.send_message(cid, _append_brand(caption))
                elif fetch_trending_pairs_text:
                    bot.send_message(cid, _append_brand(fetch_trending_pairs_text()))
                else:
                    bot.send_message(cid, _append_brand("Trending feature not available (missing market_providers)."))
            except Exception:
                logger.exception("trending handler failed")
                bot.send_message(cid, _append_brand("Failed to fetch trending pairs."))
            return

        if data == "market_news":
            # Try to produce something useful: F&G + top movers + short AI brief if available
            try:
                lines = []
                if fear_and_greed_index:
                    try:
                        fg = fear_and_greed_index()
                        lines.append(fg)
                    except Exception:
                        pass
                if top_gainers_pairs:
                    try:
                        t = top_gainers_pairs(limit=5)
                        lines.append(t)
                    except Exception:
                        pass
                # if we have ai_market_brief_text and a popular symbol
                if ai_market_brief_text and PAIRS:
                    try:
                        brief = ai_market_brief_text(PAIRS[0], exchanges=["binance", "bybit", "kucoin"])
                        lines.append("AI Brief:\n" + brief)
                    except Exception:
                        pass
                if lines:
                    bot.send_message(cid, _append_brand("\n\n".join(lines)))
                else:
                    bot.send_message(cid, _append_brand("📰 Market news: feature not available (no providers configured)."))
            except Exception:
                logger.exception("market_news failed")
                bot.send_message(cid, _append_brand("Failed to fetch market news."))
            return

        if data == "bot_status":
            # provide brief health info and whether scanner is running
            scanner_running = _scanner_thread is not None and _scanner_thread.is_alive()
            msg = (
                "⚙️ Bot is running ✅\n"
                f"Scanner running: {scanner_running}\n"
                f"Auto confidence threshold: {AUTO_CONFIDENCE_THRESHOLD*100:.0f}%"
            )
            bot.send_message(cid, _append_brand(msg))
            return

        if data == "challenge_status":
            d = load_data() if load_data else {}
            bal = d.get("challenge", {}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START
            wins = d.get("challenge", {}).get("wins", 0) if isinstance(d, dict) else 0
            losses = d.get("challenge", {}).get("losses", 0) if isinstance(d, dict) else 0
            bot.send_message(cid, _append_brand(f"Balance: ${bal:.2f}\nWins: {wins} Losses: {losses}"))
            return

        if data == "ask_ai":
            bot.send_message(cid, _append_brand("🤖 Ask AI: send a message starting with `AI:` followed by your question."))
            return

        if data == "refresh_bot":
            bot.send_message(cid, _append_brand("🔄 Refreshing bot session..."))
            stop_existing_bot_instances()
            time.sleep(2)
            bot.send_message(cid, _append_brand("✅ Refreshed."))
            return

        if data == "start_auto_brief":
            bot.send_message(cid, _append_brand("▶ Starting background market scanner (auto-send strong signals)."))
            start_background_scanner()
            return

        if data == "stop_auto_brief":
            bot.send_message(cid, _append_brand("⏹ Stopping background market scanner."))
            stop_background_scanner()
            return

        # Scheduler-based auto briefs (text/AI summaries)
        if data == "start_auto_brief_scheduler":
            if start_scheduler:
                bot.send_message(cid, _append_brand("▶ Scheduler for auto-briefs enabled. You will receive periodic market briefs."))
                try:
                    start_scheduler(bot)
                except Exception:
                    logger.exception("start_scheduler failed")
            else:
                bot.send_message(cid, _append_brand("Scheduler not available (missing scheduler module)."))
            return

        if data == "stop_auto_brief_scheduler":
            if stop_scheduler:
                stop_scheduler()
                bot.send_message(cid, _append_brand("⏹ Scheduler for auto-briefs disabled."))
            else:
                bot.send_message(cid, _append_brand("Scheduler not available (missing scheduler module)."))
            return

        if data == "pnl_upload":
            bot.send_message(cid, _append_brand("📷 Upload PnL: send a photo of your PnL now. Use /menu to return."))
            return

        if data == "history":
            try:
                d = load_data() if load_data else {}
                sigs = d.get("signals", []) if isinstance(d, dict) else []
                last = sigs[-5:] if len(sigs) else []
                if not last:
                    bot.send_message(cid, _append_brand("No signal history yet."))
                    return
                lines = []
                for s in reversed(last):
                    sig = s.get("signal", {})
                    lines.append(f"{s['id']}: {sig.get('symbol')} {sig.get('signal')} ({sig.get('confidence')*100 if sig.get('confidence') else 0:.0f}%) at {s.get('time')}")
                bot.send_message(cid, _append_brand("Recent Signals:\n" + "\n".join(lines)))
            except Exception:
                logger.exception("history fetch failed")
                bot.send_message(cid, _append_brand("Failed to fetch history."))
            return

        if data.startswith("ai_"):
            sig_id = data.split("_", 1)[1]
            d = load_data() if load_data else {}
            rec = next((s for s in d.get("signals", []) if s["id"] == sig_id), None) if isinstance(d, dict) else None
            if not rec:
                bot.send_message(cid, _append_brand("Signal not found"))
                return
            prompt = f"Provide trade rationale, risk controls and a recommended leverage for this trade:\n{rec['signal']}"
            ai_text = None
            try:
                resp = ai_analysis_text(prompt) if callable(ai_analysis_text) else None
                if isinstance(resp, dict):
                    ai_text = resp.get("analysis") or str(resp)
                else:
                    ai_text = str(resp) if resp else "AI feature not available"
            except Exception:
                ai_text = "AI feature error"
            bot.send_message(cid, _append_brand(f"🤖 AI analysis:\n{ai_text}"))
            return

        bot.send_message(cid, _append_brand("Unknown action"))
    except Exception:
        logger.exception("callback_handler failed")
        try:
            bot.answer_callback_query(call.id, "Handler error")
        except Exception:
            pass


# ----- simple AI text handler for messages starting with "AI:" -----
@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().upper().startswith("AI:"))
def ai_message_handler(message):
    try:
        prompt = message.text.strip()[3:].strip()
        if not prompt:
            bot.reply_to(message, _append_brand("Please provide a question after `AI:`"))
            return
        if not ai_analysis_text or not callable(ai_analysis_text):
            bot.reply_to(message, _append_brand("AI features not configured on this deployment."))
            return
        bot.send_chat_action(message.chat.id, "typing")
        try:
            resp = ai_analysis_text(prompt)
            if isinstance(resp, dict):
                out = resp.get("analysis") or str(resp)
            else:
                out = str(resp)
            bot.reply_to(message, _append_brand(f"🤖 AI Reply:\n{out}"))
        except Exception:
            logger.exception("ai_message_handler failed")
            bot.reply_to(message, _append_brand("AI service error."))
    except Exception:
        logger.exception("ai_message_handler top-level failed")
        bot.reply_to(message, _append_brand("Failed to process AI request."))


# If this file is executed directly allow a small self-test message (optional)
if __name__ == "__main__":
    print("bot_runner.py loaded - sanity check")
    logger.info("bot_runner loaded. Ready.")
    
