# bot_runner.py (complete updated)
import os
import time
import threading
import traceback
import requests
import logging
from datetime import datetime
import telebot
from telebot import types
import json
import re
from io import BytesIO

# Attempt to import Pillow for image creation (text -> image)
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except Exception:
    HAS_PIL = False

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

# ensure storage directory/data if module available
if ensure_storage:
    try:
        ensure_storage()
    except Exception:
        logger.exception("ensure_storage failed")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_last_signal_time = {}  # dict mapping (symbol|interval) -> datetime of last auto-send
_scanner_thread = None
_scanner_stop_event = threading.Event()


# ---------------------------
# Helper: build brand image (fallback)
# ---------------------------
def _local_create_brand_image(lines, title=None, chart_bytes: bytes = None, width=900, height=360):
    """
    Create a simple branded image using Pillow from the given lines (list of str).
    Returns BytesIO of PNG. If Pillow missing, return None.
    """
    if not HAS_PIL:
        return None
    try:
        # Basic layout: chart area left (if chart_bytes), text area right
        img = Image.new("RGB", (width, height), color=(12, 18, 28))
        draw = ImageDraw.Draw(img)

        # fonts (use default if not available)
        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            title_font = ImageFont.truetype(font_path, 22)
            text_font = ImageFont.truetype(font_path, 16)
            small_font = ImageFont.truetype(font_path, 12)
        except Exception:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        padding = 12
        x = padding
        y = padding

        # draw title
        t = title or "Destiny Trading Empire"
        draw.text((x, y), t, font=title_font, fill=(255, 215, 0))
        y += 32

        # draw lines (cap to some lines that fit)
        for li in lines[:8]:
            draw.text((x, y), li, font=text_font, fill=(230, 230, 230))
            y += 20

        # add brand label bottom-right
        brand_text = "Destiny Trading Empire Bot 💎"
        tw, th = draw.textsize(brand_text, font=small_font)
        draw.text((width - tw - padding, height - th - padding), brand_text, font=small_font, fill=(160, 160, 160))

        # If chart provided, paste it on left (resize as needed)
        if chart_bytes:
            try:
                ch = Image.open(BytesIO(chart_bytes)).convert("RGB")
                ch_w = int(width * 0.42)
                ch_h = height - 2 * padding
                ch = ch.resize((ch_w, ch_h))
                img.paste(ch, (width - ch_w - padding, padding))
            except Exception:
                pass

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:
        logger.exception("Local brand image creation failed")
        return None


# expose create_brand_image either from image_utils or local fallback
if create_brand_image is None:
    create_brand_image = _local_create_brand_image

# safe_send_with_image wrapper (if not present)
def _safe_send_with_image(bot_inst, chat_id, caption, img_buf, reply_markup=None):
    try:
        if hasattr(img_buf, "seek"):
            img_buf.seek(0)
        bot_inst.send_photo(chat_id, img_buf, caption=caption, reply_markup=reply_markup)
    except Exception:
        try:
            # sometimes Telegram expects file-like with filename
            bot_inst.send_photo(chat_id, img_buf, caption=caption, reply_markup=reply_markup)
        except Exception:
            logger.exception("safe_send_with_image failed, falling back to text")
            try:
                bot_inst.send_message(chat_id, caption, reply_markup=reply_markup)
            except Exception:
                pass


if safe_send_with_image is None:
    safe_send_with_image = _safe_send_with_image


# ---------------------------
# Utilities
# ---------------------------
def _append_brand(text: str) -> str:
    if BRAND_TAG.strip() not in text:
        return text + BRAND_TAG
    return text


def _send_branded(chat_id, text, lines_for_image=None, chart_bytes: bytes = None, reply_markup=None):
    """
    Build a branded image (text -> image) and send it. If image creation fails we fallback to text.
    lines_for_image: list of lines used to render image
    """
    try:
        caption = _append_brand("")  # caption left empty because we render text inside image per your requirement
        # Build image content lines (ensure not empty)
        lines = lines_for_image or (text.splitlines() if isinstance(text, str) else [])
        title = "Destiny Trading Empire Bot 💎"
        img_buf = None

        # prefer your project's create_brand_image if it exists (it might return BytesIO / bytes)
        try:
            tmp = create_brand_image(lines, title=title, chart_img=chart_bytes)
            if tmp:
                img_buf = tmp
        except Exception:
            # try local builder
            img_buf = _local_create_brand_image(lines, title=title, chart_bytes=chart_bytes)

        if img_buf:
            safe_send_with_image(bot, chat_id, caption or _append_brand(""), img_buf, reply_markup=reply_markup)
            return

        # final fallback: send plain text message (rare; you requested images, so prefer image)
        bot.send_message(chat_id, _append_brand(text), reply_markup=reply_markup)
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


# ---------------------------
# Market data helpers (public fallback)
# ---------------------------
def _binance_klines(symbol: str, interval: str = "1h", limit: int = 200):
    """Public Binance klines fallback (no API key)"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        # parse to DataFrame-like list of dicts for compatibility usage (we will return list of dicts)
        rows = []
        for r in raw:
            rows.append({
                "open_time": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            })
        return rows
    except Exception:
        logger.exception("binance_klines failed")
        return None


def fetch_klines_public(symbol: str, interval: str = "1h", limit: int = 200, exchange: str = "binance"):
    """
    Wrapper to return klines in a pandas-like list of dicts:
    - Prefer fetch_klines_multi (your market_providers)
    - Fallback to public binance
    """
    try:
        if fetch_klines_multi:
            try:
                df = fetch_klines_multi(symbol=symbol, interval=interval, limit=limit, exchange=exchange)
                # if returns pandas DataFrame keep it
                return df
            except Exception:
                logger.exception("fetch_klines_multi failed")
        # fallback: use binance public
        return _binance_klines(symbol, interval, limit)
    except Exception:
        logger.exception("fetch_klines_public failed")
        return None


# ---------------------------
# QuickChart helper (generate chart bytes) - uses quickchart public service
# ---------------------------
QUICKCHART_URL = "https://quickchart.io/chart"

def generate_sparkline_bytes(values, label="price", width=800, height=300):
    try:
        cfg = {
            "type": "line",
            "data": {
                "labels": list(range(len(values))),
                "datasets": [{"label": label, "data": values, "fill": False, "pointRadius": 0}]
            },
            "options": {"plugins": {"legend": {"display": False}}, "elements": {"line": {"tension": 0}}}
        }
        params = {"c": json.dumps(cfg), "width": width, "height": height, "devicePixelRatio": 2}
        r = requests.get(QUICKCHART_URL, params=params, timeout=12)
        r.raise_for_status()
        return r.content
    except Exception:
        logger.exception("generate_sparkline_bytes failed")
        return None


# ---------------------------
# AI fallback small local analyzer (if ai_client not present)
# ---------------------------
def _local_ai_analysis(prompt: str):
    # Minimalist deterministic response for when ai_analysis_text isn't configured.
    # Keep it simple and safe.
    try:
        # guess sentiment by some keywords
        up = len(re.findall(r"\b(up|bull|long|rally|breakout)\b", prompt, flags=re.I))
        dn = len(re.findall(r"\b(down|bear|short|fall|reject)\b", prompt, flags=re.I))
        score = 0.5 + (up - dn) * 0.05
        score = max(0.0, min(1.0, score))
        text = f"Local AI summary: approximate confidence {int(score*100)}%. (Limited local analysis — enable ai_client for richer output.)"
        return {"ok": True, "analysis": text}
    except Exception:
        return {"ok": False, "error": "local_ai_failed"}


# ---------------------------
# Signal analysis & integration
# ---------------------------
def _safe_generate_signal(symbol: str, interval: str):
    """
    Primary path: use analyze_pair_multi_timeframes() from market_providers to get a multi-TF analysis.
    If that fails, fallback to fetching klines and legacy signal_engine.
    Returns a standardized dict:
      { symbol, interval, signal, entry, sl, tp1, confidence, reasons }
    """
    try:
        # 1) Try your project's multi-TF analyzer (best)
        if analyze_pair_multi_timeframes:
            try:
                res = analyze_pair_multi_timeframes(symbol, timeframes=[interval] + [tf for tf in SCAN_INTERVALS if tf != interval])
                if isinstance(res, dict) and not res.get("error"):
                    combined = res.get("combined_signal", "HOLD")
                    score = float(res.get("combined_score", 0.0))

                    # extract basic fields safely
                    entry = None
                    try:
                        entry = (res["analysis"].get(interval, {}).get("close") or next(iter(res["analysis"].values())).get("close"))
                    except Exception:
                        entry = None

                    sl = None
                    tp1 = None
                    try:
                        sl = (res["analysis"].get("1h", {}).get("sl") or res["analysis"].get(interval, {}).get("sl"))
                        tp1 = (res["analysis"].get("1h", {}).get("tp1") or res["analysis"].get(interval, {}).get("tp1"))
                    except Exception:
                        pass

                    reasons = []
                    try:
                        for tf, info in res.get("analysis", {}).items():
                            if isinstance(info, dict):
                                reasons.extend(info.get("reasons", []))
                    except Exception:
                        pass

                    # Optional AI augmentation (best effort)
                    if ai_analysis_text and score > 0.3:
                        try:
                            prompt = f"Summarize confidence for {symbol} given analysis: {json.dumps(res)[:2000]}"
                            resp = ai_analysis_text(prompt)
                            ai_text = ""
                            if isinstance(resp, dict):
                                ai_text = resp.get("analysis") or ""
                            else:
                                ai_text = str(resp)
                            m = re.search(r"([+-]?[0-9]*\.?[0-9]+)", ai_text)
                            if m:
                                mod = float(m.group(1))
                                score = max(0.0, min(1.0, score + mod))
                                reasons.append("ai_adj")
                        except Exception:
                            logger.exception("AI augmentation attempt failed")

                    return {
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "signal": "LONG" if "LONG" in combined or "STRONG_LONG" in combined else ("SHORT" if "SHORT" in combined or "STRONG_SHORT" in combined else "HOLD"),
                        "entry": float(entry) if entry else None,
                        "sl": float(sl) if sl else None,
                        "tp1": float(tp1) if tp1 else None,
                        "confidence": float(score),
                        "reasons": list(dict.fromkeys(reasons)),
                    }
            except Exception:
                logger.exception("analyze_pair_multi_timeframes failed; falling back")

        # 2) Fallback: try to fetch klines for exchanges and run legacy generate_signal
        exchanges_to_try = ["binance", "bybit", "kucoin", "okx"]
        if fetch_klines_multi or True:
            for ex in exchanges_to_try:
                try:
                    # Prefer fetch_klines_multi if present
                    if fetch_klines_multi:
                        df = None
                        try:
                            df = fetch_klines_multi(symbol, interval, limit=200, exchange=ex)
                        except Exception:
                            df = None
                        if df is None:
                            # fallback to public
                            kl = fetch_klines_public(symbol, interval, limit=200, exchange=ex)
                            df = kl
                    else:
                        df = fetch_klines_public(symbol, interval, limit=200, exchange=ex)

                    # If still no data, continue
                    if not df:
                        continue

                    # If legacy_generate_signal exists and df looks like DataFrame or list
                    if legacy_generate_signal:
                        try:
                            out = legacy_generate_signal(df, pair=symbol) if callable(legacy_generate_signal) else legacy_generate_signal
                            sig_text = str(out)
                            if "BUY" in sig_text.upper() or "STRONG BUY" in sig_text.upper():
                                sig = "LONG"
                            elif "SELL" in sig_text.upper():
                                sig = "SHORT"
                            else:
                                sig = "HOLD"
                        except Exception:
                            logger.exception("legacy_generate_signal failed")
                            sig = "HOLD"
                    else:
                        # Simple local heuristic for fallback: look at last two closes if available
                        try:
                            # df may be list of dicts or pandas DataFrame
                            closes = []
                            if hasattr(df, "iloc"):
                                # pandas-like
                                closes = df["close"].astype(float).tolist()[-3:]
                            elif isinstance(df, list) and len(df) > 0 and isinstance(df[0], dict):
                                closes = [float(r["close"]) for r in df][-3:]
                            if len(closes) >= 2:
                                sig = "LONG" if closes[-1] > closes[-2] else "SHORT" if closes[-1] < closes[-2] else "HOLD"
                            else:
                                sig = "HOLD"
                        except Exception:
                            sig = "HOLD"

                    # Estimate entry/sl/tp using simple ATR-like measure if data available
                    try:
                        highs = []
                        lows = []
                        closes = []
                        if hasattr(df, "iloc"):
                            highs = df["high"].astype(float)
                            lows = df["low"].astype(float)
                            closes = df["close"].astype(float)
                            atr_val = (highs - lows).rolling(14).mean().iloc[-1]
                            last = float(closes.iloc[-1])
                        elif isinstance(df, list) and isinstance(df[0], dict):
                            highs = [float(r["high"]) for r in df]
                            lows = [float(r["low"]) for r in df]
                            closes = [float(r["close"]) for r in df]
                            # crude atr
                            diffs = [h - l for h, l in zip(highs[-14:], lows[-14:])] if len(highs) >= 14 else [h - l for h, l in zip(highs, lows)]
                            atr_val = sum(diffs) / len(diffs) if diffs else 0.0
                            last = closes[-1] if closes else None
                        else:
                            atr_val = 0.0
                            last = None

                        if last is not None and atr_val:
                            if sig == "LONG":
                                sl = last - (atr_val * 1.5)
                                tp1 = last + (atr_val * 1.5)
                            elif sig == "SHORT":
                                sl = last + (atr_val * 1.5)
                                tp1 = last - (atr_val * 1.5)
                            else:
                                sl = last * 0.995
                                tp1 = last * 1.005
                        else:
                            sl = None
                            tp1 = None
                    except Exception:
                        sl = None
                        tp1 = None

                    confidence = 0.35  # baseline
                    reasons = [f"fallback_{ex}"]
                    return {
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "signal": sig,
                        "entry": float(last) if last else None,
                        "sl": float(sl) if sl else None,
                        "tp1": float(tp1) if tp1 else None,
                        "confidence": float(confidence),
                        "reasons": reasons,
                    }
                except Exception:
                    logger.exception("fetch_klines attempt failed for exchange %s", ex)
                    continue
            return {"error": "no_data_on_exchanges"}
        else:
            return {"error": "no_fetch_klines_available"}
    except Exception as exc:
        logger.exception("_safe_generate_signal unexpected error")
        return {"error": str(exc)}


# ---------------------------
# Record + send signal (image-first)
# ---------------------------
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
            if save_data:
                save_data(d)
    except Exception:
        logger.exception("Failed to save signal record")

    # Text lines to render on image (per your requirement: all text inside images)
    lines = []
    try:
        lines.append(f"ID: {sig_id}  {sig.get('symbol')} | {sig.get('interval')}")
        lines.append(f"Signal: {sig.get('signal')}")
        lines.append(f"Entry: {sig.get('entry') or 'N/A'}  SL: {sig.get('sl') or 'N/A'}  TP1: {sig.get('tp1') or 'N/A'}")
        confidence_pct = int(sig.get("confidence", 0) * 100)
        lines.append(f"Confidence: {confidence_pct}%  Risk: ${risk_amt}")
        reasons = sig.get("reasons", []) or []
        lines.append("Reasons: " + (", ".join(reasons[:3]) or "None"))
    except Exception:
        lines = [f"Signal for {sig.get('symbol')}"]

    # Try to attach a quickchart for the symbol (60 closes) - best-effort
    chart_bytes = None
    try:
        # attempt to fetch closes using fetch_klines_multi or public
        kl = fetch_klines_public(sig.get("symbol") or sig.get("symbol"), interval=sig.get("interval") or SIGNAL_INTERVAL_DEFAULT, limit=60, exchange="binance")
        closes = []
        if kl:
            if hasattr(kl, "iloc"):
                # pandas
                closes = kl["close"].astype(float).tolist()[-60:]
            elif isinstance(kl, list) and isinstance(kl[0], dict):
                closes = [float(r["close"]) for r in kl][-60:]
        if closes:
            chart_bytes = generate_sparkline_bytes(closes, label=sig.get("symbol"))
    except Exception:
        logger.exception("chart generation failed")

    # Build image and send
    try:
        _send_branded(chat_id or ADMIN_ID, "\n".join(lines), lines_for_image=lines, chart_bytes=chart_bytes, reply_markup=None)
    except Exception:
        logger.exception("Failed to send signal image; falling back to message")
        try:
            bot.send_message(chat_id or ADMIN_ID, _append_brand("\n".join(lines)))
        except Exception:
            pass

    # optionally send a quick AI rationale follow-up if available (as separate image)
    try:
        if ai_analysis_text and sig and not sig.get("error"):
            prompt = f"Provide concise trade rationale for this signal:\n\n{json.dumps(sig, default=str)}"
            ai_text = None
            try:
                resp = ai_analysis_text(prompt) if callable(ai_analysis_text) else None
                if isinstance(resp, dict):
                    ai_text = resp.get("analysis") or str(resp)
                else:
                    ai_text = str(resp)
            except Exception:
                ai_text = None

            if not ai_text:
                # local fallback
                resp_local = _local_ai_analysis(prompt)
                ai_text = resp_local.get("analysis") if isinstance(resp_local, dict) else str(resp_local)
            if ai_text:
                # render AI text into image
                ai_lines = ["AI Rationale:"] + (ai_text.splitlines()[:8])
                _send_branded(chat_id or ADMIN_ID, ai_text, lines_for_image=ai_lines)
    except Exception:
        logger.exception("AI rationale follow-up failed")

    return sig_id


# ---------------------------
# Small background scanner wrappers
# ---------------------------
def _internal_scanner_loop(pairs, interval, exchanges, min_confidence, poll_seconds, use_ai):
    """
    Fallback scanner used when se_start_auto_scanner is not available.
    """
    logger.info("Fallback scanner started for pairs=%s interval=%s", pairs, interval)
    while not _scanner_stop_event.is_set():
        try:
            for p in pairs:
                if _scanner_stop_event.is_set():
                    break
                try:
                    # prefer generate_signal_multi if present
                    if generate_signal_multi:
                        try:
                            res = generate_signal_multi(p, interval, exchanges=exchanges, min_confidence_for_signal=min_confidence, use_ai_explain=use_ai)
                        except Exception:
                            res = _safe_generate_signal(p, interval)
                    else:
                        res = _safe_generate_signal(p, interval)

                    if not isinstance(res, dict):
                        continue
                    conf = float(res.get("confidence", 0.0) or 0.0)
                    is_strong = (conf >= min_confidence) and (res.get("signal") in ("LONG", "SHORT"))
                    consensus = bool(res.get("consensus")) if isinstance(res.get("consensus"), bool) else False
                    if is_strong and (consensus or conf >= min_confidence):
                        logger.info("Fallback scanner: strong signal %s %s %.3f", p, res.get("signal"), conf)
                        try:
                            if register_send_callback:
                                # notify registered systems
                                try:
                                    register_send_callback(lambda sig: None)
                                except Exception:
                                    pass
                            # If AUTO_SEND is configured and admin policy allows, send
                            if AUTO_SEND_ONLY_ADMIN and ADMIN_ID:
                                record_signal_and_send(res, chat_id=ADMIN_ID, user_id=ADMIN_ID, auto=True)
                            elif not AUTO_SEND_ONLY_ADMIN:
                                record_signal_and_send(res, chat_id=ADMIN_ID, user_id=ADMIN_ID, auto=True)
                        except Exception:
                            logger.exception("Fallback scanner: failed to deliver signal")
                except Exception:
                    logger.exception("Fallback scanner: per-pair error")
            # sleep with interruption checks
            for _ in range(max(1, int(poll_seconds))):
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


# ---------------------------
# Keyboard UI
# ---------------------------
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
        # always present as image (per your request)
        try:
            img = create_brand_image(lines, title="Destiny Trading Empire Bot 💎")
            if img:
                safe_send_with_image(bot, msg.chat.id, "", img, reply_markup=main_keyboard())
                return
        except Exception:
            logger.exception("Failed creating welcome brand image")
        # fallback
        bot.send_message(msg.chat.id, _append_brand(text), reply_markup=main_keyboard())
    except Exception:
        logger.exception("cmd_start failed")


@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(fi.file_path)
        if record_pnl_screenshot:
            record_pnl_screenshot(data, datetime.utcnow().strftime("%Y%m%d_%H%M%S"), message.from_user.id, message.caption)
        # respond with branded image saying saved
        _send_branded(message.chat.id, "Saved screenshot. Reply with `#link <signal_id> TP1` or `#link <signal_id> SL`", lines_for_image=["Screenshot saved"])
    except Exception:
        logger.exception("photo_handler failed")
        _send_branded(message.chat.id, "Failed to save screenshot.", lines_for_image=["Save failed"])


@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_handler(message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 3:
            _send_branded(message.chat.id, "Usage: #link <signal_id> TP1 or SL", lines_for_image=["Usage: #link <signal_id> TP1 or SL"])
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
            _send_branded(message.chat.id, "No unlinked screenshot found.", lines_for_image=["No unlinked screenshot found"])
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
        if save_data and isinstance(d, dict):
            save_data(d)
        _send_branded(message.chat.id, f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance.", lines_for_image=[f"Linked {sig_id} as {tag}"])
    except Exception:
        logger.exception("link_handler failed")
        _send_branded(message.chat.id, "Failed to link screenshot.", lines_for_image=["Failed to link screenshot"])


# ----- Callback actions -----
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        cid = call.message.chat.id
        data = call.data
        # answer callback quickly to avoid spinner
        try:
            bot.answer_callback_query(call.id)
        except Exception:
            pass

        # Choose pair -> produce inline keyboard of PAIRS
        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"sig_{p}"))
            _send_branded(cid, "Choose pair to analyze:", lines_for_image=["Choose pair to analyze:"], reply_markup=kb)
            return

        # individual pair selected
        if data.startswith("sig_"):
            pair = data.split("_", 1)[1]
            bot.send_chat_action(cid, "typing")
            sig = _safe_generate_signal(pair, SIGNAL_INTERVAL_DEFAULT)
            if sig.get("error"):
                err = sig.get("error")
                _send_branded(cid, f"Error generating signal: {err}\nTry another timeframe or check pair symbol.", lines_for_image=[f"Error: {err}"])
                return
            sig.setdefault("interval", SIGNAL_INTERVAL_DEFAULT)
            sig.setdefault("symbol", pair.upper())
            record_signal_and_send(sig, chat_id=cid, user_id=call.from_user.id, auto=False)
            return

        # quick scan top X
        if data == "scan_top4":
            _send_branded(cid, "🔎 Scanning top pairs across exchanges...", lines_for_image=["Scanning..."])
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
            _send_branded(cid, "📡 Fetching multi-exchange trending pairs... please wait.", lines_for_image=["Fetching trending pairs..."])
            try:
                if fetch_trending_pairs_branded:
                    img_buf, caption = fetch_trending_pairs_branded(limit=8)
                    if img_buf:
                        safe_send_with_image(bot, cid, "", img_buf)
                    else:
                        _send_branded(cid, caption or "No trending data", lines_for_image=[caption or "No trending data"])
                elif fetch_trending_pairs_text:
                    txt = fetch_trending_pairs_text()
                    _send_branded(cid, txt, lines_for_image=txt.splitlines()[:8])
                else:
                    # fallback: call CoinGecko trending endpoint
                    try:
                        cg = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=8).json()
                        coins = [f"{i['item']['name']} ({i['item']['symbol']})" for i in cg.get("coins", [])]
                        _send_branded(cid, "Trending coins:\n" + "\n".join(coins[:8]), lines_for_image=coins[:8])
                    except Exception:
                        _send_branded(cid, "Trending feature not available (missing market_providers).", lines_for_image=["Trending not available"])
            except Exception:
                logger.exception("trending handler failed")
                _send_branded(cid, "Failed to fetch trending pairs.", lines_for_image=["Failed to fetch trending pairs"])
            return

        if data == "bot_status":
            # provide brief health info and whether scanner is running
            scanner_running = _scanner_thread is not None and _scanner_thread.is_alive()
            msg_lines = [
                "⚙️ Bot Status",
                f"Running: ✅",
                f"Scanner running: {scanner_running}",
                f"Auto confidence threshold: {AUTO_CONFIDENCE_THRESHOLD*100:.0f}%",
            ]
            _send_branded(cid, "\n".join(msg_lines), lines_for_image=msg_lines)
            return

        if data == "market_news":
            # market news: use CoinGecko / reddit/trending or simple curated headlines via cointelegraph RSS? We'll use CoinGecko trending + fallback notes
            try:
                try:
                    # simple approach: get coin market data top movers
                    r = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=8)
                    r.raise_for_status()
                    trending_json = r.json()
                    lines = ["Market News — Trending coins"]
                    for i in trending_json.get("coins", [])[:8]:
                        it = i.get("item", {})
                        lines.append(f"{it.get('name')} ({it.get('symbol')}) — Rank {it.get('market_cap_rank')}")
                    _send_branded(cid, "\n".join(lines), lines_for_image=lines)
                except Exception:
                    # fallback: list top gainers from Binance public ticker endpoint
                    r2 = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8)
                    r2.raise_for_status()
                    arr = r2.json()
                    # pick top 6 by priceChangePercent
                    arr_sorted = sorted(arr, key=lambda x: float(x.get("priceChangePercent", 0)), reverse=True)
                    lines = ["Market News — Binance top movers"]
                    for a in arr_sorted[:6]:
                        lines.append(f"{a.get('symbol')}: {float(a.get('priceChangePercent',0)):+.2f}% vol:{int(float(a.get('quoteVolume',0))):,}")
                    _send_branded(cid, "\n".join(lines), lines_for_image=lines)
            except Exception:
                logger.exception("market_news failed")
                _send_branded(cid, "📰 Market news: feature coming soon", lines_for_image=["Market news coming soon"])
            return

        if data == "challenge_status":
            d = load_data() if load_data else {}
            bal = d.get("challenge", {}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START
            wins = d.get("challenge", {}).get("wins", 0) if isinstance(d, dict) else 0
            losses = d.get("challenge", {}).get("losses", 0) if isinstance(d, dict) else 0
            _send_branded(cid, f"Balance: ${bal:.2f}\nWins: {wins} Losses: {losses}", lines_for_image=[f"Balance: ${bal:.2f}", f"Wins: {wins} Losses: {losses}"])
            return

        if data == "ask_ai":
            # instruct user how to ask AI; accept AI: prefixed messages via message handler (not implemented here)
            _send_branded(cid, "🤖 Ask AI: send a message starting with `AI:` followed by your question.", lines_for_image=["Ask AI: send a message starting with 'AI:' followed by your question."])
            return

        if data == "refresh_bot":
            _send_branded(cid, "🔄 Refreshing bot session...", lines_for_image=["Refreshing bot session..."])
            stop_existing_bot_instances()
            time.sleep(2)
            _send_branded(cid, "✅ Refreshed.", lines_for_image=["Refreshed"])
            return

        if data == "start_auto_brief":
            _send_branded(cid, "▶ Starting background market scanner (auto-send strong signals).", lines_for_image=["Starting scanner..."])
            start_background_scanner()
            return

        if data == "stop_auto_brief":
            _send_branded(cid, "⏹ Stopping background market scanner.", lines_for_image=["Stopping scanner..."])
            stop_background_scanner()
            return

        # Scheduler-based auto briefs (text/AI summaries)
        if data == "start_auto_brief_scheduler":
            if start_scheduler:
                _send_branded(cid, "▶ Scheduler for auto-briefs enabled. You will receive periodic market briefs.", lines_for_image=["Scheduler enabled"])
                try:
                    start_scheduler(bot)
                except Exception:
                    logger.exception("start_scheduler failed")
            else:
                _send_branded(cid, "Scheduler not available (missing scheduler module).", lines_for_image=["Scheduler missing"])
            return

        if data == "stop_auto_brief_scheduler":
            if stop_scheduler:
                stop_scheduler()
                _send_branded(cid, "⏹ Scheduler for auto-briefs disabled.", lines_for_image=["Scheduler disabled"])
            else:
                _send_branded(cid, "Scheduler not available (missing scheduler module).", lines_for_image=["Scheduler missing"])
            return

        if data.startswith("ai_"):
            sig_id = data.split("_", 1)[1]
            d = load_data() if load_data else {}
            rec = next((s for s in d.get("signals", []) if s["id"] == sig_id), None) if isinstance(d, dict) else None
            if not rec:
                _send_branded(cid, "Signal not found", lines_for_image=["Signal not found"])
                return
            prompt = f"Provide trade rationale, risk controls and a recommended leverage for this trade:\n{rec['signal']}"
            ai_text = None
            try:
                if ai_analysis_text and callable(ai_analysis_text):
                    resp = ai_analysis_text(prompt)
                    if isinstance(resp, dict):
                        ai_text = resp.get("analysis") or str(resp)
                    else:
                        ai_text = str(resp)
                else:
                    ai_text = _local_ai_analysis(prompt).get("analysis")
            except Exception:
                ai_text = "AI feature error"
            _send_branded(cid, f"🤖 AI analysis:\n{ai_text}", lines_for_image=(ai_text.splitlines()[:8] if ai_text else ["AI error"]))
            return

        bot.send_message(cid, _append_brand("Unknown action"))
    except Exception:
        logger.exception("callback_handler failed")
        try:
            bot.answer_callback_query(call.id, "Handler error")
        except Exception:
            pass


# If this file is executed directly allow a small self-test message (optional)
def start_bot_polling():
    """
    Start the bot polling loop. Exposed so your bot.py can import and call it.
    """
    try:
        logger.info("Starting bot polling")
        # stop other instances if possible
        stop_existing_bot_instances()
        # Telebot has infinity_polling in recent versions; fallback to polling
        try:
            if hasattr(bot, "infinity_polling"):
                bot.infinity_polling(timeout=60, long_polling_timeout=80)
            else:
                bot.polling(none_stop=True)
        except KeyboardInterrupt:
            logger.info("Polling stopped by user")
        except Exception:
            logger.exception("Bot polling crashed")
            raise
    except Exception:
        logger.exception("start_bot_polling failed")
        raise


if __name__ == "__main__":
    print("bot_runner.py loaded - sanity check")
    logger.info("bot_runner loaded. Ready.")

