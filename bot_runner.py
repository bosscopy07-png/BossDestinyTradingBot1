# bot_runner.py
import os
import time
import threading
import traceback
import requests
import logging
from datetime import datetime
import telebot
from telebot import types

# ----- Branding and global constants -----
BRAND_TAG = "\n\n— <b>Destiny Trading Empire Bot 💎</b>"

# Config from env (with sane defaults)
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT").split(",")
# scan all timeframes per your request:
SCAN_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
SIGNAL_INTERVAL_DEFAULT = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "1"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "100.0"))
AUTO_CONFIDENCE_THRESHOLD = float(os.getenv("AUTO_CONFIDENCE_THRESHOLD", "0.90"))   # 0.90 = 90%
AUTO_SEND_ONLY_ADMIN = True        # send to admin only (as requested)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

# ----- Logging -----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bot_runner")

# ----- Import project modules (must exist). If missing, log error and fallback gracefully -----
try:
    from market_providers import (
        fetch_trending_pairs_branded,
        fetch_klines_multi,
        get_session,
        fetch_trending_pairs_text,
        detect_strong_signals,
        generate_branded_signal_image,
        analyze_pair_multi_timeframes
    )
except Exception as e:
    log.exception("market_providers import failed")
    fetch_trending_pairs_branded = None
    fetch_klines_multi = None
    get_session = None
    fetch_trending_pairs_text = None
    detect_strong_signals = None
    generate_branded_signal_image = None
    analyze_pair_multi_timeframes = None

try:
    from image_utils import build_signal_image, safe_send_with_image, create_brand_image
except Exception:
    log.exception("image_utils import failed")
    build_signal_image = None
    safe_send_with_image = None
    create_brand_image = None

try:
    from signal_engine import generate_signal
except Exception:
    log.exception("signal_engine import failed")
    generate_signal = None

try:
    from storage import ensure_storage, load_data, save_data, record_pnl_screenshot
except Exception:
    log.exception("storage import failed")
    ensure_storage = None
    load_data = None
    save_data = None
    record_pnl_screenshot = None

try:
    from ai_client import ai_analysis_text
except Exception:
    log.exception("ai_client import failed")
    ai_analysis_text = None

try:
    from pro_features import top_gainers_pairs, fear_and_greed_index, futures_leverage_suggestion, quickchart_price_image, ai_market_brief_text
except Exception:
    log.exception("pro_features import failed")
    top_gainers_pairs = None
    fear_and_greed_index = None
    futures_leverage_suggestion = None
    quickchart_price_image = None
    ai_market_brief_text = None

try:
    from scheduler import start_scheduler, stop_scheduler
except Exception:
    log.exception("scheduler import failed")
    start_scheduler = None
    stop_scheduler = None

# Ensure storage if available
if ensure_storage:
    try:
        ensure_storage()
    except Exception:
        log.exception("ensure_storage() failed")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_last_signal_time = {}      # dict mapping "symbol|interval" -> datetime
_scanner_thread = None
_scanner_stop_event = threading.Event()

# ----------------- Utilities -----------------
def _append_brand(text: str) -> str:
    if BRAND_TAG.strip() not in text:
        return text + BRAND_TAG
    return text

def _maybe_image_for_text(lines):
    """Return BytesIO image if create_brand_image available, else None."""
    try:
        if create_brand_image:
            return create_brand_image(lines)
    except Exception:
        log.exception("create_brand_image failed")
    return None

def _normalize_symbol(sym: str) -> str:
    """Normalize symbols to format expected by fetch_klines (e.g., BTCUSDT)."""
    if not sym: 
        return sym
    s = str(sym).upper().replace("/", "").replace("-", "").replace("_", "")
    # special-case if user passed with slash as "BTC/USDT" convert to "BTCUSDT"
    return s

def stop_existing_bot_instances():
    """Attempt to clear other getUpdates sessions — reduces 409 conflicts on re-deploy."""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset=-1"
        requests.get(url, timeout=5)
        log.info("[BOT] Attempted to stop other bot sessions (getUpdates offset -1).")
    except Exception:
        log.exception("stop_existing_bot_instances failed")

def can_send_signal(symbol: str, interval: str) -> bool:
    key = f"{symbol}|{interval}"
    last = _last_signal_time.get(key)
    if not last:
        return True
    return (datetime.utcnow() - last).total_seconds() > COOLDOWN_MIN * 60

def mark_signal_sent(symbol: str, interval: str):
    key = f"{symbol}|{interval}"
    _last_signal_time[key] = datetime.utcnow()

def compute_risk_and_size(entry: float, sl: float, balance: float, risk_percent: float):
    try:
        risk_amount = (balance * risk_percent) / 100.0
        diff = abs(float(entry) - float(sl))
        if diff <= 1e-12:
            return round(risk_amount, 8), 0.0
        pos_size = risk_amount / diff
        return round(risk_amount, 8), round(pos_size, 8)
    except Exception:
        log.exception("compute_risk_and_size failed")
        return 0.0, 0.0

# Wrap signal engine with robust error reporting
def _safe_generate_signal(symbol: str, interval: str):
    """Call generate_signal or fallback analyze_pair_multi_timeframes; return a dict with keys or {'error':...}"""
    try:
        symbol_norm = _normalize_symbol(symbol)
        # Prefer signal_engine.generate_signal if available
        if generate_signal:
            res = generate_signal(symbol_norm, interval)
            if isinstance(res, dict):
                return res
            # if older engine returned text, wrap it
            return {"signal": str(res), "symbol": symbol_norm, "interval": interval, "entry": None, "sl": None, "tp1": None, "confidence": 0.0}
        # fallback: use analyze_pair_multi_timeframes (market_providers)
        if analyze_pair_multi_timeframes:
            # request timeframes: chosen interval + 1h + 4h for better combined score
            tfs = [interval]
            if "1h" not in tfs: tfs.append("1h")
            if "4h" not in tfs: tfs.append("4h")
            res = analyze_pair_multi_timeframes(symbol_norm, timeframes=tfs, exchange="binance")
            if res.get("error"):
                return {"error": res.get("error")}
            # pick 1h info if present else first
            info = res.get("analysis", {}).get(interval) or res.get("analysis", {}).get("1h") or next(iter(res.get("analysis", {}).values()))
            return {
                "symbol": symbol_norm,
                "interval": interval,
                "signal": info.get("signal", "HOLD"),
                "entry": info.get("close"),
                "sl": info.get("sl"),
                "tp1": info.get("tp1"),
                "confidence": res.get("combined_score", 0.0),
                "reasons": info.get("reasons", []),
                "raw": res
            }
        return {"error": "No signal engine available (generate_signal/analyze_pair_multi_timeframes missing)."}
    except Exception as e:
        log.exception("Error in _safe_generate_signal")
        return {"error": str(e)}

# ---------------- Recording & sending ----------------
def record_signal_and_send(sig: dict, chat_id=None, user_id=None, auto=False):
    """Record in storage and send signal as image + AI follow-up text (if available)."""
    try:
        d = load_data() if load_data else {}
    except Exception:
        d = {}

    sig_id = f"S{int(time.time())}"
    balance = d.get("challenge", {}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START

    risk_amt, pos_size = compute_risk_and_size(sig.get("entry") or 0.0, sig.get("sl") or 0.0, balance, RISK_PERCENT)

    rec = {
        "id": sig_id,
        "signal": sig,
        "time": datetime.utcnow().isoformat(),
        "risk_amt": risk_amt,
        "pos_size": pos_size,
        "user": user_id or ADMIN_ID,
        "auto": bool(auto)
    }

    try:
        if isinstance(d, dict):
            d.setdefault("signals", []).append(rec)
            d.setdefault("stats", {})
            d["stats"]["total_signals"] = d["stats"].get("total_signals", 0) + 1
            if save_data:
                save_data(d)
    except Exception:
        log.exception("Failed to save signal record")

    # Build caption
    confidence_pct = 0
    try:
        confidence_pct = int(float(sig.get("confidence", 0)) * 100)
    except Exception:
        pass

    caption = (
        f"🔥 <b>Destiny Trading Empire — Signal</b>\n"
        f"ID: {sig_id}\nPair: {sig.get('symbol')} | TF: {sig.get('interval')}\n"
        f"Signal: <b>{sig.get('signal')}</b>\nEntry: {sig.get('entry')} | SL: {sig.get('sl')} | TP1: {sig.get('tp1')}\n"
        f"Confidence: {confidence_pct}% | Risk (USD): {risk_amt}\n"
        f"Reasons: {', '.join(sig.get('reasons', []) or ['None'])}\n"
    )
    caption = _append_brand(caption)

    # Always attempt to create an image for every reply (user requested)
    img = None
    try:
        # If the signal already has an image buffer (from detect_strong_signals), use it.
        if sig.get("image"):
            img = sig.get("image")
        elif build_signal_image:
            img = build_signal_image(sig)
        elif create_brand_image:
            img = create_brand_image([caption])
    except Exception:
        log.exception("Failed to build signal image")

    # Keyboard
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🔁 Share", switch_inline_query=f"{sig.get('symbol')}"))

    # Send
    try:
        if safe_send_with_image:
            safe_send_with_image(bot, chat_id or ADMIN_ID, caption, img, kb)
        else:
            if img:
                bot.send_photo(chat_id or ADMIN_ID, img, caption=caption, reply_markup=kb)
            else:
                bot.send_message(chat_id or ADMIN_ID, caption, reply_markup=kb)
    except Exception:
        log.exception("Failed to send signal message")

    # AI follow-up message (concise rationale)
    try:
        if ai_analysis_text and sig and not sig.get("error"):
            prompt = f"Provide concise trade rationale for this signal:\n{sig}"
            ai_text = ai_analysis_text(prompt)
            if ai_text:
                ai_msg = _append_brand(f"🤖 AI Rationale:\n{ai_text}")
                # we try to send as image too if create_brand_image available
                img2 = _maybe_image_for_text([ai_text]) or None
                if safe_send_with_image:
                    safe_send_with_image(bot, chat_id or ADMIN_ID, ai_msg, img2)
                else:
                    bot.send_message(chat_id or ADMIN_ID, ai_msg)
    except Exception:
        log.exception("AI rationale follow-up failed")

    return sig_id

# ---------------- Keyboard UI ----------------
def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📈 Get Signals", callback_data="get_signal"),
        types.InlineKeyboardButton("🔎 Scan Top 4", callback_data="scan_top4")
    )
    kb.add(
        types.InlineKeyboardButton("⚙️ Bot Status", callback_data="bot_status"),
        types.InlineKeyboardButton("🚀 Trending Pairs", callback_data="trending")
    )
    kb.add(
        types.InlineKeyboardButton("📰 Market News", callback_data="market_news"),
        types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status")
    )
    kb.add(
        types.InlineKeyboardButton("📸 Upload PnL", callback_data="pnl_upload"),
        types.InlineKeyboardButton("🧾 History", callback_data="history")
    )
    kb.add(
        types.InlineKeyboardButton("🤖 AI Market Brief", callback_data="ask_ai"),
        types.InlineKeyboardButton("🔄 Refresh Bot", callback_data="refresh_bot")
    )
    kb.add(
        types.InlineKeyboardButton("▶️ Start Auto Scanner", callback_data="start_auto_brief"),
        types.InlineKeyboardButton("⏹ Stop Auto Scanner", callback_data="stop_auto_brief")
    )
    kb.add(
        types.InlineKeyboardButton("📣 Start Auto Briefs", callback_data="start_auto_brief_scheduler"),
        types.InlineKeyboardButton("⛔ Stop Auto Briefs", callback_data="stop_auto_brief_scheduler")
    )
    return kb

# ---------------- Telegram handlers ----------------
@bot.message_handler(commands=['start', 'menu'])
def cmd_start(msg):
    try:
        text = "👋 Welcome Boss Destiny!\n\nThis is your Trading Empire control panel."
        text = _append_brand(text)
        if create_brand_image:
            img = create_brand_image(["Welcome — Destiny Trading Empire Bot 💎"])
            safe_send_with_image(bot, msg.chat.id, text, img, reply_markup=main_keyboard())
        else:
            bot.send_message(msg.chat.id, text, reply_markup=main_keyboard())
    except Exception:
        log.exception("cmd_start failed")

# Accept AI queries with prefix "AI:"
@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().upper().startswith("AI:"))
def handle_ai_query(message):
    try:
        prompt = message.text.strip()[3:].strip()
        if not prompt:
            bot.reply_to(message, _append_brand("Usage: AI: <your question>"))
            return
        bot.send_chat_action(message.chat.id, "typing")
        if ai_analysis_text:
            res = ai_analysis_text(prompt)
            text = _append_brand(f"🤖 AI answer:\n{res}")
            img = _maybe_image_for_text([res])
            safe_send_with_image(bot, message.chat.id, text, img)
        else:
            bot.reply_to(message, _append_brand("⚠️ AI not configured. Set OPENAI_API_KEY and install openai package."))
    except Exception:
        log.exception("handle_ai_query failed")
        bot.reply_to(message, _append_brand("AI request failed."))

@bot.message_handler(content_types=['photo'])
def photo_handler(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(fi.file_path)
        if record_pnl_screenshot:
            fname = record_pnl_screenshot(data, datetime.utcnow().strftime("%Y%m%d_%H%M%S"), message.from_user.id, message.caption)
            bot.reply_to(message, _append_brand(f"Saved screenshot as {fname}. You can link to a signal using `#link <signal_id> TP1` or `#link <signal_id> SL`."))
        else:
            # fallback: save to storage dir
            bot.reply_to(message, _append_brand("Screenshot received (storage module not available)."))
    except Exception:
        log.exception("photo_handler failed")
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
        pnl_item = None
        if isinstance(d, dict):
            pnl_item = next((p for p in reversed(d.get("pnl", [])) if p.get("linked") is None and p.get("from") == message.from_user.id), None)
        if not pnl_item:
            bot.reply_to(message, _append_brand("No unlinked screenshot found in storage."))
            return
        pnl_item["linked"] = {"signal_id": sig_id, "result": tag, "linked_by": message.from_user.id}
        if message.from_user.id == ADMIN_ID:
            srec = next((s for s in d.get("signals", []) if s["id"] == sig_id), None)
            if srec:
                risk = srec.get("risk_amt", 0)
                if tag.startswith("TP"):
                    d.setdefault("challenge", {})["balance"] = d.get("challenge", {}).get("balance", CHALLENGE_START) + risk
                    d["challenge"]["wins"] = d["challenge"].get("wins", 0) + 1
                    d.setdefault("stats", {})["wins"] = d.get("stats", {}).get("wins", 0) + 1
                elif tag == "SL":
                    d.setdefault("challenge", {})["balance"] = d.get("challenge", {}).get("balance", CHALLENGE_START) - risk
                    d["challenge"]["losses"] = d["challenge"].get("losses", 0) + 1
                    d.setdefault("stats", {})["losses"] = d.get("stats", {}).get("losses", 0) + 1
        if save_data and isinstance(d, dict):
            save_data(d)
        bot.reply_to(message, _append_brand(f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance."))
    except Exception:
        log.exception("link_handler failed")
        bot.reply_to(message, _append_brand("Failed to link screenshot."))

# ----- Callback actions -----
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        cid = call.message.chat.id
        data = call.data
        bot.answer_callback_query(call.id)
        log.info("Callback: %s from user %s", data, call.from_user.id)

        # Choose pair -> show inline keyboard of PAIRS
        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"sig_{p}"))
            bot.send_message(cid, _append_brand("Choose pair to analyze:"), reply_markup=kb)
            return

        # Pair selected
        if data.startswith("sig_"):
            pair_raw = data.split("_", 1)[1]
            pair = _normalize_symbol(pair_raw)
            bot.send_chat_action(cid, "typing")
            # generate and return
            sig = _safe_generate_signal(pair, SIGNAL_INTERVAL_DEFAULT)
            if sig.get("error"):
                # provide details and attempt to show small branded image with error
                err = sig.get("error")
                text = _append_brand(f"⚠️ Error generating signal for {pair}: {err}")
                img = _maybe_image_for_text([f"Error: {err}"])
                safe_send_with_image(bot, cid, text, img)
                return
            record_signal_and_send(sig, chat_id=cid, user_id=call.from_user.id, auto=False)
            return

              # quick scan top pairs
        if data == "scan_top4":
            bot.send_message(cid, _append_brand("🔎 Scanning top pairs across exchanges..."))
            # If market_providers.detect_strong_signals exists, use it to find candidates quickly
            try:
                if detect_strong_signals:
                    strong = detect_strong_signals(pairs=PAIRS, timeframes=["15m", "1h", "4h"], min_confidence=0.75)
                    if not strong:
                        bot.send_message(cid, _append_brand("No strong candidates found right now."))
                        return
                    for s in strong[:6]:
                        # send each candidate (image + caption)
                        img, caption = None, "\n".join(s.get("caption_lines") or [])
                        if s.get("image"):
                            img = s.get("image")
                        else:
                            img = generate_branded_signal_image(s)[0] if generate_branded_signal_image else None
                        safe_send_with_image(bot, cid, _append_brand(caption), img)
                    return
                # Fallback: generate per-pair
                for p in PAIRS[:6]:
                    if can_send_signal(p, SIGNAL_INTERVAL_DEFAULT):
                        sig = _safe_generate_signal(p, SIGNAL_INTERVAL_DEFAULT)
                        if not sig.get("error") and sig.get("signal") in ("LONG", "SHORT", "STRONG_LONG", "STRONG_SHORT"):
                            record_signal_and_send(sig, chat_id=cid)
                return
            except Exception:
                log.exception("scan_top4 failed")
                bot.send_message(cid, _append_brand("Scan failed."))
            return

        if data == "trending":
            bot.send_message(cid, _append_brand("📡 Fetching multi-exchange trending pairs... please wait."))
            try:
                if fetch_trending_pairs_branded:
                    img_buf, caption = fetch_trending_pairs_branded(limit=8)
                    if img_buf:
                        safe_send_with_image(bot, cid, _append_brand(caption), img_buf)
                    else:
                        bot.send_message(cid, _append_brand(caption))
                elif fetch_trending_pairs_text:
                    bot.send_message(cid, _append_brand(fetch_trending_pairs_text()))
                else:
                    bot.send_message(cid, _append_brand("Trending feature not available (missing market_providers)."))
            except Exception:
                log.exception("trending handler failed")
                bot.send_message(cid, _append_brand("Failed to fetch trending pairs."))
            return

        if data == "bot_status":
            scanner_running = _scanner_thread is not None and _scanner_thread.is_alive()
            msg = f"⚙️ Bot is running ✅\nScanner running: {scanner_running}\nAuto confidence threshold: {AUTO_CONFIDENCE_THRESHOLD*100:.0f}%"
            safe_send_with_image(bot, cid, _append_brand(msg), _maybe_image_for_text([msg]))
            return

        if data == "market_news":
            # quick AI market brief or placeholder
            try:
                txt = fear_and_greed_index() if fear_and_greed_index else "Market news not available."
                safe_send_with_image(bot, cid, _append_brand(txt), _maybe_image_for_text([txt]))
            except Exception:
                log.exception("market_news failed")
                bot.send_message(cid, _append_brand("Market news feature failed."))
            return

        if data == "challenge_status":
            d = load_data() if load_data else {}
            bal = d.get("challenge", {}).get("balance", CHALLENGE_START) if isinstance(d, dict) else CHALLENGE_START
            wins = d.get("challenge", {}).get("wins", 0) if isinstance(d, dict) else 0
            losses = d.get("challenge", {}).get("losses", 0) if isinstance(d, dict) else 0
            text = f"Balance: ${bal:.2f}\nWins: {wins} Losses: {losses}"
            safe_send_with_image(bot, cid, _append_brand(text), _maybe_image_for_text([text]))
            return

        if data == "ask_ai":
            bot.send_message(cid, _append_brand("🤖 Ask AI: send a message starting with `AI:` followed by your question (e.g. `AI: market summary BTC`)"))
            return

        if data == "refresh_bot":
            bot.send_message(cid, _append_brand("🔄 Refreshing bot session..."))
            stop_existing_bot_instances()
            time.sleep(1)
            bot.send_message(cid, _append_brand("✅ Refreshed."))
            return

        if data == "start_auto_brief":
            bot.send_message(cid, _append_brand("▶️ Starting background market scanner (auto-send strong signals)."))
            start_background_scanner()
            return

        if data == "stop_auto_brief":
            bot.send_message(cid, _append_brand("⏹ Stopping background market scanner."))
            stop_background_scanner()
            return

        if data == "pnl_upload":
            bot.send_message(cid, _append_brand("📸 Please send the PnL screenshot as a photo to this chat now. Add optional caption with notes."))
            return

        if data == "history":
            try:
                d = load_data() if load_data else {}
                sigs = d.get("signals", [])[-10:] if isinstance(d, dict) else []
                lines = [f"Last {len(sigs)} signals:"]
                for s in reversed(sigs):
                    lines.append(f"{s['id']} | {s['signal'].get('symbol')} | {s['signal'].get('signal')} | {s['time']}")
                if not lines:
                    bot.send_message(cid, _append_brand("No history found."))
                else:
                    img = _maybe_image_for_text(lines)
                    safe_send_with_image(bot, cid, _append_brand("\n".join(lines)), img)
            except Exception:
                log.exception("history handler failed")
                bot.send_message(cid, _append_brand("Failed to fetch history."))
            return

        # Scheduler-based auto briefs (text/AI summaries)
        if data == "start_auto_brief_scheduler":
            if start_scheduler:
                bot.send_message(cid, _append_brand("▶️ Scheduler for auto-briefs enabled."))
                try:
                    start_scheduler(bot)
                except Exception:
                    log.exception("start_scheduler failed")
                    bot.send_message(cid, _append_brand("Failed to start scheduler."))
            else:
                bot.send_message(cid, _append_brand("Scheduler not available (missing scheduler module)."))
            return

        if data == "stop_auto_brief_scheduler":
            if stop_scheduler:
                try:
                    stop_scheduler()
                    bot.send_message(cid, _append_brand("⏹ Scheduler for auto-briefs disabled."))
                except Exception:
                    log.exception("stop_scheduler failed")
            else:
                bot.send_message(cid, _append_brand("Scheduler not available (missing scheduler module)."))
            return

        if data.startswith("ai_"):
            sig_id = data.split("_", 1)[1]
            d = load_data() if load_data else {}
            rec = next((s for s in d.get("signals", []) if s["id"] == sig_id), None) if isinstance(d, dict) else None
            if not rec:
                bot.send_message(cid, _append_brand("Signal not found"))
                return
            prompt = f"Provide trade rationale, risk controls and a recommended leverage for this trade:\n{rec['signal']}"
            ai_text = ai_analysis_text(prompt) if ai_analysis_text else "AI feature not available"
            safe_send_with_image(bot, cid, _append_brand(f"🤖 AI analysis:\n{ai_text}"), _maybe_image_for_text([ai_text]))
            return

        # Unknown callback
        bot.send_message(cid, _append_brand("Unknown action (callback handler)."))
        log.warning("Unhandled callback: %s", data)

    except Exception:
        log.exception("callback_handler failed")
        try:
            bot.answer_callback_query(call.id, "Handler error")
        except Exception:
            pass

# ----- Background scanner (auto-detect strong signals) -----
def _scanner_loop():
    """
    Scans using detect_strong_signals if available (preferred), otherwise falls back to per-pair engine.
    Auto-sends to ADMIN_ID only (configurable) once a strong candidate is found and respects cooldown.
    """
    log.info("[SCANNER] Background scanner started.")
    while not _scanner_stop_event.is_set():
        try:
            # prefer market_providers.detect_strong_signals
            if detect_strong_signals:
                try:
                    candidates = detect_strong_signals(pairs=PAIRS, timeframes=["15m", "1h", "4h"], min_confidence=AUTO_CONFIDENCE_THRESHOLD)
                except Exception:
                    log.exception("detect_strong_signals invocation failed")
                    candidates = []
                for c in candidates:
                    sym = _normalize_symbol(c.get("symbol"))
                    tf = "1h"
                    if not can_send_signal(sym, tf):
                        continue
                    try:
                        # mark first to avoid duplicates
                        mark_signal_sent(sym, tf)
                        # use existing image if present
                        sig = {
                            "symbol": sym,
                            "interval": tf,
                            "signal": c.get("combined_signal"),
                            "entry": c.get("analysis", {}).get("1h", {}).get("close"),
                            "sl": c.get("sl"),
                            "tp1": c.get("tp1"),
                            "confidence": c.get("combined_score", 0.0),
                            "reasons": []
                        }
                        if c.get("image"):
                            sig["image"] = c.get("image")
                        target = ADMIN_ID if AUTO_SEND_ONLY_ADMIN and ADMIN_ID else None
                        record_signal_and_send(sig, chat_id=target, user_id=ADMIN_ID, auto=True)
                        log.info("[SCANNER] Auto-sent strong signal for %s conf=%.2f", sym, c.get("combined_score", 0.0))
                    except Exception:
                        log.exception("Failed to send candidate")
                    time.sleep(0.4)
            else:
                # fallback: iterate pairs and timeframes, use generate_signal or analysis
                for interval in SCAN_INTERVALS:
                    if _scanner_stop_event.is_set():
                        break
                    for pair in PAIRS:
                        if _scanner_stop_event.is_set():
                            break
                        sym = _normalize_symbol(pair)
                        if not can_send_signal(sym, interval):
                            continue
                        try:
                            sig = _safe_generate_signal(sym, interval)
                            if sig.get("error"):
                                continue
                            s_type = sig.get("signal")
                            conf = float(sig.get("confidence", 0.0)) if sig.get("confidence") is not None else 0.0
                            if s_type in ("LONG", "SHORT", "STRONG_LONG", "STRONG_SHORT") and conf >= AUTO_CONFIDENCE_THRESHOLD:
                                mark_signal_sent(sym, interval)
                                target = ADMIN_ID if AUTO_SEND_ONLY_ADMIN and ADMIN_ID else None
                                record_signal_and_send(sig, chat_id=target, user_id=ADMIN_ID, auto=True)
                                log.info("[SCANNER] Auto-sent %s %s conf=%.2f", sym, interval, conf)
                        except Exception:
                            log.exception("scanner per-pair iteration failed")
                        time.sleep(0.3)
            # small rest so scanner isn't hammering CPU
            time.sleep(2.0)
        except Exception:
            log.exception("Unhandled error in scanner loop")
            time.sleep(1.0)
    log.info("[SCANNER] Background scanner stopped.")

def start_background_scanner():
    global _scanner_thread, _scanner_stop_event
    if _scanner_thread and _scanner_thread.is_alive():
        log.info("[SCANNER] Already running.")
        return
    _scanner_stop_event.clear()
    _scanner_thread = threading.Thread(target=_scanner_loop, daemon=True)
    _scanner_thread.start()
    log.info("[SCANNER] Started background scanner thread.")

def stop_background_scanner():
    global _scanner_thread, _scanner_stop_event
    if not _scanner_thread:
        log.info("[SCANNER] Not running.")
        return
    _scanner_stop_event.set()
    _scanner_thread.join(timeout=5)
    _scanner_thread = None
    log.info("[SCANNER] Stop requested and thread joined.")

# ----- Start polling safely (exported) -----
def start_bot_polling():
    stop_existing_bot_instances()
    log.info("[BOT] Starting polling loop...")
    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=60, skip_pending=True)
        except Exception as e:
            log.exception("[BOT] Polling loop exception")
            # try to recover from 409 by clearing getUpdates
            if "409" in str(e):
                log.warning("[BOT] 409 Conflict - attempting to stop other sessions and retry")
                stop_existing_bot_instances()
                time.sleep(5)
            else:
                time.sleep(5)
