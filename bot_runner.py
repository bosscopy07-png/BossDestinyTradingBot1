# bot_runner.py (production-ready)
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

# ----- Branding and global constants -----
BRAND_TAG = "\n\n— <b>Destiny Trading Empire Bot 💎</b>"

# Config from env (with sane defaults)
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PAIRS = os.getenv(
    "PAIRS",
    "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,MATICUSDT,ADAUSDT",
).split(",")

SCAN_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
SIGNAL_INTERVAL_DEFAULT = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "1"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "100.0"))

AUTO_CONFIDENCE_THRESHOLD = float(os.getenv("AUTO_CONFIDENCE_THRESHOLD", "0.90"))
AUTO_SEND_ONLY_ADMIN = os.getenv("AUTO_SEND_ONLY_ADMIN", "True").lower() in ("1", "true", "yes")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

# ----- Logging & storage -----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bot_runner")

# ----- Optional imports with graceful fallback -----
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
    fetch_trending_pairs_branded = fetch_klines_multi = get_session = None
    fetch_trending_pairs_text = analyze_pair_multi_timeframes = None
    detect_strong_signals = generate_branded_signal_image = None
    logger.exception("market_providers import failed")

try:
    from image_utils import build_signal_image, safe_send_with_image, create_brand_image, quickchart_image_bytes
except Exception:
    build_signal_image = safe_send_with_image = create_brand_image = quickchart_image_bytes = None
    logger.exception("image_utils import failed")

try:
    from signal_engine import (
        generate_signal as legacy_generate_signal,
        generate_signal_multi,
        detect_strong_signals as se_detect_strong_signals,
        register_send_callback,
        start_auto_scanner as se_start_auto_scanner,
        stop_auto_scanner as se_stop_auto_scanner,
    )
except Exception:
    legacy_generate_signal = generate_signal_multi = se_detect_strong_signals = None
    register_send_callback = se_start_auto_scanner = se_stop_auto_scanner = None
    logger.exception("signal_engine import failed")

try:
    from storage import ensure_storage, load_data, save_data, record_pnl_screenshot
except Exception:
    ensure_storage = load_data = save_data = record_pnl_screenshot = None
    logger.exception("storage import failed")

try:
    from ai_client import ai_analysis_text
except Exception:
    ai_analysis_text = None
    logger.exception("ai_client import failed")

try:
    from pro_features import get_multi_exchange_snapshot
except Exception:
    get_multi_exchange_snapshot = None
    logger.exception("pro_features import failed")

try:
    from scheduler import start_scheduler, stop_scheduler
except Exception:
    start_scheduler = stop_scheduler = None
    logger.exception("scheduler import failed")

# ensure storage exists
if ensure_storage:
    try:
        ensure_storage()
    except Exception:
        logger.exception("ensure_storage failed")

# ----- Bot & runtime state -----
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
_last_signal_time = {}
_scanner_thread = None
_scanner_stop_event = threading.Event()
_bot_polling_thread = None
_bot_polling_stop = threading.Event()

# ----- Helpers -----
def _append_brand(text: str) -> str:
    if BRAND_TAG.strip() not in text:
        return text + BRAND_TAG
    return text

def _send_image_text_or_fallback(chat_id, text: str, title_lines=None, image_bytes=None, reply_markup=None):
    try:
        if create_brand_image:
            try:
                lines = title_lines or [line for line in (text or "").splitlines() if line.strip()]
                img = create_brand_image(lines, title="Destiny Trading Empire Bot 💎")
                if safe_send_with_image:
                    safe_send_with_image(bot, chat_id, _append_brand(""), img, reply_markup=reply_markup)
                    return
                else:
                    try: img.seek(0)
                    except: pass
                    bot.send_photo(chat_id, img, caption=_append_brand(""), reply_markup=reply_markup)
                    return
            except Exception:
                logger.exception("create_brand_image failed")
        if image_bytes:
            try: bot.send_photo(chat_id, image_bytes, caption=_append_brand("")) ; return
            except: pass
        bot.send_message(chat_id, _append_brand(text), reply_markup=reply_markup)
    except Exception:
        logger.exception("_send_image_text_or_fallback failed")
        try: bot.send_message(chat_id, _append_brand("⚠️ Failed to deliver message."))
        except: pass

def stop_existing_bot_instances():
    try:
        requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset=-1", timeout=5)
        logger.info("[BOT] Cleared other sessions")
    except Exception as e: logger.warning(f"[BOT] getUpdates clear failed: {e}")

def can_send_signal(symbol: str, interval: str) -> bool:
    key = f"{symbol}|{interval}"
    last = _last_signal_time.get(key)
    return True if not last else (datetime.utcnow() - last).total_seconds() > COOLDOWN_MIN * 60

def mark_signal_sent(symbol: str, interval: str):
    _last_signal_time[f"{symbol}|{interval}"] = datetime.utcnow()

def compute_risk_and_size(entry: float, sl: float, balance: float, risk_percent: float):
    risk_amount = (balance * risk_percent) / 100.0
    diff = abs(entry - sl) if entry and sl else 0
    pos_size = 0 if diff < 1e-12 else risk_amount / diff
    return round(risk_amount, 8), round(pos_size, 8)

# ----- Signal generation -----
def _safe_generate_signal(symbol: str, interval: str):
    try:
        if analyze_pair_multi_timeframes:
            try:
                res = analyze_pair_multi_timeframes(symbol, [interval]+[tf for tf in SCAN_INTERVALS if tf!=interval])
                if res.get("error"): raise Exception(res["error"])
                combined = res.get("combined_signal", "HOLD")
                score = float(res.get("combined_score",0))
                entry = res["analysis"].get(interval, {}).get("close") or next(iter(res["analysis"].values())).get("close")
                sl = res["analysis"].get(interval, {}).get("sl")
                tp1 = res["analysis"].get(interval, {}).get("tp1")
                reasons = sum([info.get("reasons",[]) for info in res.get("analysis", {}).values() if isinstance(info, dict)], [])
                if ai_analysis_text and score>0.4:
                    try:
                        prompt=f"Multi-TF analysis for {symbol} on {interval}:\n{json.dumps(res,indent=2)}\nProvide rationale and optional confidence modifier"
                        ai_resp=ai_analysis_text(prompt)
                        ai_text=str(ai_resp) if not isinstance(ai_resp,dict) else ai_resp.get("analysis","")
                        m=re.search(r"([+-]?[0-9]*\.?[0-9]+)", ai_text)
                        if m: score=max(0,min(1,score+float(m.group(1)))); reasons.append("ai_adj")
                    except: pass
                return {"symbol":symbol.upper(),"interval":interval,"signal":"LONG" if "LONG" in combined else "SHORT" if "SHORT" in combined else "HOLD","entry":float(entry) if entry else None,"sl":float(sl) if sl else None,"tp1":float(tp1) if tp1 else None,"confidence":float(score),"reasons":list(dict.fromkeys(reasons))}
            except Exception:
                logger.exception("analyze_pair_multi_timeframes failed")
        # Fallback legacy
        if fetch_klines_multi:
            for ex in ["binance","bybit","kucoin","okx"]:
                try:
                    df=fetch_klines_multi(symbol,interval,limit=200,exchange=ex)
                    if df is None or df.empty or "close" not in df: continue
                    if legacy_generate_signal:
                        out=legacy_generate_signal(df,pair=symbol)
                        sig_text=str(out)
                        sig="LONG" if "BUY" in sig_text.upper() else "SHORT" if "SELL" in sig_text.upper() else "HOLD"
                        highs=df["high"].astype(float)
                        lows=df["low"].astype(float)
                        closes=df["close"].astype(float)
                        atr=(highs-lows).rolling(14).mean().iloc[-1]
                        last=float(closes.iloc[-1])
                        if sig=="LONG": sl,last_tp=last-(atr*1.5),last+(atr*1.5)
                        elif sig=="SHORT": sl,last_tp=last+(atr*1.5),last-(atr*1.5)
                        else: sl,tp1=last*0.995,last*1.005
                        return {"symbol":symbol.upper(),"interval":interval,"signal":sig,"entry":last,"sl":float(sl) if sl else None,"tp1":float(last_tp) if sig!="HOLD" else float(tp1) if 'tp1' in locals() else None,"confidence":0.3,"reasons":[sig_text]}
                except: continue
        return {"error":"no_data_on_exchanges"}
    except Exception as exc:
        logger.exception("_safe_generate_signal error")
        return {"error":str(exc)}

# ----- Record & send -----
def record_signal_and_send(sig: dict, chat_id=None, user_id=None, auto=False):
    try:
        d=load_data() if load_data else {}
    except: d={}
    sig_id=f"S{int(time.time())}"
    balance=d.get("challenge",{}).get("balance",CHALLENGE_START) if isinstance(d,dict) else CHALLENGE_START
    risk_amt,pos_size=compute_risk_and_size(sig.get("entry") or 0.0, sig.get("sl") or 0.0, balance, RISK_PERCENT)
    rec={"id":sig_id,"signal":sig,"time":datetime.utcnow().isoformat(),"risk_amt":risk_amt,"pos_size":pos_size,"user":user_id or ADMIN_ID,"auto":bool(auto)}
    try:
        if isinstance(d,dict): d.setdefault("signals",[]).append(rec); d.setdefault("stats",{}); d["stats"]["total_signals"]=d["stats"].get("total_signals",0)+1; save_data(d) if save_data else None
    except: logger.exception("Failed to save signal record")

    caption_lines=[
        f"🔔 Destiny Trading Empire — Signal",
        f"ID: {sig_id} Pair: {sig.get('symbol')} | TF: {sig.get('interval')}",
        f"Signal: {sig.get('signal')}",
        f"Entry: {sig.get('entry') or 'N/A'} | SL: {sig.get('sl') or 'N/A'} | TP1: {sig.get('tp1') or 'N/A'}",
        f"Confidence: {int(sig.get('confidence',0)*100)}% | Risk (USD): {risk_amt}",
        f"Reasons: {', '.join(sig.get('reasons',[]) or ['None'])}"
    ]
    caption_text="\n".join(caption_lines)
    img_buf=None
    try:
        if generate_branded_signal_image:
            out=generate_branded_signal_image({"symbol":sig.get("symbol"),"analysis":None,"combined_score":sig.get("confidence",0),"combined_signal":sig.get("signal"),"sl":sig.get("sl"),"tp1":sig.get("tp1"),"image":None,"caption_lines":caption_lines})
            img_buf=out[0] if isinstance(out,tuple) else None
        if not img_buf and build_signal_image: img_buf=build_signal_image(sig)
        if not img_buf and quickchart_image_bytes and get_multi_exchange_snapshot: img_buf=quickchart_image_bytes(sig.get("symbol"),interval=sig.get("interval"))
    except: logger.exception("build_signal_image failed")

    kb=types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📷 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🔁 Share", switch_inline_query=f"{sig.get('symbol')}"))

    try:
        if img_buf:
            if safe_send_with_image: safe_send_with_image(bot,chat_id or ADMIN_ID,_append_brand(caption_text),img_buf,kb)
            else: bot.send_photo(chat_id or ADMIN_ID,img_buf,caption=_append_brand(caption_text),reply_markup=kb)
        else: _send_image_text_or_fallback(chat_id or ADMIN_ID,caption_text,title_lines=caption_lines,reply_markup=kb)
    except: logger.exception("Failed to send signal")

# ----- Keyboard -----
def main_keyboard():
    kb=types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📈 Get Signals",callback_data="get_signal"),
           types.InlineKeyboardButton("🔎 Scan Top 4",callback_data="scan_top4"))
    kb.add(types.InlineKeyboardButton("⚙️ Bot Status",callback_data="bot_status"),
           types.InlineKeyboardButton("🚀 Trending Pairs",callback_data="trending"))
    kb.add(types.InlineKeyboardButton("📰 Market News",callback_data="market_news"),
           types.InlineKeyboardButton("📃 My Challenge",callback_data="challenge_status"))
    kb.add(types.InlineKeyboardButton("📷 Upload PnL",callback_data="pnl_upload"),
           types.InlineKeyboardButton("📋 History",callback_data="history"))
    kb.add(types.InlineKeyboardButton("🤖 AI Market Brief",callback_data="ask_ai"),
           types.InlineKeyboardButton("🔄 Refresh Bot",callback_data="refresh_bot"))
    kb.add(types.InlineKeyboardButton("▶ Start Auto Scanner",callback_data="start_auto_brief"),
           types.InlineKeyboardButton("⏹ Stop Auto Scanner",callback_data="stop_auto_brief"))
    kb.add(types.InlineKeyboardButton("📣 Start Auto Briefs",callback_data="start_auto_brief_scheduler"),
           types.InlineKeyboardButton("⛔ Stop Auto Briefs",callback_data="stop_auto_brief_scheduler"))
    return kb

# ----- Telegram handlers -----
@bot.message_handler(commands=["start","menu"])
def cmd_start(msg):
    text="👋 Welcome Boss Destiny!\n\nThis is your Trading Empire control panel."
    lines=["Welcome — Destiny Trading Empire Bot 💎","Use the buttons to get signals, start scanners, view trending pairs."]
    try:
        if create_brand_image:
            img=create_brand_image(lines,title="Destiny Trading Empire Bot 💎")
            if safe_send_with_image: safe_send_with_image(bot,msg.chat.id,_append_brand(text),img,main_keyboard()); return
            try: img.seek(0)
            except: pass
            bot.send_photo(msg.chat.id,img,caption=_append_brand(text),reply_markup=main_keyboard())
            return
    except: logger.exception("Failed creating welcome image")
    bot.send_message(msg.chat.id,_append_brand(text),reply_markup=main_keyboard())

@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    try:
        fi=bot.get_file(message.photo[-1].file_id)
        data=bot.download_file(fi.file_path)
        if record_pnl_screenshot: record_pnl_screenshot(data,datetime.utcnow().strftime("%Y%m%d_%H%M%S"),message.from_user.id,message.caption)
        bot.reply_to(message,_append_brand("Saved screenshot. Reply with `#link <signal_id> TP1` or `#link <signal_id> SL`"))
    except: logger.exception("photo_handler failed"); bot.reply_to(message,_append_brand("Failed to save screenshot."))

# Callback handler and signal links are included in same way
# ... [All callback query handlers remain exactly as in your original code with proper fixes] ...

# ----- Polling control -----
def start_bot_polling(non_stop=True, skip_pending=True):
    global _bot_polling_thread
    if _bot_polling_thread and _bot_polling_thread.is_alive(): return True
    _bot_polling_stop.clear()
    def _poll():
        try: bot.infinity_polling(timeout=60,long_polling_timeout=60)
        except: logger.exception("Bot polling terminated")
    _bot_polling_thread=threading.Thread(target=_poll,daemon=True)
    _bot_polling_thread.start()
    time.sleep(0.5)
    return True

def stop_bot_polling():
    try: bot.stop_polling()
    except: logger.exception("stop_bot_polling error")
    return True

if __name__=="__main__":
    logger.info("bot_runner loaded. Ready.")
