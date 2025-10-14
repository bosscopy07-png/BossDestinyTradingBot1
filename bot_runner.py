import os
import time
import json
import traceback
import threading
from datetime import datetime
from io import BytesIO

import telebot
from telebot import types

from pro_features import top_gainers_pairs, fear_and_greed_index, quickchart_price_image, futures_leverage_suggestion, ai_market_brief_text
from scheduler import start_scheduler, stop_scheduler
from storage import ensure_storage, load_data, save_data, record_pnl_screenshot
from market_providers import fetch_klines_df, fetch_trending_pairs
from signal_engine import generate_signal_for
from ai_client import ai_analysis_text
from image_utils import build_signal_image, safe_send_with_image

# --- Config from env ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT").split(",")
SIGNAL_INTERVAL = os.getenv("SIGNAL_INTERVAL", "1h")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "5"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required")

# Ensure storage file exists
ensure_storage()

# Create bot instance and remove webhook (prevent 409)
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
try:
    bot.remove_webhook()
except Exception:
    pass

# last sent times to enforce cooldown
_last_signal_time = {}

def can_send_signal(sym):
    now = datetime.utcnow()
    last = _last_signal_time.get(sym)
    if not last:
        return True
    diff = (now - last).total_seconds()
    return diff > COOLDOWN_MIN * 60

def record_signal_and_send(sig_record, chat_id=None, user_id=None):
    d = load_data()
    sig_id = f"S{int(time.time())}"
    balance = d.get("challenge", {}).get("balance", float(os.getenv("CHALLENGE_START", "10")))
    risk_usd = sig_record.get("suggested_risk_usd", round(balance * RISK_PERCENT / 100.0, 8))
    rec = {
        "id": sig_id,
        "signal": sig_record,
        "time": datetime.utcnow().isoformat(),
        "risk_amt": risk_usd,
        "result": None,
        "posted_by": user_id or ADMIN_ID
    }
    d["signals"].append(rec)
    d["stats"]["total_signals"] = d["stats"].get("total_signals", 0) + 1
    save_data(d)
    _last_signal_time[sig_record["symbol"]] = datetime.utcnow()

    # prepare message & image
    caption = (f"🔥 <b>Boss Destiny Trading Empire — Signal</b>\n"
               f"ID: {sig_id}\nPair: {sig_record['symbol']} | TF: {sig_record['interval']}\n"
               f"Signal: <b>{sig_record['signal']}</b>\nEntry: {sig_record['entry']} | SL: {sig_record['sl']} | TP1: {sig_record.get('tp1')}\n"
               f"Confidence: {int(sig_record.get('confidence',0)*100)}% | Risk (USD): {risk_usd}\n"
               f"Reasons: {', '.join(sig_record.get('reasons',[])) if sig_record.get('reasons') else 'None'}\n\n"
               f"— Boss Destiny Trading Empire")
    img = build_signal_image(sig_record)
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
    target = chat_id or ADMIN_ID
    safe_send_with_image(bot, target, caption, img, kb)
    return sig_id

# --- Handlers and UI ---
def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📈 Get Signal", callback_data="get_signal"),
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
        types.InlineKeyboardButton("🤖 Ask AI", callback_data="ask_ai"),
        types.InlineKeyboardButton("🔄 Refresh Bot", callback_data="refresh_bot")
    )
    kb.row(
        types.InlineKeyboardButton("🚀 Top Movers", callback_data="top_gainers"),
        types.InlineKeyboardButton("📈 Fear & Greed", callback_data="fear_greed")
    )
    kb.row(
        types.InlineKeyboardButton("🖼️ Quick Chart", callback_data="open_chart_menu"),
        types.InlineKeyboardButton("⚖️ Futures Suggest", callback_data="open_fut_menu")
    )
    # Add admin-only quick toggles
    kb.row(
        types.InlineKeyboardButton("▶️ Start Auto Brief", callback_data="start_auto_brief"),
        types.InlineKeyboardButton("⏹ Stop Auto Brief", callback_data="stop_auto_brief")
    )
    return kb

@bot.message_handler(commands=['start','menu'])
def cmd_start(msg):
    text = f"Welcome — Boss Destiny Trading Empire\nTap a button:"
    bot.send_message(msg.chat.id, text, reply_markup=main_keyboard())

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(fi.file_path)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = record_pnl_screenshot(data, now, message.from_user.id, message.caption)
        bot.reply_to(message, "Screenshot saved. To link: reply with '#link <signal_id> TP1' or '#link <signal_id> SL'")
    except Exception:
        traceback.print_exc()
        bot.reply_to(message, "Failed to save screenshot.")

@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_handler(message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 3:
            bot.reply_to(message, "Usage: #link <signal_id> TP1 or SL"); return
        sig_id = parts[1]; tag = parts[2].upper()
        d = load_data()
        pnl_item = None
        for p in reversed(d.get("pnl", [])):
            if p.get("linked") is None and p["from"] == message.from_user.id:
                pnl_item = p; break
        if not pnl_item:
            bot.reply_to(message, "No unlinked screenshot found."); return
        pnl_item["linked"] = {"signal_id": sig_id, "result": tag, "linked_by": message.from_user.id}
        # admin confirms updates challenge
        if message.from_user.id == ADMIN_ID:
            srec = next((s for s in d.get("signals", []) if s["id"] == sig_id), None)
            if srec:
                risk = srec.get("risk_amt", 0)
                if tag.startswith("TP"):
                    d["challenge"]["balance"] = d["challenge"].get("balance", 0) + risk
                    d["challenge"]["wins"] = d["challenge"].get("wins", 0) + 1
                    srec["result"] = tag
                elif tag == "SL":
                    d["challenge"]["balance"] = d["challenge"].get("balance", 0) - risk
                    d["challenge"]["losses"] = d["challenge"].get("losses", 0) + 1
                    srec["result"] = "SL"
        save_data(d)
        bot.reply_to(message, f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance.")
    except Exception:
        traceback.print_exc(); bot.reply_to(message, "Failed to link screenshot.")

@bot.callback_query_handler(func=lambda c: True)
def callback_handler(call):
    try:
        data = call.data; cid = call.message.chat.id
        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"sig_{p}"))
            bot.send_message(cid, "Choose pair:", reply_markup=kb); return

        if data.startswith("sig_"):
            pair = data.split("_",1)[1]
            bot.send_chat_action(cid, "typing")
            sig = generate_signal_for(pair, SIGNAL_INTERVAL)
            if sig.get("error"):
                bot.send_message(cid, f"Error generating signal: {sig['error']}"); return
            record_signal_and_send(sig, chat_id=cid, user_id=call.from_user.id)
            bot.answer_callback_query(call.id, "Signal generated"); return

        if data == "scan_top4":
            # scan PAIRS top 4 by order
            top = PAIRS[:4]
            for p in top:
                if can_send_signal(p):
                    sig = generate_signal_for(p, SIGNAL_INTERVAL)
                    if sig.get("error"): continue
                    record_signal_and_send(sig, chat_id=cid)
                    time.sleep(1)
            bot.answer_callback_query(call.id, "Scan complete"); return

        if data == "trending":
            txt = fetch_trending_pairs()
            bot.send_message(cid, txt); return

        if data.startswith("ai_"):
            sig_id = data.split("_",1)[1]; d = load_data()
            rec = next((s for s in d.get("signals", []) if s["id"]==sig_id), None)
            if not rec:
                bot.send_message(cid, "Signal not found."); return
            prompt = f"Provide trade rationale, risk management and an explicit BUY/SELL verdict for this signal:\n{json.dumps(rec['signal'], indent=2)}"
            analysis = ai_analysis_text(prompt)
            bot.send_message(cid, f"🤖 AI — Boss Destiny Trading Empire:\n{analysis}"); return

        if data == "pnl_upload":
            bot.send_message(cid, "Upload PnL screenshot now; then link with: #link <signal_id> TP1/SL"); return

        if data == "challenge_status":
            d = load_data(); c = d.get("challenge", {})
            bot.send_message(cid, f"🏆 Boss Destiny Trading Empire — Challenge\nBalance: ${c.get('balance',0):.2f}\nWins: {c.get('wins',0)} Losses: {c.get('losses',0)}"); return

        if data == "history":
            d = load_data(); recs = d.get("signals", [])[-30:]
            if not recs:
                bot.send_message(cid, "No signals yet."); return
            txt = "Recent signals:\n" + "\n".join([f"{r['id']} {r['signal']['symbol']} {r['signal']['signal']} conf:{int(r['signal'].get('confidence',0)*100)}% res:{r.get('result') or '-'}" for r in recs[::-1]])
            bot.send_message(cid, txt); return

        if data == "ask_ai":
            bot.send_message(cid, "Type message starting with 'AI: ' followed by your question."); return

        if data == "refresh_bot":
            bot.send_message(cid, "🔄 Boss Destiny Trading Empire — Bot refreshed."); return

        if data == "bot_status":
            bot.send_message(cid, f"🤖 Bot active. Time: {datetime.utcnow().isoformat()}"); return
            
        if data == "top_gainers":
            txt = top_gainers_pairs()
            bot.send_message(cid, f"📊 Boss Destiny Top Movers:\n\n{txt}")
            return

        if data == "fear_greed":
            txt = fear_and_greed_index()
            bot.send_message(cid, f"📈 Boss Destiny Fear & Greed:\n\n{txt}")
            return

        if data.startswith("chart_"):
            # callback data like "chart_BTCUSDT"
            pair = data.split("_",1)[1]
            img_bytes, err = quickchart_price_image(pair, interval="1h", points=60)
            if img_bytes:
                bot.send_photo(cid, img_bytes, caption=f"📈 {pair} — Boss Destiny Trading Empire")
            else:
                bot.send_message(cid, f"Chart error: {err}")
            return

        if data.startswith("fut_"):
            pair = data.split("_",1)[1]
            suggestion = futures_leverage_suggestion(pair)
            bot.send_message(cid, f"⚖️ Futures suggestion for {pair}:\n{suggestion.get('suggestion')}")
            return

        if data == "start_auto_brief":
            # admin only
            if call.from_user.id != ADMIN_ID:
                bot.answer_callback_query(call.id, "Admins only")
                return
            start_scheduler(ADMIN_ID, interval_hours=4)
            bot.send_message(cid, "🟢 Auto AI Market Brief started (every 4h).")
            return

        if data == "stop_auto_brief":
            if call.from_user.id != ADMIN_ID:
                bot.answer_callback_query(call.id, "Admins only")
                return
            stop_scheduler()
            bot.send_message(cid, "🔴 Auto AI Market Brief stopped.")
            return

        bot.answer_callback_query(call.id, "Unknown action")
    except Exception:
        traceback.print_exc()
        try: bot.answer_callback_query(call.id, "Handler error")
        except: pass

@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().upper().startswith("AI:"))
def ai_text_handler(message):
    prompt = message.text.strip()[3:].strip()
    if not prompt:
        bot.reply_to(message, "AI: provide a question after 'AI:'")
        return
    bot.send_chat_action(message.chat.id, "typing")
    ans = ai_analysis_text(prompt)
    bot.send_message(message.chat.id, f"🤖 AI — Boss Destiny Trading Empire:\n{ans}")

@bot.message_handler(func=lambda m: True)
def fallback(message):
    bot.send_message(message.chat.id, "Tap a button to start - Boss Destiny Trading Empire", reply_markup=main_keyboard())

# Public functions to be used by main.py
def stop_existing_bot_instances():
    try:
        bot.remove_webhook()
    except Exception:
        pass

def start_health_server(port=8080):
    # minimal Flask server to bind port if needed
    try:
        from flask import Flask, jsonify
        app = Flask("boss_destiny_health")
        @app.route("/", methods=["GET"])
        def root():
            return jsonify({"service":"boss_destiny_bot","time": datetime.utcnow().isoformat()})
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", port)))
    except Exception:
        traceback.print_exc()

def start_bot_polling():
    # start polling; blocking call
    try:
        stop_existing_bot_instances()
        print("Starting polling (bot_process.start_bot_polling)...")
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception:
        traceback.print_exc()
        time.sleep(5)
        start_bot_polling()
        # ai_client.py
import os
import traceback
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

def ai_analysis_text(prompt):
    if not client:
        return "AI unavailable (OPENAI_API_KEY not set)."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional crypto market analyst. Provide concise trade rationale, risk controls, and a one-line BUY/SELL verdict."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return f"AI error: {e}"
        # market_providers.py
import os
import time
import traceback
from datetime import datetime
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
COINGECKO_MAP = {"BTCUSDT":"bitcoin","ETHUSDT":"ethereum","BNBUSDT":"binancecoin","SOLUSDT":"solana","XRPUSDT":"ripple","DOGEUSDT":"dogecoin"}

def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent":"BossDestiny/1.0"})
    proxy = os.getenv("PROXY_URL")
    if proxy:
        s.proxies.update({"http":proxy,"https":proxy})
    retries = Retry(total=3, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=frozenset(["GET","POST"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def normalize_interval(s):
    if not s: return "1h"
    s2 = s.strip().lower()
    return s2 if s2 in {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"} else "1h"

def fetch_klines_binance(symbol="BTCUSDT", interval="1h", limit=300):
    sess = get_session()
    params = {"symbol": symbol.upper(), "interval": normalize_interval(interval), "limit": int(limit)}
    r = sess.get(BINANCE_KLINES, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def fetch_klines_coingecko(symbol="BTCUSDT", interval="1h", limit=300):
    coin = COINGECKO_MAP.get(symbol.upper())
    if not coin:
        raise RuntimeError("Coingecko fallback not available for symbol " + symbol)
    sess = get_session()
    r = sess.get(COINGECKO_MARKET_CHART.format(id=coin), params={"vs_currency":"usd","days":7}, timeout=10)
    r.raise_for_status()
    js = r.json()
    prices = js.get("prices", [])[-limit:]
    rows = []
    for p in prices:
        ts = int(p[0]); price = float(p[1])
        rows.append([pd.to_datetime(ts, unit="ms"), price, price, price, price, 0.0])
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume"])
    return df

def fetch_klines_df(symbol="BTCUSDT", interval="1h", limit=300):
    errs = []
    try:
        return fetch_klines_binance(symbol, interval, limit)
    except Exception as e:
        errs.append("Binance:" + str(e))
        time.sleep(0.3)
    try:
        return fetch_klines_coingecko(symbol, interval, limit)
    except Exception as e:
        errs.append("Coingecko:" + str(e))
    raise RuntimeError("All providers failed: " + " | ".join(errs))

def fetch_trending_pairs():
    try:
        sess = get_session()
        r = sess.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8)
        r.raise_for_status()
        tickers = r.json()
        pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
        out = []
        for p in pairs:
            t = next((x for x in tickers if x.get("symbol") == p), None)
            if t:
                out.append(f"{p}: {float(t.get('priceChangePercent', 0)):.2f}% vol:{int(float(t.get('quoteVolume', 0))):,}")
        return "Trending Pairs:\n" + "\n".join(out)
    except Exception:
        traceback.print_exc()
        return "Failed to fetch trending pairs."
        # signal_engine.py
import os
import traceback
from datetime import datetime
import numpy as np

from market_providers import fetch_klines_df

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    mc = ef - es
    msig = mc.ewm(span=signal, adjust=False).mean()
    hist = mc - msig
    return mc, msig, hist

def generate_signal_for(symbol="BTCUSDT", interval="1h"):
    try:
        df = fetch_klines_df(symbol, interval, limit=300)
        if df is None or len(df) < 20:
            return {"error": "insufficient data"}
        fast = int(os.getenv("EMA_FAST", "9"))
        slow = int(os.getenv("EMA_SLOW", "21"))
        rsip = int(os.getenv("RSI_PERIOD", "14"))
        df["ema_fast"] = ema(df["close"], fast)
        df["ema_slow"] = ema(df["close"], slow)
        df["rsi"] = rsi(df["close"], rsip)
        mc, msig, mh = macd(df["close"])
        df["macd_hist"] = mh
        last = df.iloc[-1]; prev = df.iloc[-2]
        signal = None; reasons = []; score = 0.0
        if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
            signal = "BUY"; reasons.append("EMA cross up"); score += 0.3
        if prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
            signal = "SELL"; reasons.append("EMA cross down"); score += 0.3
        if last["macd_hist"] > 0: score += 0.1; reasons.append("MACD > 0")
        r = float(last["rsi"])
        if signal == "BUY" and r > 80:
            reasons.append("High RSI"); score -= 0.12
        if signal == "SELL" and r < 20:
            reasons.append("Low RSI"); score -= 0.12
        price = float(last["close"])
        if signal == "BUY":
            sl = float(df["low"].iloc[-3])
            tp1 = price + (price - sl) * 1.5
        elif signal == "SELL":
            sl = float(df["high"].iloc[-3])
            tp1 = price - (sl - price) * 1.5
        else:
            sl = price * 0.995; tp1 = price * 1.005
        confidence = max(0.05, min(0.98, 0.5 + score))
        # risk sizing
        try:
            from storage import load_data
            d = load_data()
            balance = d.get("challenge", {}).get("balance", float(os.getenv("CHALLENGE_START", "10")))
        except Exception:
            balance = float(os.getenv("CHALLENGE_START", "10"))
        risk_pct = float(os.getenv("RISK_PERCENT", "5"))
        risk_usd = round((balance * risk_pct) / 100.0, 8)
        diff = abs(price - sl) if abs(price - sl) > 1e-12 else 1e-12
        units = round(risk_usd / diff, 8)
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "timestamp": datetime.utcnow().isoformat(),
            "signal": signal or "HOLD",
            "entry": round(price, 8),
            "sl": round(sl, 8),
            "tp1": round(tp1, 8),
            "confidence": round(confidence, 2),
            "reasons": reasons,
            "suggested_risk_usd": risk_usd,
            "suggested_units": units
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
        # image_utils.py
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

BRAND = "Boss Destiny Trading Empire"

def _get_font(size=18, bold=False):
    try:
        if bold:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def create_brand_image(lines):
    w, h = 900, 360
    img = Image.new("RGB", (w, h), (12, 14, 20))
    draw = ImageDraw.Draw(img)
    title_font = _get_font(22, bold=True)
    text_font = _get_font(16)
    draw.text((20, 12), BRAND, fill=(255, 200, 0), font=title_font)
    y = 56
    line_h = 28
    for line in lines:
        # Use textbbox to avoid .textsize errors
        try:
            draw.text((20, y), str(line), fill=(230, 230, 230), font=text_font)
        except Exception:
            draw.text((20, y), str(line), fill=(230, 230, 230))
        y += line_h
    bio = BytesIO()
    img.save(bio, "PNG")
    bio.seek(0)
    return bio

def build_signal_image(sig):
    lines = [
        f"{sig.get('symbol')}  |  {sig.get('interval')}  |  {sig.get('signal')}",
        f"Entry: {sig.get('entry')}   SL: {sig.get('sl')}   TP1: {sig.get('tp1')}",
        f"Confidence: {int(sig.get('confidence', 0)*100)}%   Risk USD: {sig.get('suggested_risk_usd')}",
        "Reasons: " + (", ".join(sig.get('reasons', [])) if sig.get('reasons') else "None")
    ]
    return create_brand_image(lines)

def safe_send_with_image(bot, chat_id, text, image_buf=None, reply_markup=None):
    try:
        if image_buf:
            bot.send_photo(chat_id, image_buf, caption=text, reply_markup=reply_markup)
        else:
            bot.send_message(chat_id, text, reply_markup=reply_markup)
    except Exception:
        try:
            bot.send_message(chat_id, text)
        except:
            pass
            
