# main.py
import os
import threading
import time
from bot_core import start_bot_polling, start_flask_app, stop_existing_bot_instances

MODE = os.getenv("RUN_MODE", "background").lower()  # "background" or "web"
PORT = int(os.getenv("PORT", "8080"))

if __name__ == "__main__":
    # Stop any previous webhook (safety)
    stop_existing_bot_instances()

    if MODE == "web":
        # Start telegram polling in background and also bind PORT (Render web service)
        t = threading.Thread(target=start_bot_polling, daemon=True)
        t.start()
        start_flask_app(port=PORT)   # will block and bind port
    else:
        # Background worker mode (no port required). Use Render Background Worker service.
        print("Starting bot in BACKGROUND mode (no web port binding).")
        start_bot_polling()
        # bot_core.py
import os
import time
import json
import traceback
from datetime import datetime
from io import BytesIO
import threading

import telebot
from telebot import types

from market import generate_signal_for, fetch_trending_pairs
from ai import ai_analysis_text
from storage import load_data, save_data, ensure_storage, record_pnl_screenshot
from utils import build_signal_image, safe_send

# --- config from env ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment var not set")

# ensure storage exists
ensure_storage()

# create bot and prevent webhook conflict
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

def stop_existing_bot_instances():
    # attempt to remove webhook to avoid 409 conflict
    try:
        bot.remove_webhook()
    except Exception:
        pass

# Keyboard
def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📈 Get Signal", callback_data="get_signal"),
        types.InlineKeyboardButton("🧠 AI Opinion", callback_data="ask_ai"),
    )
    kb.add(
        types.InlineKeyboardButton("🔥 Trending Pairs", callback_data="trending"),
        types.InlineKeyboardButton("📰 Market News", callback_data="market_news"),
    )
    kb.add(
        types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status"),
        types.InlineKeyboardButton("🧾 History", callback_data="history"),
    )
    kb.add(
        types.InlineKeyboardButton("🔄 Refresh", callback_data="refresh_bot"),
        types.InlineKeyboardButton("⚙️ Bot Status", callback_data="bot_status"),
    )
    return kb

# send message helper (handles photos)
def send_with_brand(chat_id, caption, image_buf=None, reply_markup=None):
    if image_buf:
        try:
            bot.send_photo(chat_id, image_buf, caption=caption, reply_markup=reply_markup)
            return
        except Exception:
            traceback.print_exc()
    bot.send_message(chat_id, caption, reply_markup=reply_markup)

# Callback handler
@bot.callback_query_handler(func=lambda c: True)
def callback_handler(call):
    try:
        data = call.data
        chat_id = call.message.chat.id

        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
            for p in pairs:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"sig_{p}"))
            bot.send_message(chat_id, "Choose pair:", reply_markup=kb)
            return

        if data.startswith("sig_"):
            pair = data.split("_",1)[1]
            bot.send_chat_action(chat_id, "typing")
            sig = generate_signal_for(pair, os.getenv("SIGNAL_INTERVAL","1h"))
            if sig.get("error"):
                bot.send_message(chat_id, f"Signal error: {sig['error']}")
                return
            # build image & message
            img = build_signal_image(sig)
            caption = (f"🔥 <b>Boss Destiny Signal</b>\nPair: {sig['symbol']} | TF: {sig['interval']}\n"
                       f"Signal: <b>{sig['signal']}</b>\nEntry: {sig['entry']}  SL: {sig['sl']}  TP1: {sig['tp1']}\n"
                       f"Confidence: {int(sig.get('confidence',0)*100)}%\nRisk USD: {sig.get('suggested_risk_usd')}\n")
            # record
            d = load_data()
            sig_id = f"S{int(time.time())}"
            rec = {"id":sig_id,"signal":sig,"time":datetime.utcnow().isoformat(),"risk_amt":sig.get("suggested_risk_usd",0),"result":None}
            d["signals"].append(rec); save_data(d)
            kb2 = types.InlineKeyboardMarkup()
            kb2.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_{sig_id}"),
                    types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{sig_id}"))
            send_with_brand(chat_id, caption, image_buf=img, reply_markup=kb2)
            return

        if data.startswith("ai_"):
            sig_id = data.split("_",1)[1]
            d = load_data()
            rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if not rec:
                bot.send_message(chat_id, "Signal not found.")
                return
            prompt = f"Analyze and give one-line verdict (BUY/SELL) and risk advice for:\n{json.dumps(rec['signal'], indent=2)}"
            analysis = ai_analysis_text(prompt)
            bot.send_message(chat_id, f"🤖 AI Analysis:\n{analysis}")
            return

        if data.startswith("link_"):
            sig_id = data.split("_",1)[1]
            bot.send_message(chat_id, f"Upload screenshot now and then reply with: #link {sig_id} TP1 or #link {sig_id} SL")
            return

        if data == "trending":
            try:
                text = fetch_trending_pairs()
                bot.send_message(chat_id, text)
            except Exception:
                bot.send_message(chat_id, "Failed to fetch trending.")
            return

        if data == "challenge_status":
            d = load_data(); c = d.get("challenge",{})
            bot.send_message(chat_id, f"🏆 Challenge Balance: ${c.get('balance',0):.2f}\nWins:{c.get('wins',0)} Losses:{c.get('losses',0)}")
            return

        if data == "history":
            d = load_data(); recs = d.get("signals",[])[-20:]
            text = "Recent signals:\n" + "\n".join([f"{r['id']} {r['signal']['symbol']} {r['signal']['signal']} res:{r.get('result') or '-'}" for r in recs[::-1]])
            bot.send_message(chat_id, text)
            return

        if data == "refresh_bot":
            bot.send_message(chat_id, "🔄 Refreshed.")
            return

        if data == "bot_status":
            bot.send_message(chat_id, f"Bot running. Time: {datetime.utcnow().isoformat()}")
            return

        bot.answer_callback_query(call.id, "Unknown action")
    except Exception:
        traceback.print_exc()
        try: bot.answer_callback_query(call.id, "Handler error")
        except: pass

# photo handler saves PnL screenshot
@bot.message_handler(content_types=['photo'])
def handle_photo(msg):
    try:
        file_info = bot.get_file(msg.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = record_pnl_screenshot(downloaded, now, msg.from_user.id, msg.caption)
        bot.reply_to(msg, "Screenshot saved. To link to signal: reply with: #link <signal_id> TP1 or SL")
    except Exception:
        traceback.print_exc(); bot.reply_to(msg, "Failed to save screenshot.")

@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_handler(message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 3:
            bot.reply_to(message, "Usage: #link <signal_id> TP1 or SL"); return
        sig_id, res = parts[1], parts[2].upper()
        d = load_data()
        # find last unlinked screenshot by this user
        pnl_item = next((p for p in reversed(d["pnl"]) if p.get("linked") is None and p["from"]==message.from_user.id), None)
        if not pnl_item:
            bot.reply_to(message, "No unlinked screenshot found.")
            return
        pnl_item["linked"] = {"signal_id":sig_id,"result":res,"linked_by":message.from_user.id}
        # if admin confirms, update challenge
        if message.from_user.id == ADMIN_ID:
            sigrec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if sigrec:
                risk = sigrec.get("risk_amt",0)
                if res.startswith("TP"):
                    d["challenge"]["balance"] = d["challenge"].get("balance",0)+risk
                    d["challenge"]["wins"] = d["challenge"].get("wins",0)+1
                    sigrec["result"]=res
                elif res == "SL":
                    d["challenge"]["balance"] = d["challenge"].get("balance",0)-risk
                    d["challenge"]["losses"]= d["challenge"].get("losses",0)+1
                    sigrec["result"]="SL"
        save_data(d)
        bot.reply_to(message, f"Linked screenshot to {sig_id} as {res}. Admin confirmation updates challenge.")
    except Exception:
        traceback.print_exc(); bot.reply_to(message, "Failed linking screenshot.")

# simple text handlers
@bot.message_handler(func=lambda m: True)
def fallback(message):
    txt = (message.text or "").strip().lower()
    if txt == "menu":
        bot.send_message(message.chat.id, "Menu:", reply_markup=main_keyboard())
        return
    bot.send_message(message.chat.id, "Tap a button to start:", reply_markup=main_keyboard())

# start polling and provide function to be called by main
def start_bot_polling():
    stop_existing_bot_instances()
    print("Starting Telegram polling...")
    # ensure polling runs in the main thread (blocking) when called directly
    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception:
        traceback.print_exc()

# small Flask binding if needed (optional)
def start_flask_app(port=8080):
    from flask import Flask, jsonify, request
    app = Flask("boss_destiny_app")
    @app.route("/", methods=["GET"])
    def health():
        return jsonify({"status":"ok","time": datetime.utcnow().isoformat()})
    @app.route("/stop", methods=["POST"])
    def stop():
        # not implemented; placeholder
        return jsonify({"status":"stopping"}), 200
    app.run(host="0.0.0.0", port=port)
    # market.py
import os, time, traceback
from datetime import datetime
import pandas as pd
import numpy as np

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
COINGECKO_MAP = {"BTCUSDT":"bitcoin","ETHUSDT":"ethereum","BNBUSDT":"binancecoin","SOLUSDT":"solana","XRPUSDT":"ripple","DOGEUSDT":"dogecoin"}

def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent":"BossDestiny/1.0"})
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
    errors=[]
    try:
        return fetch_klines_binance(symbol, interval, limit)
    except Exception as e:
        errors.append("Binance:" + str(e))
    try:
        return fetch_klines_coingecko(symbol, interval, limit)
    except Exception as e:
        errors.append("Coingecko:" + str(e))
    raise RuntimeError("All providers failed: " + " | ".join(errors))

# indicators
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up/(ma_down+1e-9)
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    mc = ef - es
    msig = mc.ewm(span=signal, adjust=False).mean()
    hist = mc - msig
    return mc, msig, hist

# main generator (simpler, robust)
def generate_signal_for(symbol="BTCUSDT", interval="1h"):
    try:
        df = fetch_klines_df(symbol, interval, limit=300)
        if df is None or len(df)<20:
            return {"error":"insufficient data"}
        df["ema9"] = ema(df["close"], 9); df["ema21"] = ema(df["close"], 21)
        df["rsi"] = rsi(df["close"], 14)
        mc,msig,mh = macd(df["close"]); df["macd_hist"]=mh
        last = df.iloc[-1]; prev = df.iloc[-2]

        signal=None; reasons=[]; score=0.0
        if prev["ema9"]<=prev["ema21"] and last["ema9"]>last["ema21"]:
            signal="LONG"; reasons.append("EMA cross up"); score+=0.3
        if prev["ema9"]>=prev["ema21"] and last["ema9"]<last["ema21"]:
            signal="SHORT"; reasons.append("EMA cross down"); score+=0.3
        if last["macd_hist"]>0: score+=0.1
        r = float(last["rsi"])
        if signal=="LONG" and r>80: reasons.append("High RSI"); score-=0.12
        if signal=="SHORT" and r<20: reasons.append("Low RSI"); score-=0.12
        price=float(last["close"])
        sl = float(df["low"].iloc[-3]) if signal=="LONG" else float(df["high"].iloc[-3]) if signal=="SHORT" else price*0.995
        tp1 = price + (price-sl)*1.5 if signal=="LONG" else price - (sl-price)*1.5 if signal=="SHORT" else price*1.005
        confidence = max(0.05, min(0.98, 0.5 + score))
        # risk calc (balance stored in data.json)
        from storage import load_data
        d = load_data(); balance = d.get("challenge",{}).get("balance", 10)
        risk_usd = round((balance * float(os.getenv("RISK_PERCENT","5"))) / 100.0, 8)
        diff = abs(price - sl) if abs(price-sl)>1e-12 else 1e-12
        units = round(risk_usd / diff, 8)
        return {"symbol":symbol.upper(), "interval":interval, "signal":signal or "HOLD", "entry":round(price,8), "sl":round(sl,8), "tp1":round(tp1,8), "confidence":round(confidence,2), "reasons":reasons, "suggested_risk_usd":risk_usd, "suggested_units":units}
    except Exception as e:
        traceback.print_exc()
        return {"error":str(e)}

def fetch_trending_pairs():
    try:
        sess = get_session()
        r = sess.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8); r.raise_for_status()
        tickers = r.json()
        pairs = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","DOGEUSDT"]
        out=[]
        for p in pairs:
            t = next((x for x in tickers if x.get("symbol")==p), None)
            if t:
                out.append(f"{p}: {float(t.get('priceChangePercent',0)):.2f}% vol:{int(float(t.get('quoteVolume',0))):,}")
        return "Trending Pairs:\n" + "\n".join(out)
    except Exception:
        return "Failed to fetch trending."
        # ai.py
import os, traceback
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

def ai_analysis_text(prompt):
    if not client:
        return "AI not configured."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a professional trading mentor. Provide short actionable advice and a BUY/SELL verdict."},
                      {"role":"user","content":prompt}],
            max_tokens=300, temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return f"AI error: {e}"
        # storage.py
import os, json
from datetime import datetime

DATA_FILE = os.getenv("DATA_FILE","data.json")
UPLOAD_DIR = os.getenv("UPLOAD_DIR","uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def ensure_storage():
    if not os.path.exists(DATA_FILE):
        base = {"signals":[],"pnl":[],"challenge":{"balance":float(os.getenv("CHALLENGE_START","10")),"wins":0,"losses":0,"history":[]},"stats":{"total_signals":0,"wins":0,"losses":0},"auto_scan":False}
        with open(DATA_FILE,"w") as f: json.dump(base,f,indent=2)

def load_data():
    ensure_storage()
    with open(DATA_FILE,"r") as f: return json.load(f)

def save_data(d):
    with open(DATA_FILE + ".tmp","w") as f: json.dump(d,f,indent=2)
    os.replace(DATA_FILE + ".tmp", DATA_FILE)

def record_pnl_screenshot(binary_bytes, now_str, user_id, caption):
    fname = f"{UPLOAD_DIR}/{now_str}_{user_id}.jpg"
    with open(fname,"wb") as f: f.write(binary_bytes)
    d = load_data()
    d["pnl"].append({"file":fname,"from":user_id,"time":now_str,"caption":caption,"linked":None})
    save_data(d)
    return fname
    # utils.py
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

LOGO_TEXT = "Boss Destiny Trading Empire"

def create_brand_image(lines):
    w,h = 900,320
    img = Image.new("RGB",(w,h),(18,18,20))
    draw = ImageDraw.Draw(img)
    try:
        ftb = ImageFont.truetype("DejaVuSans-Bold.ttf",20)
        fts = ImageFont.truetype("DejaVuSans.ttf",16)
    except:
        ftb = ImageFont.load_default(); fts = ImageFont.load_default()
    draw.text((16,12), LOGO_TEXT, fill=(255,215,0), font=ftb)
    y = 56
    for l in lines:
        draw.text((16,y), l, fill=(230,230,230), font=fts)
        y += 28
    bio = BytesIO(); img.save(bio,"PNG"); bio.seek(0)
    return bio

def build_signal_image(sig):
    lines = [
        f"{sig['symbol']}  |  {sig['interval']}  |  {sig['signal']}",
        f"Entry: {sig['entry']}  SL: {sig['sl']}  TP1: {sig['tp1']}",
        f"Confidence: {int(sig.get('confidence',0)*100)}%  Risk: ${sig.get('suggested_risk_usd',0)}",
        "Reasons: " + (", ".join(sig.get('reasons',[])) if sig.get('reasons') else "None")
    ]
    return create_brand_image(lines)

def safe_send(bot, chat_id, text, image_buf=None, reply_markup=None):
    try:
        if image_buf:
            bot.send_photo(chat_id, image_buf, caption=text, reply_markup=reply_markup)
        else:
            bot.send_message(chat_id, text, reply_markup=reply_markup)
    except Exception:
        bot.send_message(chat_id, text)
