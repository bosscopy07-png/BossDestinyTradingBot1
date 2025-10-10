# bot.py - Boss Destiny Trading Assistant (Webhook-ready)
import os
import json
import time
import math
import threading
import requests
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
import numpy as np
from PIL import Image

import telebot
from telebot import types
from flask import Flask, request

# ---------------- CONFIG (env) ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN required in env vars.")

ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
CHANNEL_ID = os.getenv("CHANNEL_ID")  # optional channel id or @name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
SIGNAL_INTERVAL = os.getenv("SIGNAL_INTERVAL", "5m")
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "0"))
SIGNAL_COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "10"))
CHALLENGE_TARGET = float(os.getenv("CHALLENGE_TARGET", "100"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "5"))
LOGO_PATH = os.getenv("LOGO_PATH", "bd_logo.png")
USE_WEBHOOK = os.getenv("USE_WEBHOOK", "1") in ("1", "true", "True", "yes")
APP_URL = os.getenv("APP_URL") or os.getenv("RENDER_EXTERNAL_URL")  # webhook target url
PORT = int(os.getenv("PORT", "10000"))

DATA_FILE = "data.json"
UPLOAD_DIR = "uploads"

# ---------- sanity ----------
if ADMIN_ID == 0:
    raise RuntimeError("ADMIN_ID (your Telegram numeric id) required.")

# ---------- init bot & flask ----------
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
app = Flask(__name__)

# ---------- storage ----------
def init_storage():
    if not os.path.exists(DATA_FILE):
        d = {
            "signals": [],
            "pnl": [],
            "challenge": {"balance": CHALLENGE_START, "wins": 0, "losses": 0, "history": []},
            "stats": {"total_signals": 0, "wins": 0, "losses": 0},
            "users": []
        }
        atomic_write(DATA_FILE, d)

def atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def load_data():
    init_storage()
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(d):
    atomic_write(DATA_FILE, d)

init_storage()

# ---------- helpers ----------
def send_logo_with_text(chat_id, text, reply_markup=None, reply_to=None):
    try:
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as img:
                bot.send_photo(chat_id, img, caption=text, reply_markup=reply_markup, reply_to_message_id=reply_to)
                return
    except Exception:
        pass
    bot.send_message(chat_id, text, reply_markup=reply_markup, reply_to_message_id=reply_to)

def nice_num(x, ndigits=8):
    try:
        return float(round(x, ndigits))
    except Exception:
        return x

# ---------- Binance klines ----------
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
def fetch_klines_df(symbol="BTCUSDT", interval="5m", limit=300):
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

# ---------- indicators ----------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

# ---------- signal engine ----------
def generate_signal_for(symbol, interval):
    try:
        df = fetch_klines_df(symbol=symbol, interval=interval, limit=300)
    except Exception as e:
        return {"error": f"klines error: {e}"}

    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal = None
    reasons = []

    crossed_buy = (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"])
    crossed_sell = (prev["ema_fast"] >= prev["ema_slow"]) and (last["ema_fast"] < last["ema_slow"])

    if crossed_buy:
        signal = "BUY"
        reasons.append(f"EMA{EMA_FAST}>EMA{EMA_SLOW} crossover")
    if crossed_sell:
        signal = "SELL"
        reasons.append(f"EMA{EMA_FAST}<EMA{EMA_SLOW} crossover")

    body = abs(last["close"] - last["open"])
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]
    if body > 0:
        upper_ratio = upper_wick / body
        lower_ratio = lower_wick / body
    else:
        upper_ratio = lower_ratio = 0

    if upper_ratio > 2 and upper_wick > lower_wick:
        reasons.append("Upper-wick rejection")
        if not signal:
            signal = "SELL"
    if lower_ratio > 2 and lower_wick > upper_wick:
        reasons.append("Lower-wick rejection")
        if not signal:
            signal = "BUY"

    rsi_val = last["rsi"]
    if signal == "BUY" and rsi_val > 80:
        reasons.append(f"RSI {rsi_val:.1f} high - caution")
    if signal == "SELL" and rsi_val < 20:
        reasons.append(f"RSI {rsi_val:.1f} low - caution")

    vol = last["volume"]
    if MIN_VOLUME and vol < MIN_VOLUME:
        reasons.append(f"low vol {vol}")

    price = float(last["close"])
    if signal == "BUY":
        sl = float(df["low"].iloc[-3])
        tp1 = price + (price - sl) * 1.5
        tp2 = price + (price - sl) * 3
    elif signal == "SELL":
        sl = float(df["high"].iloc[-3])
        tp1 = price - (sl - price) * 1.5
        tp2 = price - (sl - price) * 3
    else:
        sl = price * 0.995
        tp1 = price * 1.005
        tp2 = price * 1.01

    confidence = 0.45
    if any("EMA" in r for r in reasons): confidence += 0.25
    if ("Lower-wick" in " ".join(reasons) and signal=="BUY") or ("Upper-wick" in " ".join(reasons) and signal=="SELL"):
        confidence += 0.15
    if not signal:
        confidence = 0.15

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "timestamp": datetime.utcnow().isoformat(),
        "signal": signal or "HOLD",
        "entry": nice_num(price,8),
        "sl": nice_num(sl,8),
        "tp1": nice_num(tp1,8),
        "tp2": nice_num(tp2,8),
        "rsi": round(float(rsi_val),2),
        "volume": vol,
        "reasons": reasons,
        "confidence": round(min(confidence,0.95),2)
    }

# ---------- risk sizing ----------
def compute_risk_and_size(entry, sl, balance, risk_percent):
    risk_amount = (balance * risk_percent) / 100.0
    diff = abs(entry - sl)
    if diff <= 0:
        pos_size = 0.0
    else:
        pos_size = risk_amount / diff
    return round(risk_amount,8), round(float(pos_size),8)

# ---------- record & broadcast ----------
last_signal_time = {}
def can_send_signal(symbol):
    last = last_signal_time.get(symbol)
    if not last:
        return True
    return (datetime.utcnow() - last) > timedelta(minutes=SIGNAL_COOLDOWN_MIN)

def record_and_send_signal(sig, chat_id=None, user_id=None):
    d = load_data()
    sig_id = f"S{int(time.time())}"
    balance = d["challenge"].get("balance", CHALLENGE_START)
    risk_amt, pos_size = compute_risk_and_size(sig["entry"], sig["sl"], balance, RISK_PERCENT)
    rec = {
        "id": sig_id, "signal": sig, "time": datetime.utcnow().isoformat(),
        "risk_amt": risk_amt, "pos_size": pos_size, "user": user_id or ADMIN_ID, "posted": False
    }
    d["signals"].append(rec)
    d["stats"]["total_signals"] = d["stats"].get("total_signals",0) + 1
    save_data(d)
    last_signal_time[sig["symbol"]] = datetime.utcnow()

    stats = d.get("stats",{})
    wins = stats.get("wins",0); total = stats.get("total_signals",0)
    accuracy = (wins/total*100) if total else 0.0

    text = (f"🔥 <b>Boss Destiny Signal</b> 🔥\n"
            f"ID: {sig_id}\nPair: {sig['symbol']} | TF: {sig['interval']}\n"
            f"Signal: <b>{sig['signal']}</b>\nEntry: {sig['entry']}\nSL: {sig['sl']}\nTP1: {sig['tp1']} | TP2: {sig['tp2']}\n\n"
            f"💰 Capital (risk): ${risk_amt:.4f} ({RISK_PERCENT}% of ${balance:.2f})\n"
            f"📈 Position size (units): {pos_size}\n"
            f"🎯 Confidence: {int(sig['confidence']*100)}% | Accuracy: {accuracy:.1f}%\n"
            f"Reasons: {', '.join(sig['reasons']) if sig['reasons'] else 'None'}\n\nExecute manually. Admin can post to channel or link PnL.")
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("📤 Post (Admin)", callback_data=f"post_{sig_id}"))
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_pnl_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 Ask AI", callback_data=f"ai_sig_{sig_id}"))
    target = chat_id or (CHANNEL_ID if CHANNEL_ID else ADMIN_ID)
    send_logo_with_text(target, text, reply_markup=kb)
    return sig_id

# ---------- scanner ----------
def scanner_loop():
    print("Scanner started:", PAIRS, SIGNAL_INTERVAL)
    while True:
        try:
            for pair in PAIRS:
                try:
                    sig = generate_signal_for(pair, SIGNAL_INTERVAL)
                    if sig.get("error"):
                        print("kline error:", sig["error"])
                        continue
                    if sig["signal"] in ("BUY","SELL") and can_send_signal(pair):
                        if (sig["signal"]=="BUY" and sig["rsi"] < 85) or (sig["signal"]=="SELL" and sig["rsi"] > 15):
                            record_and_send_signal(sig)
                            print("signal sent", pair, sig["signal"])
                except Exception as e:
                    print("pair error", pair, e)
            # sleep aligned with interval
            secs = 60
            if SIGNAL_INTERVAL.endswith("m"):
                try:
                    mins = int(SIGNAL_INTERVAL[:-1]); secs = max(60, mins*60)
                except: secs = 60
            elif SIGNAL_INTERVAL.endswith("h"):
                try:
                    hrs = int(SIGNAL_INTERVAL[:-1]); secs = hrs*3600
                except: secs = 300
            time.sleep(secs)
        except Exception as e:
            print("scanner outer", e)
            time.sleep(10)

# ---------- OpenAI opinion ----------
def ai_text_analysis(prompt):
    if not OPENAI_API_KEY:
        return "AI not configured."
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a professional crypto market analyst."},{"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {e}"

# ---------- Telegram handlers ----------
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    data = call.data; user_id = call.from_user.id; chat_id = call.message.chat.id
    d = load_data()

    if data.startswith("post_"):
        sig_id = data.split("_",1)[1]
        if user_id != ADMIN_ID:
            bot.answer_callback_query(call.id, "Only admin can post official signals.")
            return
        rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
        if not rec:
            bot.answer_callback_query(call.id, "Signal not found.")
            return
        if CHANNEL_ID:
            try:
                bot.send_message(CHANNEL_ID, f"📢 Official Signal:\n\n{rec['signal']['signal']} {rec['signal']['entry']} SL:{rec['signal']['sl']} TP1:{rec['signal']['tp1']}")
            except Exception as e:
                bot.send_message(user_id, f"Failed to post to channel: {e}")
        bot.send_message(user_id, f"Signal {sig_id} posted.")
        bot.answer_callback_query(call.id, "Posted.")
        return

    if data.startswith("link_pnl_"):
        sig_id = data.split("_",1)[1]
        bot.send_message(chat_id, f"Reply with screenshot image then send: #link {sig_id} TP1 or #link {sig_id} SL")
        return

    if data.startswith("ai_sig_"):
        sig_id = data.split("_",1)[1]
        rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
        if not rec:
            bot.answer_callback_query(call.id, "Signal not found.")
            return
        prompt = f"Explain and critique this signal; give risk controls and two alternate exits:\n{json.dumps(rec['signal'], indent=2)}"
        analysis = ai_text_analysis(prompt)
        bot.send_message(chat_id, f"🤖 AI Analysis for {sig_id}:\n\n{analysis}")
        bot.answer_callback_query(call.id, "AI sent.")
        return

    if data == "get_signal":
        kb = types.InlineKeyboardMarkup()
        for p in PAIRS:
            kb.add(types.InlineKeyboardButton(p, callback_data=f"signal_pair_{p}"))
        bot.send_message(chat_id, "Choose pair:", reply_markup=kb)
        return

    if data.startswith("signal_pair_"):
        pair = data.split("_",2)[2]
        kb = types.InlineKeyboardMarkup()
        for t in ["1m","5m","15m","1h","4h"]:
            kb.add(types.InlineKeyboardButton(t, callback_data=f"signal_run_{pair}_{t}"))
        bot.send_message(chat_id, f"Selected {pair}. Choose timeframe:", reply_markup=kb)
        return

    if data.startswith("signal_run_"):
        _, pair, tf = data.split("_",2)
        sig = generate_signal_for(pair, tf)
        if sig.get("error"):
            bot.send_message(chat_id, f"Error: {sig['error']}")
            return
        sig_id = record_and_send_signal(sig, chat_id=chat_id, user_id=user_id)
        bot.answer_callback_query(call.id, "Signal generated.")
        return

    if data == "challenge_status":
        c = d["challenge"]
        wins = c.get("wins",0); losses = c.get("losses",0); bal = c.get("balance", CHALLENGE_START)
        total = wins + losses
        acc = (wins/total*100) if total else 0.0
        txt = (f"🏆 Boss Destiny Challenge\nBalance: ${bal:.2f}\nWins: {wins} | Losses: {losses}\nAccuracy: {acc:.1f}%\nTarget: ${CHALLENGE_TARGET}")
        send_logo_with_text(chat_id, txt, reply_markup=main_menu())
        return

    if data == "send_chart_info":
        bot.send_message(chat_id, "Send chart image in this chat; the bot will save it for analysis.")
        return

    if data == "pnl_upload":
        bot.send_message(chat_id, "Upload PnL screenshot now; then link with: #link <signal_id> TP1 or SL")
        return

    if data == "ask_ai":
        bot.send_message(chat_id, "Type your market question starting with: AI: <your text>")
        return

# photo handler
@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded = bot.download_file(file_info.file_path)
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    fname = os.path.join(UPLOAD_DIR, f"{now}_{message.photo[-1].file_id}.jpg")
    with open(fname,"wb") as f:
        f.write(downloaded)
    d = load_data()
    d["pnl"].append({"file": fname, "from": message.from_user.id, "time": now, "caption": message.caption})
    save_data(d)
    bot.reply_to(message, "Saved screenshot. To link: send: #link <signal_id> TP1 or SL")

# link command
@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_handler(message):
    parts = message.text.strip().split()
    if len(parts) < 3:
        bot.reply_to(message, "Usage: #link <signal_id> TP1 or SL")
        return
    sig_id = parts[1]; tag = parts[2].upper()
    d = load_data()
    pnl_item = None
    for p in reversed(d["pnl"]):
        if p.get("linked") is None and p["from"] == message.from_user.id:
            pnl_item = p; break
    if not pnl_item:
        bot.reply_to(message, "No unlinked screenshot found.")
        return
    pnl_item["linked"] = {"signal_id": sig_id, "result": tag, "linked_by": message.from_user.id}
    # apply to challenge only if admin
    if message.from_user.id == ADMIN_ID:
        srec = next((s for s in d["signals"] if s["id"]==sig_id), None)
        if srec:
            risk_amt = srec.get("risk_amt",0)
            if tag.startswith("TP"):
                profit = risk_amt * 1.0
                d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) + profit
                d["challenge"]["wins"] = d["challenge"].get("wins",0) + 1
                d["stats"]["wins"] = d["stats"].get("wins",0) + 1
                d["challenge"]["history"].append({"time": datetime.utcnow().isoformat(), "note": f"{sig_id} {tag}", "change": +profit})
            elif tag == "SL":
                loss = risk_amt
                d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) - loss
                d["challenge"]["losses"] = d["challenge"].get("losses",0) + 1
                d["stats"]["losses"] = d["stats"].get("losses",0) + 1
                d["challenge"]["history"].append({"time": datetime.utcnow().isoformat(), "note": f"{sig_id} {tag}", "change": -loss})
            save_data(d)
    else:
        save_data(d)
    bot.reply_to(message, f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance.")

# menu builder
def main_menu():
    kb = types.InlineKeyboardMarkup()
    kb.row(
        types.InlineKeyboardButton("📈 Get Signal", callback_data="get_signal"),
        types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status")
    )
    kb.row(
        types.InlineKeyboardButton("📸 Send Chart (image)", callback_data="send_chart_info"),
        types.InlineKeyboardButton("📁 PnL Upload", callback_data="pnl_upload")
    )
    kb.row(
        types.InlineKeyboardButton("💬 Ask AI", callback_data="ask_ai"),
        types.InlineKeyboardButton("🧾 History", callback_data="history")
    )
    return kb

@bot.message_handler(func=lambda m: True)
def all_messages(message):
    text = (message.text or "").strip()
    # register user
    d = load_data()
    if message.from_user.id not in d.get("users",[]):
        d["users"].append(message.from_user.id); save_data(d)
    if text.lower() == "menu":
        send_logo_with_text(message.chat.id, "Boss Destiny Menu", reply_markup=main_menu())
        return
    if text.startswith("AI:"):
        prompt = text[3:].strip()
        ans = ai_text_analysis(prompt)
        send_logo_with_text(message.chat.id, f"🤖 AI Answer:\n\n{ans}")
        return
    # fallback
    send_logo_with_text(message.chat.id, "Tap a button to start:", reply_markup=main_menu())

# ---------- Flask webhook endpoints ----------
@app.route("/" + BOT_TOKEN, methods=["POST"])
def telegram_webhook():
    json_str = request.get_data().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "Boss Destiny Trading Assistant", 200

# ---------- port binder for Render (keeps open) ----------
def keep_port_open():
    # Flask will bind the port; this function just prints info (Flask run handles port)
    print(f"Port binding active on {PORT}")

# ---------- start scanner thread ----------
scanner_thread = threading.Thread(target=scanner_loop, dae
