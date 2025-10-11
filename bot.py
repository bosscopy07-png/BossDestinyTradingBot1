# ---------- bot.py (PART 1 of 4) ----------
# Boss Destiny Trading Empire — Final Ultimate Bot (Part 1: setup & utils)

import os
import time
import json
import math
import threading
import traceback
from datetime import datetime, timedelta
from io import BytesIO

import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import telebot
from telebot import types

# Plotting optional
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Flask keepalive
from flask import Flask
app = Flask(__name__)
@app.route("/")
def index():
    return "Boss Destiny Trading Empire Bot — Alive"

def run_keepalive(host="0.0.0.0", port=8080):
    try:
        app.run(host=host, port=int(port))
    except Exception as e:
        print("Keepalive error:", e)

# ------------------ Configuration ------------------
BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
CHANNEL_ID = os.getenv("CHANNEL_ID") or None

DEFAULT_INTERVAL = os.getenv("SIGNAL_INTERVAL", "1h")
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,SOLUSDT").split(",")
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "5"))
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "0"))
SIGNAL_COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "10"))
CHALLENGE_TARGET = float(os.getenv("CHALLENGE_TARGET", "100"))
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "20"))

DATA_FILE = os.getenv("DATA_FILE", "data.json")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOGO_TEXT = os.getenv("LOGO_TEXT", "Boss Destiny Trading Empire")

if not BOT_TOKEN:
    raise RuntimeError("Missing BOT_TOKEN environment variable.")
# Admin ID optional but advisable:
if ADMIN_ID == 0:
    print("Warning: ADMIN_ID is 0. Set ADMIN_ID to your Telegram user id to enable admin-only features.")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

def atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def init_storage():
    if not os.path.exists(DATA_FILE):
        obj = {
            "signals": [],
            "pnl": [],
            "challenge": {"balance": CHALLENGE_START, "wins":0, "losses":0, "history":[]},
            "stats": {"total_signals":0, "wins":0, "losses":0},
            "last_scan": {},
            "auto_scan": False,
            "users": []
        }
        atomic_write(DATA_FILE, obj)

def load_data():
    init_storage()
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(d):
    atomic_write(DATA_FILE, d)

def log_exc(e):
    s = f"[{datetime.utcnow().isoformat()}] ERROR: {e}\n" + traceback.format_exc()
    try:
        with open(os.path.join(LOG_DIR, "error_log.txt"), "a") as lf:
            lf.write(s + "\n")
    except:
        pass
    print(s)

def log_info(msg):
    s = f"[{datetime.utcnow().isoformat()}] INFO: {msg}"
    try:
        with open(os.path.join(LOG_DIR, "info_log.txt"), "a") as lf:
            lf.write(s + "\n")
    except:
        pass
    print(s)

def nice(x, nd=8):
    try:
        return float(round(x, nd))
    except:
        return x

def now_iso():
    return datetime.utcnow().isoformat()

def create_brand_image(text=LOGO_TEXT, size=(800,120), font_size=36):
    try:
        img = Image.new("RGBA", size, (255,255,255,255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
        w,h = draw.textsize(text, font=font)
        x = (size[0]-w)//2
        y = (size[1]-h)//2
        draw.text((x,y), text, fill=(0,0,0), font=font)
        buf = BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        return buf.read()
    except Exception as e:
        log_exc(e)
        return None

BRAND_IMAGE_BYTES = create_brand_image()

# End of Part 1
# ---------- bot.py (PART 2 of 4) ----------
# Market data fetchers + indicators + chart generation

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BYBIT_KLINES = "https://api.bybit.com/public/linear/kline"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
COINGECKO_MAP = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    "ADAUSDT": "cardano",
    "SOLUSDT": "solana"
}

_session = None
def get_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent":"BossDestinyBot/1.0"})
        retries = Retry(total=5, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))
        _session = s
    return _session

def normalize_interval(s):
    if not s: return DEFAULT_INTERVAL
    s2 = s.strip().lower()
    s2 = s2.replace("hours","h").replace("hour","h").replace("hrs","h").replace("mins","m").replace("min","m")
    if s2 in {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"}:
        return s2
    if s2.endswith(("m","h","d")):
        return s2
    return DEFAULT_INTERVAL

def fetch_klines_binance(symbol="BTCUSDT", interval="1h", limit=200):
    params = {"symbol": symbol.upper(), "interval": normalize_interval(interval), "limit": int(limit)}
    r = get_session().get(BINANCE_KLINES, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Binance HTTP {r.status_code}: {r.text[:200]}")
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def fetch_klines_bybit(symbol="BTCUSDT", interval="1h", limit=200):
    tf_map = {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30","1h":"60","2h":"120","4h":"240","1d":"D"}
    tf = tf_map.get(normalize_interval(interval), "60")
    params = {"symbol": symbol.upper(), "interval": tf, "limit": int(limit)}
    r = get_session().get(BYBIT_KLINES, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Bybit HTTP {r.status_code}: {r.text[:200]}")
    js = r.json()
    data = js.get("result") or js
    if isinstance(data, dict) and "list" in data:
        arr = data["list"]
    elif isinstance(data, list):
        arr = data
    else:
        arr = data.get("data", [])
    df = pd.DataFrame(arr)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="s")
    elif "start_at" in df.columns:
        df["open_time"] = pd.to_datetime(df["start_at"], unit="s")
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    df = df[["open_time","open","high","low","close","volume"]]
    return df

def fetch_klines_coingecko(symbol="BTCUSDT", interval="1h", limit=200):
    coin = COINGECKO_MAP.get(symbol.upper())
    if not coin:
        raise RuntimeError("Fallback not supported for " + symbol)
    days = 7 if normalize_interval(interval).endswith("h") else 30
    r = get_session().get(COINGECKO_MARKET_CHART.format(id=coin), params={"vs_currency":"usd","days":days}, timeout=10)
    r.raise_for_status()
    j = r.json()
    prices = j.get("prices", [])[-limit:]
    rows = []
    for p in prices:
        ts = int(p[0]); price = float(p[1])
        rows.append([pd.to_datetime(ts, unit="ms"), price, price, price, price, 0.0])
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume"])
    return df

def fetch_klines_df(symbol="BTCUSDT", interval="1h", limit=200):
    errs = []
    try:
        return fetch_klines_binance(symbol, interval, limit)
    except Exception as e:
        errs.append(str(e)); log_info(f"Binance failed {symbol}/{interval}: {e}")
    try:
        return fetch_klines_bybit(symbol, interval, limit)
    except Exception as e:
        errs.append(str(e)); log_info(f"Bybit failed {symbol}/{interval}: {e}")
    try:
        return fetch_klines_coingecko(symbol, interval, limit)
    except Exception as e:
        errs.append(str(e)); log_info(f"Coingecko failed {symbol}/{interval}: {e}")
    raise RuntimeError("All providers failed: " + " | ".join(errs))

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    mc = ef - es
    msig = mc.ewm(span=signal, adjust=False).mean()
    hist = mc - msig
    return mc, msig, hist

def compute_atr(df, period=14):
    high = df["high"]; low = df["low"]; close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

def generate_candlestick_image(df, symbol, candles=80):
    if not MATPLOTLIB_AVAILABLE:
        return None
    try:
        dfp = df.copy().tail(candles)
        dates = mdates.date2num(dfp["open_time"].dt.to_pydatetime())
        o = dfp["open"].values; h = dfp["high"].values
        l = dfp["low"].values; c = dfp["close"].values
        fig, ax = plt.subplots(figsize=(10,4), dpi=100)
        width = (dates[1]-dates[0]) * 0.6 if len(dates) > 1 else 0.0005
        for i in range(len(dates)):
            color = "green" if c[i] >= o[i] else "red"
            ax.plot([dates[i], dates[i]], [l[i], h[i]], color=color, linewidth=0.8)
            rect = plt.Rectangle((dates[i]-width/2, min(o[i], c[i])), width, abs(c[i]-o[i]), color=color)
            ax.add_patch(rect)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))
        ax.set_title(f"{symbol} — Last {len(dfp)} candles")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        buf = BytesIO(); plt.savefig(buf, format="png"); plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        log_exc(e)
        return None

# End of Part 2
# ---------- bot.py (PART 3 of 4) ----------
# Signal engine + leverage + AI analysis + record + scan

def suggest_leverage(confidence, price, atr, base_max_leverage=MAX_LEVERAGE):
    try:
        vol_ratio = (atr / price) if price > 0 else 0
        base = max(1.0, confidence * base_max_leverage)
        if vol_ratio > 0.03:
            base *= 0.3
        elif vol_ratio > 0.015:
            base *= 0.6
        lev = int(max(1, min(base_max_leverage, round(base))))
        return lev
    except Exception as e:
        log_exc(e)
        return 1

def generate_signal_for(symbol="BTCUSDT", base_interval="1h"):
    try:
        base_tf = normalize_interval(base_interval)
        high_tf = "4h"
        df = fetch_klines_df(symbol, interval=base_tf, limit=300)
        df_high = None
        try:
            df_high = fetch_klines_df(symbol, interval=high_tf, limit=200)
        except:
            df_high = None

        if df is None or len(df) < 30:
            return {"error":"insufficient data"}

        df["ema_fast"] = ema(df["close"], EMA_FAST)
        df["ema_slow"] = ema(df["close"], EMA_SLOW)
        df["rsi"] = rsi(df["close"], RSI_PERIOD)
        mc, msig, mh = macd(df["close"])
        df["macd_hist"] = mh
        atr_s = compute_atr(df, 14)
        atr = float(atr_s.iloc[-1]) if len(atr_s)>0 else 0.0

        last = df.iloc[-1]; prev = df.iloc[-2]
        signal = None; reasons=[]; score = 0.0

        # EMA cross
        if (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"]):
            signal = "BUY"; reasons.append("EMA cross up"); score += 0.28
        if (prev["ema_fast"] >= prev["ema_slow"]) and (last["ema_fast"] < last["ema_slow"]):
            signal = "SELL"; reasons.append("EMA cross down"); score += 0.28

        # MACD
        if last["macd_hist"] > 0:
            reasons.append("MACD positive"); score += 0.10
        else:
            score -= 0.03

        # wick rejection
        body = abs(last["close"] - last["open"])
        upper_wick = last["high"] - max(last["close"], last["open"])
        lower_wick = min(last["close"], last["open"]) - last["low"]
        if body>0:
            ur = upper_wick/body; lr = lower_wick/body
            if ur > 2 and upper_wick > lower_wick:
                reasons.append("Upper wick rejection"); score -= 0.12
                if not signal: signal = "SELL"
            if lr > 2 and lower_wick > upper_wick:
                reasons.append("Lower wick rejection"); score -= 0.12
                if not signal: signal = "BUY"

        r = float(df["rsi"].iloc[-1])
        if signal=="BUY" and r>78:
            reasons.append(f"High RSI {r:.1f}"); score -= 0.14
        if signal=="SELL" and r<22:
            reasons.append(f"Low RSI {r:.1f}"); score -= 0.14

        if df_high is not None and len(df_high)>=10:
            df_high["ema_fast"] = ema(df_high["close"], EMA_FAST)
            df_high["ema_slow"] = ema(df_high["close"], EMA_SLOW)
            last_h = df_high.iloc[-1]
            if signal=="BUY" and last_h["ema_fast"] < last_h["ema_slow"]:
                reasons.append("4h contradicts"); score -= 0.25
            if signal=="SELL" and last_h["ema_fast"] > last_h["ema_slow"]:
                reasons.append("4h contradicts"); score -= 0.25
            if ((last_h["ema_fast"] > last_h["ema_slow"]) and signal=="BUY") or ((last_h["ema_fast"] < last_h["ema_slow"]) and signal=="SELL"):
                score += 0.12

        vol = float(last.get("volume",0.0))
        if MIN_VOLUME and vol < MIN_VOLUME:
            reasons.append("Low volume"); score -= 0.08
        price = float(last["close"])
        if atr > price * 0.03:
            reasons.append("High ATR"); score -= 0.30

        confidence = max(0.05, min(0.98, 0.5 + score))

        if signal=="BUY":
            swing_sl = float(df["low"].iloc[-3])
            sl_by_atr = price - (atr * 1.5) if atr>0 else swing_sl
            sl = max(swing_sl, sl_by_atr)
            tp1 = price + (price - sl) * 1.5
            tp2 = price + (price - sl) * 3
        elif signal=="SELL":
            swing_sl = float(df["high"].iloc[-3])
            sl_by_atr = price + (atr * 1.5) if atr>0 else swing_sl
            sl = min(swing_sl, sl_by_atr)
            tp1 = price - (sl - price) * 1.5
            tp2 = price - (sl - price) * 3
        else:
            sl = price*0.995; tp1 = price*1.005; tp2 = price*1.01

        suggested_leverage = suggest_leverage(confidence, price, atr, base_max_leverage=MAX_LEVERAGE)
        d = load_data()
        balance = d["challenge"].get("balance", CHALLENGE_START)
        suggested_risk_usd = round((balance * RISK_PERCENT)/100.0, 8)
        diff = abs(price - sl) if abs(price - sl)>1e-12 else 1e-12
        suggested_units = round(suggested_risk_usd / diff, 8)

        return {
            "symbol": symbol.upper(),
            "interval": base_tf,
            "time": now_iso(),
            "signal": signal or "HOLD",
            "entry": round(price,8),
            "sl": round(sl,8),
            "tp1": round(tp1,8),
            "tp2": round(tp2,8),
            "atr": round(atr,8),
            "rsi": round(r,2),
            "volume": vol,
            "reasons": reasons,
            "confidence": round(confidence,2),
            "suggested_risk_usd": suggested_risk_usd,
            "suggested_units": suggested_units,
            "suggested_leverage": int(suggested_leverage)
        }
    except Exception as e:
        log_exc(e)
        return {"error": str(e)}

# record & send
last_signal_time = {}
def can_send_signal(symbol):
    last = last_signal_time.get(symbol.upper())
    if not last: return True
    return (datetime.utcnow() - last) > timedelta(minutes=SIGNAL_COOLDOWN_MIN)

def record_and_send(sig_obj, chat_id=None, user_id=None):
    try:
        d = load_data()
        sig_id = f"S{int(time.time())}"
        balance = d["challenge"].get("balance", CHALLENGE_START)
        risk_amt = round((balance * RISK_PERCENT)/100.0, 8)
        pos_units = sig_obj.get("suggested_units", 0)
        rec = {"id": sig_id, "signal": sig_obj, "time": now_iso(),
               "risk_amt": risk_amt, "pos_size": pos_units,
               "user": user_id or ADMIN_ID, "result": None}
        d["signals"].append(rec)
        d["stats"]["total_signals"] = d["stats"].get("total_signals",0) + 1
        save_data(d)
        last_signal_time[sig_obj["symbol"]] = datetime.utcnow()

        chart = None
        try:
            df = fetch_klines_df(sig_obj["symbol"], interval=sig_obj["interval"], limit=120)
            chart = generate_candlestick_image(df, sig_obj["symbol"]) if MATPLOTLIB_AVAILABLE else None
        except:
            chart = None

        stats = d.get("stats", {}); wins = stats.get("wins",0); total = stats.get("total_signals",0)
        accuracy = (wins/total*100) if total else 0.0

        text = (f"🔥 <b>Boss Destiny Signal</b> 🔥\nID: {sig_id}\nPair: {sig_obj['symbol']} | TF: {sig_obj['interval']}\n"
                f"Signal: <b>{sig_obj['signal']}</b>\nEntry: {sig_obj['entry']} SL: {sig_obj['sl']}\n"
                f"TP1: {sig_obj['tp1']} TP2: {sig_obj['tp2']}\n\n"
                f"💰 Risk: ${risk_amt:.4f} ({RISK_PERCENT}% of ${balance:.2f})\n"
                f"📈 Units: {pos_units} | Leverage: {sig_obj.get('suggested_leverage',1)}x\n"
                f"🎯 Confidence: {int(sig_obj['confidence']*100)}% | Accuracy: {accuracy:.1f}%\n"
                f"Reasons: {', '.join(sig_obj['reasons']) if sig_obj['reasons'] else 'None'}")

        kb = types.InlineKeyboardMarkup(row_width=2)
        kb.add(types.InlineKeyboardButton("📤 Post (Admin)", callback_data=f"post_{sig_id}"),
               types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_pnl_{sig_id}"))
        kb.add(types.InlineKeyboardButton("🤖 AI Analysis", callback_data=f"ai_sig_{sig_id}"),
               types.InlineKeyboardButton("🔄 Refresh", callback_data="refresh_bot"))

        target = chat_id or (CHANNEL_ID if CHANNEL_ID else (user_id or ADMIN_ID))
        if BRAND_IMAGE_BYTES:
            try:
                bot.send_photo(target, BytesIO(BRAND_IMAGE_BYTES), caption=text, reply_markup=kb)
                if chart:
                    bot.send_photo(target, BytesIO(chart))
            except Exception:
                bot.send_message(target, text, reply_markup=kb)
                if chart:
                    bot.send_photo(target, BytesIO(chart))
        else:
            bot.send_message(target, text, reply_markup=kb)
            if chart:
                bot.send_photo(target, BytesIO(chart))
        return sig_id
    except Exception as e:
        log_exc(e)
        return None

def scan_and_get_top4(pairs=None, interval=DEFAULT_INTERVAL):
    pairs = pairs or PAIRS
    picks = []
    for p in pairs:
        try:
            sig = generate_signal_for(p, base_interval=interval)
            if sig and not sig.get("error") and sig.get("signal") in ("BUY","SELL"):
                picks.append(sig)
        except Exception as e:
            log_info(f"scan {p} error {e}")
    picks_sorted = sorted(picks, key=lambda x: x.get("confidence",0), reverse=True)
    top4 = picks_sorted[:4]
    d = load_data(); d["last_scan"] = {"time": now_iso(), "picks": top4}; save_data(d)
    return top4

def get_trending_pairs(n=5):
    # A simple heuristic: sort by volatility (ATR) or volume across PAIRS
    trends = []
    for p in PAIRS:
        try:
            df = fetch_klines_df(p, interval="1h", limit=100)
            atr = compute_atr(df, 14).iloc[-1]
            vol = df["volume"].iloc[-1]
            trends.append({"pair": p, "atr": float(atr), "vol": float(vol)})
        except:
            continue
    trends_sorted = sorted(trends, key=lambda x: x["atr"]*x["vol"], reverse=True)
    return trends_sorted[:n]

# End of Part 3
# ---------- bot.py (PART 4 of 4) ----------
# Telegram handlers + AI & utility buttons + startup

# OpenAI new-client interface
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def ask_ai(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a professional crypto analyst."},
                      {"role":"user","content":prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        log_exc(e)
        return f"AI error: {e}"

def main_menu():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📈 Get Signal", callback_data="get_signal"),
        types.InlineKeyboardButton("🔎 Scan Top 4", callback_data="scan_top4"),
        types.InlineKeyboardButton("📊 Bot Status", callback_data="bot_status"),
        types.InlineKeyboardButton("📡 Trending Pairs", callback_data="trending_pairs"),
        types.InlineKeyboardButton("📰 Market News", callback_data="market_news"),
        types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status"),
        types.InlineKeyboardButton("📸 Upload PnL", callback_data="pnl_upload"),
        types.InlineKeyboardButton("🧾 History", callback_data="history"),
        types.InlineKeyboardButton("🤖 Ask AI", callback_data="ask_ai"),
        types.InlineKeyboardButton("🔄 Refresh Bot", callback_data="refresh_bot")
    )
    return kb

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        data = call.data; uid = call.from_user.id; cid = call.message.chat.id
        d = load_data()

        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"signal_pair_{p}"))
            kb.add(types.InlineKeyboardButton("Custom (type symbol)", callback_data="custom_symbol"))
            bot.send_message(cid, "Choose pair:", reply_markup=kb)
            return

        if data.startswith("signal_pair_"):
            pair = data.split("_",2)[2]
            sig = generate_signal_for(pair, base_interval=DEFAULT_INTERVAL)
            if sig.get("error"):
                bot.send_message(cid, f"Signal error: {sig['error']}")
                return
            record_and_send(sig, chat_id=cid, user_id=uid)
            bot.answer_callback_query(call.id, "Signal generated.")
            return

        if data == "scan_top4":
            top4 = scan_and_get_top4(PAIRS, DEFAULT_INTERVAL)
            if not top4:
                bot.send_message(cid, "No picks found.")
                return
            kb = types.InlineKeyboardMarkup()
            for t in top4:
                label = f"{t['symbol']} {t['signal']} ({int(t['confidence']*100)}%)"
                kb.add(types.InlineKeyboardButton(label, callback_data=f"picksend_{t['symbol']}"))
            kb.add(types.InlineKeyboardButton("Suggest allocation", callback_data="alloc_top4"))
            bot.send_message(cid, "Top 4 picks:", reply_markup=kb)
            return

        if data.startswith("picksend_"):
            sym = data.split("_",1)[1]
            scan = d.get("last_scan", {}).get("picks", [])
            pick = next((p for p in scan if p["symbol"].upper()==sym.upper()), None)
            if not pick:
                bot.answer_callback_query(call.id, "Pick not found.")
                return
            record_and_send(pick, chat_id=cid, user_id=uid)
            bot.answer_callback_query(call.id, "Pick sent.")
            return

        if data == "alloc_top4":
            scan = d.get("last_scan", {}).get("picks", [])
            if not scan:
                bot.answer_callback_query(call.id, "No scan data.")
                return
            sug = suggest_allocation_for_picks(scan)
            txt = f"Allocation (balance ${d['challenge']['balance']:.2f}):\n"
            for s in sug:
                txt += f"- {s['signal']['symbol']} {s['signal']['signal']} conf:{int(s['signal']['confidence']*100)}% alloc:${s['allocated_capital']:.2f}\n"
            send_brand_then_text(cid, txt)
            bot.answer_callback_query(call.id, "Allocation shown.")
            return

        if data == "bot_status":
            txt = ("📡 Bot Status\n"
                   f"- Uptime: (not tracked)\n"
                   f"- Pairs watched: {', '.join(PAIRS)}\n"
                   f"- Interval: {DEFAULT_INTERVAL}\n"
                   f"- Challenge balance: ${load_data()['challenge']['balance']:.2f}")
            send_brand_then_text(cid, txt)
            return

        if data == "trending_pairs":
            trends = get_trending_pairs(5)
            txt = "📈 Trending Pairs (volatility * ATR):\n"
            for t in trends:
                txt += f"- {t['pair']}: ATR={t['atr']:.4f}, Vol={t['vol']:.1f}\n"
            send_brand_then_text(cid, txt)
            return

        if data == "market_news":
            # Simple news via CoinGecko / crypto news API fallback
            try:
                j = safe_request("https://api.coingecko.com/api/v3/status_updates", retries=2)
                if j and "status_updates" in j:
                    updates = j["status_updates"][:5]
                    txt = "📰 Crypto News (recent):\n"
                    for u in updates:
                        txt += f"- {u.get('description')} ({u.get('category')})\n"
                    send_brand_then_text(cid, txt)
                else:
                    send_brand_then_text(cid, "No news available.")
            except Exception as e:
                send_brand_then_text(cid, f"News error: {e}")
            return

        if data == "challenge_status":
            c = d["challenge"]; w=c.get("wins",0); l=c.get("losses",0); b=c.get("balance", CHALLENGE_START)
            tot = w + l; acc = (w/tot*100) if tot else 0.0
            txt = (f"🏆 Challenge\nBalance: ${b:.2f}\nWins: {w}, Losses: {l}\nAccuracy: {acc:.1f}%\nTarget: ${CHALLENGE_TARGET}")
            send_brand_then_text(cid, txt, reply_markup=main_menu())
            return

        if data == "pnl_upload":
            bot.send_message(cid, "Upload your PnL screenshot now, then send: #link <signal_id> TP1 or SL")
            return

        if data.startswith("post_"):
            sig_id = data.split("_",1)[1]
            if uid != ADMIN_ID:
                bot.answer_callback_query(call.id, "Admin only.")
                return
            rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if not rec:
                bot.answer_callback_query(call.id, "Signal not found.")
                return
            if CHANNEL_ID:
                bot.send_message(CHANNEL_ID, f"📢 Official Signal:\n{json.dumps(rec['signal'], indent=2)}")
            bot.send_message(uid, f"Signal {sig_id} posted.")
            bot.answer_callback_query(call.id, "Posted.")
            return

        if data.startswith("link_pnl_"):
            sig_id = data.split("_",1)[1]
            bot.send_message(cid, f"Reply with screenshot, then send: #link {sig_id} TP1 or SL")
            return

        if data.startswith("ai_sig_"):
            sig_id = data.split("_",1)[1]
            rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if not rec:
                bot.answer_callback_query(call.id, "Signal not found.")
                return
            prompt = f"Signal analysis (buy/sell, rationale, leverage):\n{json.dumps(rec['signal'], indent=2)}"
            out = ask_ai(prompt)
            bot.send_message(cid, f"🤖 AI says:\n{out}")
            bot.answer_callback_query(call.id, "AI sent.")
            return

        if data == "history":
            recs = d.get("signals", [])[-40:]
            txt = "Recent signals:\n"
            for r in reversed(recs):
                s = r["signal"]
                txt += f"- {r['id']} {s['symbol']} {s['signal']} entry:{s['entry']} conf:{int(s.get('confidence',0)*100)}% result:{r.get('result') or '-'}\n"
            bot.send_message(cid, txt)
            return

        if data == "refresh_bot":
            bot.answer_callback_query(call.id, "Refreshing...")
            send_brand_then_text(cid, "Bot is alive and refreshed ✅", reply_markup=main_menu())
            return

        bot.answer_callback_query(call.id, "Unknown action.")
    except Exception as e:
        log_exc(e)
        try:
            bot.answer_callback_query(call.id, "Handler error.")
        except:
            pass

@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    try:
        fi = bot.get_file(message.photo[-1].file_id)
        dl = bot.download_file(fi.file_path)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        fname = os.path.join(UPLOAD_DIR, f"{now}_{message.photo[-1].file_id}.jpg")
        with open(fname, "wb") as f:
            f.write(dl)
        d = load_data()
        d["pnl"].append({"file": fname, "from": message.from_user.id, "time": now, "caption": message.caption, "linked": None})
        save_data(d)
        bot.reply_to(message, "Screenshot saved. To link: #link <signal_id> TP1 or SL")
    except Exception as e:
        log_exc(e); bot.reply_to(message, "Failed to save.")

@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_handler(message):
    try:
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
            bot.reply_to(message, "No screenshot found.")
            return
        pnl_item["linked"] = {"signal_id": sig_id, "result": tag, "linked_by": message.from_user.id}
        if message.from_user.id == ADMIN_ID:
            rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if rec:
                entry = float(rec["signal"].get("entry",0)); sl = float(rec["signal"].get("sl",entry))
                tp1 = float(rec["signal"].get("tp1",entry)); tp2 = float(rec["signal"].get("tp2",tp1))
                pos = float(rec.get("pos_size",0.0))
                side = rec["signal"].get("signal","BUY")
                if tag.startswith("TP"):
                    exit_price = tp1 if tag=="TP1" else tp2
                    pnl_units = (exit_price - entry)*pos if side != "SELL" else (entry - exit_price)*pos
                else:
                    exit_price = sl
                    pnl_units = (exit_price - entry)*pos if side!="SELL" else (entry - exit_price)*pos
                d["challenge"]["balance"] = d["challenge"].get("balance",CHALLENGE_START) + float(pnl_units)
                if tag.startswith("TP"):
                    d["challenge"]["wins"] = d["challenge"].get("wins",0) + 1
                    d["stats"]["wins"] = d["stats"].get("wins",0) + 1
                else:
                    d["challenge"]["losses"] = d["challenge"].get("losses",0) + 1
                    d["stats"]["losses"] = d["stats"].get("losses",0) + 1
                rec["result"] = tag
                d["challenge"]["history"].append({"time": now_iso(), "note": f"{sig_id} {tag}", "change": float(pnl_units)})
            save_data(d)
            try:
                bot.send_photo(message.chat.id, open(pnl_item["file"], "rb"), caption=f"Linked {sig_id} as {tag}. Balance: ${d['challenge']['balance']:.2f}")
            except:
                pass
        else:
            save_data(d)
        bot.reply_to(message, f"Linked {sig_id} as {tag}. Admin must confirm for balance update.")
    except Exception as e:
        log_exc(e); bot.reply_to(message, "Link error.")

@bot.message_handler(func=lambda m: True)
def all_messages(message):
    try:
        text = (message.text or "").strip()
        d = load_data()
        if message.from_user.id not in d.get("users", []):
            d["users"].append(message.from_user.id); save_data(d)

        if text.lower() == "menu":
            send_brand_then_text(message.chat.id, "Menu:", reply_markup=main_menu())
            return

        if text.startswith("AI:"):
            prompt = text[3:].strip()
            out = ask_ai(prompt)
            send_brand_then_text(message.chat.id, f"🤖 AI:\n{out}")
            return

        if text.lower().startswith("price "):
            parts = text.split()
            if len(parts)>=2:
                sym = parts[1].upper()
                try:
                    df = fetch_klines_df(sym, interval=DEFAULT_INTERVAL, limit=50)
                    last = df.iloc[-1]; price = float(last["close"])
                    bot.reply_to(message, f"{sym}: ${price}")
                    return
                except Exception as e:
                    bot.reply_to(message, f"Price error: {e}")
                    return
            bot.reply_to(message, "Usage: price <SYMBOL>")
            return

        if text.upper() in [p.upper() for p in PAIRS]:
            sym = text.upper()
            try:
                sig = generate_signal_for(sym, base_interval=DEFAULT_INTERVAL)
                if sig.get("error"):
                    bot.reply_to(message, f"Error: {sig['error']}")
                    return
                send_brand_then_text(message.chat.id, f"{sym} analysis:\nSignal: {sig['signal']}\nEntry: {sig['entry']} SL: {sig['sl']} Conf: {int(sig['confidence']*100)}% Lev: {sig.get('suggested_leverage',1)}x")
            except Exception as e:
                bot.reply_to(message, f"Fetch error: {e}")
            return

        send_brand_then_text(message.chat.id, "Tap a button:", reply_markup=main_menu())
    except Exception as e:
        log_exc(e)
        try:
            bot.reply_to(message, "Handler error.")
        except:
            pass

def send_brand_then_text(chat_id, text, reply_markup=None):
    try:
        if BRAND_IMAGE_BYTES:
            bot.send_photo(chat_id, BytesIO(BRAND_IMAGE_BYTES), caption=text, reply_markup=reply_markup)
            return
    except:
        pass
    bot.send_message(chat_id, text, reply_markup=reply_markup)

# scanner thread
def scanner_loop():
    log_info(f"Scanner started for {PAIRS} at {DEFAULT_INTERVAL}")
    while True:
        try:
            d = load_data()
            if d.get("auto_scan", False):
                top4 = scan_and_get_top4(PAIRS, DEFAULT_INTERVAL)
                if top4:
                    bal = d["challenge"].get("balance", CHALLENGE_START)
                    sug = suggest_allocation_for_picks(top4)
                    msg = f"🔔 Auto-scan picks (balance ${bal:.2f}):\n"
                    for s in sug:
                        sig = s["signal"]
                        msg += f"- {sig['symbol']} {sig['signal']} conf:{int(sig['confidence']*100)}% entry:{sig['entry']}\n"
                    send_brand_then_text(ADMIN_ID, msg)
            time.sleep(60)
        except Exception as e:
            log_exc(e)
            time.sleep(10)

def start_services():
    port = int(os.getenv("PORT", "8080"))
    tk = threading.Thread(target=run_keepalive, args=("0.0.0.0",port), daemon=True)
    tk.start(); log_info(f"Keepalive on port {port}")
    ts = threading.Thread(target=scanner_loop, daemon=True)
    ts.start(); log_info("Scanner started")
    log_info("Starting Telegram polling...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

if __name__ == "__main__":
    log_info("Boss Destiny Trading Empire Bot starting...")
    start_services()
