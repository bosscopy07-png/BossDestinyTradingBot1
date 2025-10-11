# ---------- bot.py (PART 1 of 4) ----------
# Boss Destiny Trading Bot — Part 1 (setup, keep-alive server, storage, fetcher, indicators)

import os
import json
import time
import threading
import requests
import traceback
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import telebot
from telebot import types

# Optional plotting (candles). If missing, charts will be skipped but bot still works.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ---------- Keep-alive web server (Flask) ----------
# This ensures platforms like Render detect an open port.
from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "Boss Destiny Trading Bot — alive"

def run_keepalive(host="0.0.0.0", port=8080):
    try:
        app.run(host=host, port=int(port))
    except Exception as e:
        print("Keep-alive server failed:", e)

# ---------- CONFIG (env vars) ----------
# IMPORTANT: set these env vars on Render / your host
BOT_TOKEN = os.getenv("BOT_TOKEN")               # required
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))       # required, numeric telegram id
CHANNEL_ID = os.getenv("CHANNEL_ID")             # optional channel to post official signals
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") # optional for AI analysis

# Trading / signal defaults
# You requested default timeframe = 1h
DEFAULT_INTERVAL = os.getenv("SIGNAL_INTERVAL", "1h")
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,SOLUSDT").split(",")
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "5"))   # percent of balance risked per trade
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "0"))
SIGNAL_COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))

# Challenge settings
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "10"))   # $10 start
CHALLENGE_TARGET = float(os.getenv("CHALLENGE_TARGET", "100"))# $100 target

LOGO_PATH = os.getenv("LOGO_PATH", "bd_logo.png")
DATA_FILE = os.getenv("DATA_FILE", "data.json")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Binance endpoints
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_24H = "https://api.binance.com/api/v3/ticker/24hr"

# Interval normalization to avoid bad inputs like "1hrs"
VALID_INTERVALS = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","2h":"2h","4h":"4h","6h":"6h","12h":"12h","1d":"1d"}
def normalize_interval(s):
    if not s:
        return DEFAULT_INTERVAL
    s2 = s.strip().lower()
    if s2.endswith("hrs"): s2 = s2.replace("hrs","h")
    if s2.endswith("hours"): s2 = s2.replace("hours","h")
    if s2.endswith("hour"): s2 = s2.replace("hour","h")
    if s2.endswith("mins"): s2 = s2.replace("mins","m")
    if s2.endswith("min"): s2 = s2.replace("min","m")
    if s2 in VALID_INTERVALS: return VALID_INTERVALS[s2]
    if s2.endswith(("m","h","d")): return s2
    return DEFAULT_INTERVAL

# ---------- Sanity checks ----------
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var is required.")
if ADMIN_ID == 0:
    raise RuntimeError("ADMIN_ID env var is required (your numeric Telegram id).")

# ---------- initialize bot ----------
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

# ---------- storage helpers ----------
def atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def init_storage():
    if not os.path.exists(DATA_FILE):
        d = {
            "signals": [],       # stored signals
            "pnl": [],           # uploaded screenshots metadata
            "challenge": {"balance": CHALLENGE_START, "wins":0, "losses":0, "history":[]},
            "stats": {"total_signals":0, "wins":0, "losses":0},
            "last_scan": {},
            "auto_scan": False,
            "users": []
        }
        atomic_write(DATA_FILE, d)

def load_data():
    init_storage()
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(d):
    atomic_write(DATA_FILE, d)

init_storage()

# ---------- small utilities ----------
def nice(x, nd=8):
    try:
        return float(round(x, nd))
    except:
        return x

def log_exc(e):
    print("ERROR:", e)
    traceback.print_exc()

def send_logo_with_optional_chart(chat_id, text, reply_markup=None, chart_bytes=None, reply_to=None):
    """
    Sends logo (if available) with caption text. If chart_bytes provided, send logo with caption first, then the chart image.
    """
    try:
        if chart_bytes:
            if os.path.exists(LOGO_PATH):
                with open(LOGO_PATH, "rb") as logo_f:
                    bot.send_photo(chat_id, logo_f, caption=text, reply_markup=reply_markup, reply_to_message_id=reply_to)
            bio = BytesIO(chart_bytes)
            bio.seek(0)
            bot.send_photo(chat_id, bio)
            return
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as logo_f:
                bot.send_photo(chat_id, logo_f, caption=text, reply_markup=reply_markup, reply_to_message_id=reply_to)
                return
    except Exception as e:
        log_exc(e)
    bot.send_message(chat_id, text, reply_markup=reply_markup, reply_to_message_id=reply_to)

# ---------- Binance klines fetcher with retries ---------
# ---------- REPLACEMENT: Robust Binance fetcher + improved signal logic ----------
import math
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# create a session with retries (one global session for bot)
_session = None
def get_requests_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "BossDestinyBot/1.0 (+https://t.me/yourbot)"
        })
        # configure urllib3 retries for connection-level errors
        retries = Retry(total=5, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=["GET","POST"])
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _session = s
    return _session

def fetch_klines_df(symbol="BTCUSDT", interval="1h", limit=200, max_attempts=4):
    """
    More robust fetch:
      - normalizes interval
      - uses session + user-agent
      - exponential backoff
      - returns DataFrame or raises RuntimeError with HTTP status (so logs show exact cause)
    """
    interval = normalize_interval(interval)
    symbol = symbol.upper()
    limit = int(limit) if limit else 200
    url = BINANCE_KLINES
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    session = get_requests_session()

    attempt = 0
    while attempt < max_attempts:
        try:
            resp = session.get(url, params=params, timeout=10)
            # if the status is 200, parse and return
            if resp.status_code == 200:
                data = resp.json()
                cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
                df = pd.DataFrame(data, columns=cols)
                for c in ["open","high","low","close","volume"]:
                    df[c] = df[c].astype(float)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                return df
            # For 4xx/5xx produce a clear error
            # 429 = rate limit, 418/451 = blocked/tos, 403 = forbidden
            status = resp.status_code
            text = resp.text[:400]
            # handle 429 specifically by backing off longer
            if status == 429:
                wait = 2 ** attempt + 1
                print(f"Binance 429 rate limit for {symbol}/{interval}. Backoff {wait}s. Resp: {text[:200]}")
                time.sleep(wait)
                attempt += 1
                continue
            if status in (418, 451, 403):
                # likely blocked or legal restriction => do NOT retry heavily
                raise RuntimeError(f"Binance HTTP {status} ({resp.reason}) for {symbol} {interval} — response snippet: {text}")
            # for other 5xx, retry with backoff
            if 500 <= status < 600:
                wait = 2 ** attempt
                print(f"Binance server error {status} for {symbol}/{interval}. Backoff {wait}s.")
                time.sleep(wait)
                attempt += 1
                continue
            # for other client errors raise immediately
            raise RuntimeError(f"Binance HTTP {status} for {symbol} {interval} — {resp.reason} - {text}")
        except requests.RequestException as e:
            # network error -> backoff and retry
            wait = 2 ** attempt + 1
            print(f"Network error fetching klines {symbol}/{interval}: {e}. Backoff {wait}s.")
            time.sleep(wait)
            attempt += 1
    raise RuntimeError(f"Failed to fetch klines for {symbol} {interval} after {max_attempts} attempts")

# ---------- ATR helper ----------
def compute_atr(df, period=14):
    """
    Average True Range (ATR) for volatility-based SL/TP.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

# ---------- improved signal (multi-timeframe confirmation + ATR) ----------
def generate_signal_improved(symbol="BTCUSDT", base_interval="1h"):
    """
    Multi-timeframe: check base_interval (1h) and a higher timeframe (4h).
    Rules:
      - Primary signal from base timeframe EMA crossover + wick checks + MACD/Rsi
      - Confirm that the higher timeframe isn't strongly against the base signal (optional)
      - Compute ATR and use it to set SL/TP (sl = recent swing or ATR*1.5)
      - Filter out signals if ATR is very large relative to price (too volatile) or volume very low
      - Return: signal, entry, sl, tp1, tp2, confidence, reasons, suggested_risk_usd, suggested_pos_units
    """
    try:
        base_tf = normalize_interval(base_interval)
        higher_tf = "4h"  # confirmation timeframe
        # fetch both timeframes (reduce limit to 150/100 to be lighter)
        df_base = fetch_klines_df(symbol, interval=base_tf, limit=200)
        df_high = fetch_klines_df(symbol, interval=higher_tf, limit=150)

        # compute indicators
        df_base["ema_fast"] = ema(df_base["close"], EMA_FAST)
        df_base["ema_slow"] = ema(df_base["close"], EMA_SLOW)
        df_base["rsi"] = rsi(df_base["close"], RSI_PERIOD)
        mc, msig, mhist = macd(df_base["close"])
        df_base["macd_hist"] = mhist
        atr_series = compute_atr(df_base, period=14)
        atr = float(atr_series.iloc[-1]) if len(atr_series)>0 else 0.0

        last = df_base.iloc[-1]; prev = df_base.iloc[-2]

        signal = None; reasons=[]; score=0.0

        # EMA cross detection
        if (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"]):
            signal = "BUY"; reasons.append("EMA cross up"); score += 0.28
        if (prev["ema_fast"] >= prev["ema_slow"]) and (last["ema_fast"] < last["ema_slow"]):
            signal = "SELL"; reasons.append("EMA cross down"); score += 0.28

        # MACD
        if last["macd_hist"] > 0:
            reasons.append("MACD hist positive"); score += 0.10
        else:
            score -= 0.03

        # wick rejection (body and wicks)
        body = abs(last["close"] - last["open"])
        upper_wick = last["high"] - max(last["close"], last["open"])
        lower_wick = min(last["close"], last["open"]) - last["low"]
        if body > 0:
            ur = upper_wick / body
            lr = lower_wick / body
            if ur > 2 and upper_wick > lower_wick:
                reasons.append("Upper wick rejection"); score -= 0.12
                if not signal: signal = "SELL"
            if lr > 2 and lower_wick > upper_wick:
                reasons.append("Lower wick rejection"); score -= 0.12
                if not signal: signal = "BUY"

        # RSI filter
        r = float(df_base["rsi"].iloc[-1])
        if signal=="BUY" and r>78:
            reasons.append(f"High RSI {r:.1f}"); score -= 0.14
        if signal=="SELL" and r<22:
            reasons.append(f"Low RSI {r:.1f}"); score -= 0.14

        # higher timeframe confirmation: if higher_tf trend strongly opposite, reduce score heavily
        try:
            df_high["ema_fast"] = ema(df_high["close"], EMA_FAST)
            df_high["ema_slow"] = ema(df_high["close"], EMA_SLOW)
            last_h = df_high.iloc[-1]; prev_h = df_high.iloc[-2]
            if (last_h["ema_fast"] > last_h["ema_slow"]) and signal=="SELL":
                reasons.append("4h trend contradicts (bullish)"); score -= 0.25
            if (last_h["ema_fast"] < last_h["ema_slow"]) and signal=="BUY":
                reasons.append("4h trend contradicts (bearish)"); score -= 0.25
            # if both timeframes align, boost confidence
            if ((last_h["ema_fast"] > last_h["ema_slow"]) and signal=="BUY") or ((last_h["ema_fast"] < last_h["ema_slow"]) and signal=="SELL"):
                score += 0.12
        except Exception:
            # ignore high-tf confirmation errors
            pass

        # volume check
        vol = float(df_base["volume"].iloc[-1])
        if MIN_VOLUME and vol < MIN_VOLUME:
            reasons.append("Low volume"); score -= 0.08

        # volatility / ATR sanity: if ATR > price*0.02 (2%) then market is too volatile to trade small account
        price = float(last["close"])
        if atr > price * 0.03:
            reasons.append(f"High ATR {atr:.6f} relative to price -> avoid"); score -= 0.30

        confidence = max(0.05, min(0.98, 0.5 + score))
        # compute SL using ATR or recent swing
        if signal == "BUY":
            swing_sl = float(df_base["low"].iloc[-3])
            sl_by_atr = price - (atr * 1.5) if atr>0 else swing_sl
            sl = max(swing_sl, sl_by_atr)
            tp1 = price + (price - sl) * 1.5
            tp2 = price + (price - sl) * 3
        elif signal == "SELL":
            swing_sl = float(df_base["high"].iloc[-3])
            sl_by_atr = price + (atr * 1.5) if atr>0 else swing_sl
            sl = min(swing_sl, sl_by_atr)
            tp1 = price - (sl - price) * 1.5
            tp2 = price - (sl - price) * 3
        else:
            sl = price * 0.995
            tp1 = price * 1.005
            tp2 = price * 1.01

        # suggested risk (USD) and suggested position units (approx) based on current balance
        d = load_data()
        balance = d.get("challenge", {}).get("balance", CHALLENGE_START)
        suggested_risk_usd = round((balance * RISK_PERCENT) / 100.0, 8)
        # approximate unit size in quote currency units: risk_usd / |entry - sl|
        diff = abs(price - sl) if abs(price - sl) > 1e-12 else 1e-12
        suggested_units = round(suggested_risk_usd / diff, 8)

        return {
            "symbol": symbol.upper(),
            "interval": base_tf,
            "signal": signal or "HOLD",
            "entry": round(price, 8),
            "sl": round(sl, 8),
            "tp1": round(tp1, 8),
            "tp2": round(tp2, 8),
            "atr": round(atr, 8),
            "rsi": round(r, 2),
            "volume": vol,
            "reasons": reasons,
            "confidence": round(confidence, 2),
            "suggested_risk_usd": suggested_risk_usd,
            "suggested_units": suggested_units
        }
    except Exception as e:
        log_exc(e)
        return {"error": str(e)}

# ---------- indicators ----------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff()
    up = d.clip(lower=0)
    down = -1 * d.clip(upper=0)
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

# ---------- optional candlestick image generator (uses matplotlib if available) ----------
def generate_candlestick_image(df, symbol):
    """
    Returns PNG bytes for a candlestick image of last ~60 candles, or None if plotting unavailable.
    ""

# ---------- risk sizing ----------
def compute_risk_and_size(entry, sl, balance, risk_percent):
    """
    Calculate risk amount (USD) and position size in quote units approximated by price difference.
    (This is a simplified calculation for manual execution).
    """
Returns DataFrame of klines from Binance API
Retries up to 3 times on network errors
"""
def fetch_klines_df(...):
    ...
    try:
        risk_amount = (balance * risk_percent) / 100.0
        diff = abs(entry - sl)
        if diff <= 1e-12:
            pos_size = 0.0
        else:
            pos_size = risk_amount / diff
        return round(risk_amount, 8), round(float(pos_size), 8)
    except Exception as e:
        log_exc(e)
        return 0.0, 0.0

# ---------- record & send signal ----------
last_signal_time = {}
def can_send_signal(symbol):
    last = last_signal_time.get(symbol)
    if not last:
        return True
    return (datetime.utcnow() - last) > timedelta(minutes=SIGNAL_COOLDOWN_MIN)

def record_and_send(sig, chat_id=None, user_id=None):
    d = load_data()
    sig_id = f"S{int(time.time())}"
    balance = d["challenge"].get("balance", CHALLENGE_START)
    risk_amt, pos_size = compute_risk_and_size(sig["entry"], sig["sl"], balance, RISK_PERCENT)
    rec = {"id": sig_id, "signal": sig, "time": datetime.utcnow().isoformat(), "risk_amt": risk_amt, "pos_size": pos_size, "user": user_id or ADMIN_ID, "result": None}
    d["signals"].append(rec)
    d["stats"]["total_signals"] = d["stats"].get("total_signals", 0) + 1
    save_data(d)
    last_signal_time[sig["symbol"]] = datetime.utcnow()

    # compute accuracy percentage
    stats = d.get("stats", {})
    wins = stats.get("wins", 0); total = stats.get("total_signals", 0)
    accuracy = (wins / total * 100) if total else 0.0

    # produce optional chart
    chart = None
    try:
        df = fetch_klines_df(sig["symbol"], interval=DEFAULT_INTERVAL, limit=120)
        chart = generate_candlestick_image(df, sig["symbol"]) if MATPLOTLIB_AVAILABLE else None
    except Exception:
        chart = None

    text = (f"🔥 <b>Boss Destiny Signal</b> 🔥\nID: {sig_id}\nPair: {sig['symbol']} | TF: {sig['interval']}\n"
            f"Signal: <b>{sig['signal']}</b>\nEntry: {sig['entry']}\nSL: {sig['sl']}\nTP1: {sig['tp1']} | TP2: {sig['tp2']}\n\n"
            f"💰 Risk per trade: ${risk_amt:.4f} ({RISK_PERCENT}% of ${balance:.2f})\n"
            f"📈 Pos size: {pos_size}\n"
            f"🎯 Confidence: {int(sig['confidence']*100)}% | Accuracy: {accuracy:.1f}%\n"
            f"Reasons: {', '.join(sig['reasons']) if sig['reasons'] else 'None'}")

    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("📤 Post (Admin)", callback_data=f"post_{sig_id}"))
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_pnl_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Analysis", callback_data=f"ai_sig_{sig_id}"))
    kb.add(types.InlineKeyboardButton("📊 Add to Portfolio", callback_data=f"add_port_{sig_id}"))

    target = chat_id or (CHANNEL_ID if CHANNEL_ID else ADMIN_ID)
    send_logo_with_optional_chart(target, text, reply_markup=kb, chart_bytes=chart)
    return sig_id

# ---------- scan & top4 ----------
def scan_and_get_top4(pairs=None, interval=DEFAULT_INTERVAL):
    pairs = pairs or PAIRS
    picks = []
    for p in pairs:
        try:
            df = fetch_klines_df(p, interval=interval, limit=250)
            sig = generate_signal_from_df(df, p, interval)
            if not sig or "error" in sig:
                continue
            picks.append(sig)
        except Exception as e:
            print("scan error", p, e)
            continue
    picks_sorted = sorted(picks, key=lambda x: x.get("confidence", 0), reverse=True)
    top4 = picks_sorted[:4]
    d = load_data(); d["last_scan"] = {"time": datetime.utcnow().isoformat(), "picks": top4}; save_data(d)
    return top4

def suggest_allocation_for_picks(top4, balance):
    total_conf = sum([t.get("confidence", 0) for t in top4]) or 1.0
    suggestions = []
    for t in top4:
        conf = t.get("confidence", 0)
        allocated_cap = (conf / total_conf) * balance
        risk_amt = (balance * RISK_PERCENT) / 100.0
        entry = t["entry"]; sl = t["sl"]
        diff = abs(entry - sl) if abs(entry - sl) > 1e-12 else 1e-12
        pos_size = risk_amt / diff
        suggestions.append({"signal": t, "allocated_capital": round(allocated_cap, 4), "risk_amt": round(risk_amt, 6), "pos_size": round(pos_size, 6)})
    return suggestions

# ---------- scanner loop ----------
def scanner_loop():
    print("Scanner thread running")
    while True:
        data = load_data()
        if data.get("auto_scan", False):
            try:
                top = scan_and_get_top4(PAIRS, DEFAULT_INTERVAL)
                if top:
                    balance = data["challenge"].get("balance", CHALLENGE_START)
                    suggestions = suggest_allocation_for_picks(top, balance)
                    txt = f"🤖 Auto-scan Top {len(suggestions)} picks (balance ${balance:.2f}):\n"
                    for s in suggestions:
                        sig = s["signal"]
                        txt += f"- {sig['symbol']} {sig['signal']} conf:{int(sig['confidence']*100)}% entry:{sig['entry']} SL:{sig['sl']} alloc:${s['allocated_capital']:.2f}\n"
                    send_logo_with_optional_chart(ADMIN_ID, txt)
                time.sleep(60)
            except Exception as e:
                print("scanner error:", e)
                time.sleep(10)
        else:
            time.sleep(5)

# ---------- AI wrapper ----------
def ai_text_analysis(prompt):
    if not OPENAI_API_KEY:
        return "AI disabled (OPENAI_API_KEY not set)."
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a professional crypto market analyst."},
                      {"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log_exc(e)
        return f"AI error: {e}"

# End of PART 2 of 4
# Paste Part 3 next (Telegram handlers, callbacks, upload/link).
# ---------- bot.py (PART 3 of 4) ----------
# Telegram handlers, callback handling, upload/link flows, menu

# ---------- callback handler ----------
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        data = call.data; user_id = call.from_user.id; chat_id = call.message.chat.id
        d = load_data()

        # Get signal menu
        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS:
                kb.add(types.InlineKeyboardButton(p, callback_data=f"signal_pair_{p}"))
            kb.add(types.InlineKeyboardButton("Scan Top 4", callback_data="scan_top4"))
            bot.send_message(chat_id, "Choose pair or Scan Top 4:", reply_markup=kb)
            return

        # Single pair signal
        if data.startswith("signal_pair_"):
            pair = data.split("_",2)[2]
            try:
                df = fetch_klines_df(pair, interval=DEFAULT_INTERVAL, limit=300)
                sig = generate_signal_from_df(df, pair, DEFAULT_INTERVAL)
                if sig.get("error"):
                    bot.send_message(chat_id, f"Error generating signal: {sig['error']}")
                    return
                record_and_send(sig, chat_id=chat_id, user_id=user_id)
                bot.answer_callback_query(call.id, "Signal generated.")
            except Exception as e:
                bot.send_message(chat_id, f"Error fetching market data: {e}")
            return

        # Scan Top 4
        if data == "scan_top4":
            try:
                top4 = scan_and_get_top4(PAIRS, DEFAULT_INTERVAL)
                if not top4:
                    bot.send_message(chat_id, "No picks found.")
                    return
                kb = types.InlineKeyboardMarkup()
                for t in top4:
                    label = f"{t['symbol']} {t['signal']} ({int(t['confidence']*100)}%)"
                    kb.add(types.InlineKeyboardButton(label, callback_data=f"picksend_{t['symbol']}"))
                kb.add(types.InlineKeyboardButton("Suggest allocation (Top4)", callback_data="alloc_top4"))
                bot.send_message(chat_id, "Top 4 picks:", reply_markup=kb)
            except Exception as e:
                bot.send_message(chat_id, f"Scan error: {e}")
            return

        if data.startswith("picksend_"):
            sym = data.split("_",1)[1]
            last_scan = d.get("last_scan", {}).get("picks", [])
            pick = next((p for p in last_scan if p["symbol"].upper() == sym.upper()), None)
            if not pick:
                bot.answer_callback_query(call.id, "Pick not found.")
                return
            record_and_send(pick, chat_id=chat_id, user_id=user_id)
            bot.answer_callback_query(call.id, "Pick sent.")
            return

        if data == "alloc_top4":
            last_scan = d.get("last_scan", {}).get("picks", [])
            if not last_scan:
                bot.answer_callback_query(call.id, "No last scan data.")
                return
            bal = d["challenge"].get("balance", CHALLENGE_START)
            sug = suggest_allocation_for_picks(last_scan, bal)
            txt = f"Allocation suggestions (balance ${bal:.2f}):\n"
            for s in sug:
                txt += f"- {s['signal']['symbol']} {s['signal']['signal']} conf:{int(s['signal']['confidence']*100)}% alloc:${s['allocated_capital']:.2f} risk:${s['risk_amt']:.4f}\n"
            send_logo_with_optional_chart(chat_id, txt)
            bot.answer_callback_query(call.id, "Allocation shown.")
            return

        # Admin post
        if data.startswith("post_"):
            sig_id = data.split("_",1)[1]
            if user_id != ADMIN_ID:
                bot.answer_callback_query(call.id, "Admin only.")
                return
            rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if not rec:
                bot.answer_callback_query(call.id, "Signal not found.")
                return
            if CHANNEL_ID:
                try:
                    bot.send_message(CHANNEL_ID, f"📢 Official Signal:\n\n{rec['signal']}")
                except Exception as e:
                    bot.send_message(user_id, f"Failed to post: {e}")
            bot.send_message(user_id, f"Signal {sig_id} posted.")
            bot.answer_callback_query(call.id, "Posted.")
            return

        # Link PnL flow
        if data.startswith("link_pnl_"):
            sig_id = data.split("_",1)[1]
            bot.send_message(chat_id, f"Reply to this message with your screenshot, then send: #link {sig_id} TP1 or #link {sig_id} SL")
            return

        # AI analysis for signal
        if data.startswith("ai_sig_"):
            sig_id = data.split("_",1)[1]
            rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if not rec:
                bot.answer_callback_query(call.id, "Signal not found.")
                return
            prompt = f"Analyze this trading signal and market context:\n{json.dumps(rec['signal'], indent=2)}\nProvide rationale, risk controls, and two alternative exits."
            out = ai_text_analysis(prompt)
            bot.send_message(chat_id, f"🤖 AI Analysis:\n\n{out}")
            bot.answer_callback_query(call.id, "AI sent.")
            return

        if data == "challenge_status":
            c = d["challenge"]
            wins = c.get("wins", 0); losses = c.get("losses", 0); bal = c.get("balance", CHALLENGE_START)
            total = wins + losses; acc = (wins/total*100) if total else 0.0
            txt = f"🏆 Boss Destiny Challenge\nBalance: ${bal:.2f}\nWins: {wins} Losses: {losses}\nAccuracy: {acc:.1f}%\nTarget: ${CHALLENGE_TARGET}"
            send_logo_with_optional_chart(chat_id, txt, reply_markup=main_menu())
            return

        if data == "send_chart_info":
            bot.send_message(chat_id, "Send the chart image (photo) to this chat to save it.")
            return

        if data == "pnl_upload":
            bot.send_message(chat_id, "Upload your PnL screenshot now; then link it with: #link <signal_id> TP1 or SL")
            return

        if data == "ask_ai":
            bot.send_message(chat_id, "Type your market question like: AI: Is BTC bullish on 1h?")
            return

        if data == "history":
            recent = d.get("signals", [])[-12:]
            if not recent:
                bot.send_message(chat_id, "No history yet.")
                return
            txt = "Recent signals:\n"
            for r in reversed(recent):
                s = r["signal"]
                txt += f"- {r['id']} {s['symbol']} {s['signal']} entry:{s['entry']} conf:{int(s.get('confidence',0)*100)}% result:{r.get('result') or '-'}\n"
            bot.send_message(chat_id, txt)
            return

        if data == "export_csv":
            if user_id != ADMIN_ID:
                bot.answer_callback_query(call.id, "Admin only.")
                return
            rows = []
            for r in d.get("signals", []):
                s = r["signal"]
                rows.append({"id": r["id"], "symbol": s["symbol"], "signal": s["signal"], "entry": s["entry"], "sl": s["sl"], "tp1": s["tp1"], "conf": s.get("confidence",0), "time": r.get("time"), "result": r.get("result")})
            if not rows:
                bot.send_message(chat_id, "No records.")
                return
            df = pd.DataFrame(rows)
            csv_path = "signals_export.csv"
            df.to_csv(csv_path, index=False)
            with open(csv_path, "rb") as fh:
                bot.send_document(chat_id, fh)
            return

        if data == "toggle_auto_scan":
            if user_id != ADMIN_ID:
                bot.answer_callback_query(call.id, "Admin only.")
                return
            d["auto_scan"] = not d.get("auto_scan", False)
            save_data(d)
            bot.answer_callback_query(call.id, f"Auto-scan set to {d['auto_scan']}")
            bot.send_message(user_id, f"Auto-scan is now {d['auto_scan']}")
            return

        bot.answer_callback_query(call.id, "Unknown action.")
    except Exception as e:
        log_exc(e)
        try:
            bot.answer_callback_query(call.id, "Handler error.")
        except:
            pass

# ---------- photo upload handler ----------
@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        fname = os.path.join(UPLOAD_DIR, f"{now}_{message.photo[-1].file_id}.jpg")
        with open(fname, "wb") as f:
            f.write(downloaded)
        d = load_data()
        d["pnl"].append({"file": fname, "from": message.from_user.id, "time": now, "caption": message.caption, "linked": None})
        save_data(d)
        bot.reply_to(message, "Saved screenshot. To link: send #link <signal_id> TP1 or SL")
    except Exception as e:
        log_exc(e)
        bot.reply_to(message, "Failed to save screenshot.")

# ---------- #link handler ----------
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
            bot.reply_to(message, "No unlinked screenshot found. Upload first.")
            return
        pnl_item["linked"] = {"signal_id": sig_id, "result": tag, "linked_by": message.from_user.id}
        # admin confirms and updates challenge
        if message.from_user.id == ADMIN_ID:
            srec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if srec:
                entry = float(srec["signal"].get("entry", 0))
                sl = float(srec["signal"].get("sl", entry))
                tp1 = float(srec["signal"].get("tp1", entry))
                tp2 = float(srec["signal"].get("tp2", tp1))
                pos_size = float(srec.get("pos_size", 0.0))
                side = srec["signal"].get("signal", "BUY")
                if tag.startswith("TP"):
                    exit_price = tp1 if tag == "TP1" else tp2
                    pnl_units = (exit_price - entry) * pos_size if side != "SELL" else (entry - exit_price) * pos_size
                    d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) + pnl_units
                    d["challenge"]["wins"] = d["challenge"].get("wins", 0) + 1
                    d["stats"]["wins"] = d["stats"].get("wins", 0) + 1
                    d["challenge"]["history"].append({"time": datetime.utcnow().isoformat(), "note": f"{sig_id} {tag}", "change": float(pnl_units)})
                elif tag == "SL":
                    exit_price = sl
                    pnl_units = (exit_price - entry) * pos_size if side != "SELL" else (entry - exit_price) * pos_size
                    d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) + pnl_units
                    d["challenge"]["losses"] = d["challenge"].get("losses", 0) + 1
                    d["stats"]["losses"] = d["stats"].get("losses", 0) + 1
                    d["challenge"]["history"].append({"time": datetime.utcnow().isoformat(), "note": f"{sig_id} SL", "change": float(pnl_units)})
                srec["result"] = tag
            save_data(d)
            try:
                bot.send_photo(message.chat.id, open(pnl_item["file"], "rb"), caption=f"Linked {sig_id} as {tag}. Balance: ${d['challenge']['balance']:.2f}")
            except Exception:
                pass
        else:
            save_data(d)
        bot.reply_to(message, f"Linked screenshot to {sig_id} as {tag}. Admin confirmation required to update challenge.")
    except Exception as e:
        log_exc(e)
        bot.reply_to(message, "Error linking screenshot.")

# ---------- menu builder ----------
def main_menu():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📈 Get Signal", callback_data="get_signal"),
        types.InlineKeyboardButton("🔎 Scan Top 4", callback_data="scan_top4"),
        types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status"),
        types.InlineKeyboardButton("📸 Upload PnL", callback_data="pnl_upload"),
        types.InlineKeyboardButton("🧾 History", callback_data="history"),
        types.InlineKeyboardButton("🤖 Ask AI", callback_data="ask_ai"),
        types.InlineKeyboardButton("📤 Export CSV", callback_data="export_csv"),
    )
    d = load_data()
    if ADMIN_ID:
        auto = d.get("auto_scan", False)
        kb.add(types.InlineKeyboardButton(f"Auto-Scan: {'ON' if auto else 'OFF'}", callback_data="toggle_auto_scan"))
    return kb

# End of PART 3 of 4
# Paste Part 4 next (message handlers, AI quick commands, startup & keep-alive).
# ---------- bot.py (PART 4 of 4) ----------
# Message handlers, quick commands, startup (Flask keep-alive + scanner)

# ---------- message handler ----------
@bot.message_handler(func=lambda m: True)
def all_messages(message):
    try:
        text = (message.text or "").strip()
        d = load_data()
        if message.from_user.id not in d.get("users", []):
            d["users"].append(message.from_user.id); save_data(d)

        # menu trigger
        if text.lower() == "menu":
            send_logo_with_optional_chart(message.chat.id, "Boss Destiny Menu", reply_markup=main_menu())
            return

        # AI prompt (prefix)
        if text.startswith("AI:"):
            prompt = text[3:].strip()
            out = ai_text_analysis(prompt)
            send_logo_with_optional_chart(message.chat.id, f"🤖 AI:\n\n{out}")
            return

        # price quick check: "price BTCUSDT"
        if text.lower().startswith("price "):
            parts = text.split()
            if len(parts) >= 2:
                sym = parts[1].upper()
                t = fetch_ticker_24h(sym)
                if not t:
                    bot.reply_to(message, f"No ticker for {sym}")
                    return
                price = float(t.get("lastPrice", 0)); change = float(t.get("priceChangePercent", 0)); vol = float(t.get("volume", 0))
                send_logo_with_optional_chart(message.chat.id, f"{sym} price: ${price:.6f}\n24h change: {change:.2f}%\nVolume: {vol}")
                return
            bot.reply_to(message, "Usage: price BTCUSDT")
            return

        # quick pair check if user types a symbol
        if text.upper() in [p.upper() for p in PAIRS]:
            sym = text.upper()
            try:
                df = fetch_klines_df(sym, interval=DEFAULT_INTERVAL, limit=300)
                sig = generate_signal_from_df(df, sym, DEFAULT_INTERVAL)
                if sig.get("error"):
                    bot.reply_to(message, f"Error: {sig['error']}")
                    return
                send_logo_with_optional_chart(message.chat.id, f"Quick analysis for {sym}:\nSignal: {sig['signal']}\nEntry: {sig['entry']}\nSL: {sig['sl']}\nTP1: {sig['tp1']}\nConf: {int(sig['confidence']*100)}%")
                return
            except Exception as e:
                bot.reply_to(message, f"Error fetching: {e}")
                return

        # fallback: show menu
        send_logo_with_optional_chart(message.chat.id, "Tap a button to start:", reply_markup=main_menu())
    except Exception as e:
        log_exc(e)
        try:
            bot.reply_to(message, "Handler error.")
        except:
            pass

# ---------- startup ----------
def start_services():
    # start Flask keep-alive server in a separate thread
    flask_port = int(os.getenv("PORT", "8080"))
    t_flask = threading.Thread(target=run_keepalive, args=("0.0.0.0", flask_port), daemon=True)
    t_flask.start()
    print("Keep-alive server started on port", flask_port)

    # start scanner thread
    t_scan = threading.Thread(target=scanner_loop, daemon=True)
    t_scan.start()
    print("Scanner thread started.")

    # start Telegram polling (blocking)
    print("Starting Telegram polling...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

if __name__ == "__main__":
    print("Boss Destiny Trading Bot starting...")
    start_services()

# End of PART 4 of 4 — Full bot.py assembled
