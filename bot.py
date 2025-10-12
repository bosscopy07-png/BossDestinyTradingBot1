# part1: imports, config, storage
import os
import time
import json
import math
import requests
import traceback
import threading
from datetime import datetime, timedelta
from io import BytesIO

# data & math
import numpy as np
import pandas as pd

# images
from PIL import Image, ImageDraw, ImageFont

# Telegram
import telebot
from telebot import types

# OpenAI new client
from openai import OpenAI

# ---------- CONFIG ----------
# Replace these two values with your keys (do not include quotes beyond the string)
BOT_TOKEN = os.getenv("BOT_TOKEN", "8367038069:AAFXwx9Bm7GVhDdSy8u0y9SJs9gR_K-WsMs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# Admin ID (set your Telegram numeric id in env or here)
ADMIN_ID = int(os.getenv("ADMIN_ID", "6442735461"))

# Trading settings
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,DOGEUSDT").split(",")
DEFAULT_INTERVAL = os.getenv("SIGNAL_INTERVAL", "1h")
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "5"))  # percent of challenge balance to risk per trade
SIGNAL_COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "10"))
CHALLENGE_TARGET = float(os.getenv("CHALLENGE_TARGET", "100"))
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "20"))

DATA_FILE = os.getenv("DATA_FILE", "data.json")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
LOGO_TEXT = os.getenv("LOGO_TEXT", "Boss Destiny Trading Empire")

# Webhook/polling flags (kept but default to polling)
RUN_WEBHOOK = os.getenv("RUN_WEBHOOK", "false").lower() in ("1","true","yes")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# ---------- VALIDATION ----------
if BOT_TOKEN.startswith("REPLACE_WITH"):
    raise RuntimeError("Set BOT_TOKEN before running bot.py")
# OPENAI optional: if missing AI features will be disabled but bot still runs
if OPENAI_API_KEY.startswith("REPLACE_WITH"):
    print("WARNING: OPENAI_API_KEY not set. AI features will be disabled (bot will still run).")

# ---------- CLIENTS ----------
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("REPLACE_WITH") else None

# ---------- STORAGE INIT ----------
os.makedirs(UPLOAD_DIR, exist_ok=True)
def init_storage():
    if not os.path.exists(DATA_FILE):
        base = {
            "signals": [],
            "pnl": [],
            "challenge": {"balance": CHALLENGE_START, "wins":0, "losses":0, "history":[]},
            "stats": {"total_signals":0, "wins":0, "losses":0},
            "last_scan": {},
        }
        with open(DATA_FILE, "w") as f:
            json.dump(base, f, indent=2)

def load_data():
    init_storage()
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(d):
    with open(DATA_FILE + ".tmp", "w") as f:
        json.dump(d, f, indent=2)
    os.replace(DATA_FILE + ".tmp", DATA_FILE)

init_storage()

# ---------- HTTP SESSION (retries) ----------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
_session = None
def get_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent":"BossDestiny/1.0"})
        retries = Retry(total=4, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=frozenset(["GET","POST"]))
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))
        _session = s
    return _session
    # part2: market fetchers, indicators, helpers
# ---------- endpoints ----------
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BYBIT_KLINES = "https://api.bybit.com/public/linear/kline"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
COINGECKO_MAP = {"BTCUSDT":"bitcoin","ETHUSDT":"ethereum","BNBUSDT":"binancecoin","SOLUSDT":"solana","ADAUSDT":"cardano","XRPUSDT":"ripple","DOGEUSDT":"dogecoin"}

def normalize_interval(s):
    if not s: return DEFAULT_INTERVAL
    s2 = s.strip().lower()
    s2 = s2.replace("hours","h").replace("hour","h").replace("hrs","h").replace("hr","h")
    valid = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"}
    return s2 if s2 in valid else DEFAULT_INTERVAL

# ---------- kline fetchers ----------
def fetch_klines_binance(symbol="BTCUSDT", interval="1h", limit=200):
    params = {"symbol": symbol.upper(), "interval": normalize_interval(interval), "limit": int(limit)}
    r = get_session().get(BINANCE_KLINES, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def fetch_klines_bybit(symbol="BTCUSDT", interval="1h", limit=200):
    try:
        params = {"symbol": symbol.upper(), "interval": normalize_interval(interval), "limit": int(limit)}
        r = get_session().get(BYBIT_KLINES, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        arr = js.get("result") or js.get("data") or js
        rows = []
        for row in arr:
            if isinstance(row, list):
                rows.append([pd.to_datetime(row[0], unit='s'), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
            elif isinstance(row, dict):
                t = row.get("open_time") or row.get("start_at") or row.get("t")
                o = float(row.get("open", row.get("open_price",0)))
                h = float(row.get("high", row.get("high_price",0)))
                l = float(row.get("low", row.get("low_price",0)))
                c = float(row.get("close", row.get("close_price",0)))
                v = float(row.get("volume", row.get("trade_volume",0)))
                rows.append([pd.to_datetime(t, unit='s'), o, h, l, c, v])
        df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume"])
        return df
    except Exception as e:
        raise

def fetch_klines_coingecko(symbol="BTCUSDT", interval="1h", limit=200):
    coin = COINGECKO_MAP.get(symbol.upper())
    if not coin:
        raise RuntimeError("CoinGecko fallback not supported for symbol: " + symbol)
    days = 7
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
    errors = []
    try:
        return fetch_klines_binance(symbol, interval, limit)
    except Exception as e:
        errors.append("Binance:"+str(e))
    try:
        return fetch_klines_bybit(symbol, interval, limit)
    except Exception as e:
        errors.append("Bybit:"+str(e))
    try:
        return fetch_klines_coingecko(symbol, interval, limit)
    except Exception as e:
        errors.append("Coingecko:"+str(e))
    raise RuntimeError("All data providers failed: " + " | ".join(errors))

# ---------- indicators ----------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100/(1+rs))

def compute_atr(df, period=14):
    high = df["high"]; low = df["low"]; close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    mc = ef - es
    msig = mc.ewm(span=signal, adjust=False).mean()
    hist = mc - msig
    return mc, msig, hist

# ---------- helpers ----------
def nice(x, nd=8):
    try: return float(round(x, nd))
    except: return x

def create_brand_image(title_lines):
    try:
        img = Image.new("RGB", (900,300), color=(18,18,20))
        draw = ImageDraw.Draw(img)
        try:
            font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
            font_text = ImageFont.truetype("DejaVuSans.ttf", 16)
        except:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
        draw.text((20,14), LOGO_TEXT, fill=(255,215,0), font=font_title)
        y = 64
        for line in title_lines:
            draw.text((20,y), line, fill=(230,230,230), font=font_text)
            y += 24
        buf = BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        return buf
    except Exception as e:
        traceback.print_exc()
        return None
        # part3: signal engine, AI wrapper, risk sizing
# ---------- signal engine ----------
def suggest_leverage(confidence, price, atr, base_max_leverage=MAX_LEVERAGE):
    try:
        vol_ratio = (atr / price) if price > 0 else 0
        base = max(1.0, confidence * base_max_leverage * 2)
        if vol_ratio > 0.03:
            base *= 0.3
        elif vol_ratio > 0.015:
            base *= 0.6
        lev = int(max(1, min(base_max_leverage, round(base))))
        return lev
    except Exception:
        return 1

def generate_signal_for(symbol="BTCUSDT", base_interval=DEFAULT_INTERVAL):
    try:
        df = fetch_klines_df(symbol, base_interval, limit=300)
        if df is None or len(df) < 20:
            return {"error":"insufficient data"}
        df["ema_fast"] = ema(df["close"], EMA_FAST)
        df["ema_slow"] = ema(df["close"], EMA_SLOW)
        df["rsi"] = rsi(df["close"], RSI_PERIOD)
        mc, msig, mh = macd(df["close"])
        df["macd_hist"] = mh
        atr_series = compute_atr(df, 14)
        atr = float(atr_series.iloc[-1]) if len(atr_series)>0 else 0.0
        last = df.iloc[-1]; prev = df.iloc[-2]

        signal = None; reasons=[]; score=0.0
        # EMA cross
        if (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"]):
            signal = "LONG"; reasons.append("EMA cross up"); score += 0.30
        if (prev["ema_fast"] >= prev["ema_slow"]) and (last["ema_fast"] < last["ema_slow"]):
            signal = "SHORT"; reasons.append("EMA cross down"); score += 0.30
        # MACD
        if last["macd_hist"] > 0: score += 0.10; reasons.append("MACD positive")
        else: score -= 0.03
        # wick checks
        body = abs(last["close"] - last["open"])
        upper_wick = last["high"] - max(last["close"], last["open"])
        lower_wick = min(last["close"], last["open"]) - last["low"]
        if body>0:
            ur = upper_wick / body; lr = lower_wick / body
            if ur > 2 and upper_wick > lower_wick:
                reasons.append("Upper-wick rejection"); score -= 0.12
                if not signal: signal = "SHORT"
            if lr > 2 and lower_wick > upper_wick:
                reasons.append("Lower-wick rejection"); score -= 0.12
                if not signal: signal = "LONG"
        r = float(df["rsi"].iloc[-1])
        if signal=="LONG" and r>80: reasons.append(f"RSI high {r:.1f}"); score -= 0.14
        if signal=="SHORT" and r<20: reasons.append(f"RSI low {r:.1f}"); score -= 0.14
        vol = float(last.get("volume",0.0))
        if MIN_VOLUME and vol < MIN_VOLUME: reasons.append("Low volume"); score -= 0.08
        price = float(last["close"])
        if atr > price * 0.03: reasons.append("High volatility"); score -= 0.30

        confidence = max(0.05, min(0.98, 0.5 + score))
        if signal == "LONG":
            swing_sl = float(df["low"].iloc[-3]); sl_by_atr = price - (atr * 1.5) if atr>0 else swing_sl
            sl = max(swing_sl, sl_by_atr)
            tp1 = price + (price - sl) * 1.5; tp2 = price + (price - sl) * 3
        elif signal == "SHORT":
            swing_sl = float(df["high"].iloc[-3]); sl_by_atr = price + (atr * 1.5) if atr>0 else swing_sl
            sl = min(swing_sl, sl_by_atr)
            tp1 = price - (sl - price) * 1.5; tp2 = price - (sl - price) * 3
        else:
            sl = price * 0.995; tp1 = price * 1.005; tp2 = price * 1.01

        suggested_leverage = suggest_leverage(confidence, price, atr, base_max_leverage=MAX_LEVERAGE)
        d = load_data()
        balance = d.get("challenge",{}).get("balance", CHALLENGE_START)
        suggested_risk_usd = round((balance * RISK_PERCENT) / 100.0, 8)
        diff = abs(price - sl) if abs(price - sl) > 1e-12 else 1e-12
        suggested_units = round(suggested_risk_usd / diff, 8)

        return {
            "symbol": symbol.upper(), "interval": base_interval, "time": datetime.utcnow().isoformat(),
            "signal": signal or "HOLD", "entry": nice(price), "sl": nice(sl), "tp1": nice(tp1), "tp2": nice(tp2),
            "atr": nice(atr), "rsi": round(r,2), "volume": vol, "reasons": reasons,
            "confidence": round(confidence,2),
            "suggested_risk_usd": suggested_risk_usd, "suggested_units": suggested_units,
            "suggested_leverage": int(suggested_leverage)
        }
    except Exception as e:
        return {"error": str(e)}

# ---------- AI wrapper ----------
def ai_analysis_text(prompt):
    if not openai_client:
        return "AI not configured."
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a motivational trading mentor. Give short punchy actionable advice and verdict (BUY/SELL)."}, {"role":"user","content":prompt}],
            max_tokens=240,
            temperature=0.25
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return f"AI error: {e}"

# ---------- image builder ----------
def build_signal_image(sig):
    title_lines = [
        f"{sig['symbol']}  |  {sig['interval']}  |  {sig['signal']}",
        f"Entry: {sig['entry']}   SL: {sig['sl']}   TP1: {sig['tp1']}",
        f"Confidence: {int(sig.get('confidence',0)*100)}%   Leverage: {sig.get('suggested_leverage',1)}x",
        "Reasons: " + (", ".join(sig.get('reasons',[])) if sig.get('reasons') else "None")
    ]
    return create_brand_image(title_lines)
    # part4: telegram handlers, pnl linking, background scanner, run
# ---------- persistence helpers ----------
last_signal_time = {}
def can_send_signal(symbol):
    last = last_signal_time.get(symbol.upper()); 
    if not last: return True
    return (datetime.utcnow() - last) > timedelta(minutes=SIGNAL_COOLDOWN_MIN)

def record_and_send(sig_obj, chat_id):
    d = load_data()
    sig_id = f"S{int(time.time())}"
    balance = d.get("challenge",{}).get("balance", CHALLENGE_START)
    risk_amt = round((balance * RISK_PERCENT) / 100.0, 8)
    rec = {"id": sig_id, "signal": sig_obj, "time": datetime.utcnow().isoformat(), "risk_amt": risk_amt, "pos_size": sig_obj.get("suggested_units",0), "result": None}
    d["signals"].append(rec)
    d["stats"]["total_signals"] = d["stats"].get("total_signals",0) + 1
    save_data(d)
    last_signal_time[sig_obj["symbol"]] = datetime.utcnow()

    ai_prompt = f"One-line motivational verdict and 1-sentence reason for this signal:\n{json.dumps(sig_obj, indent=2)}"
    ai_text = ai_analysis_text(ai_prompt)
    img = build_signal_image(sig_obj)
    text = (f"🔥 <b>Boss Destiny Signal</b> 🔥\nID: {sig_id}\nPair: {sig_obj['symbol']} | TF: {sig_obj['interval']}\nSignal: <b>{sig_obj['signal']}</b>\nEntry: {sig_obj['entry']}\nSL: {sig_obj['sl']}\nTP1: {sig_obj['tp1']}\n\n"
            f"💰 Risk: ${risk_amt:.4f} ({RISK_PERCENT}% of ${balance:.2f})\nUnits: {sig_obj.get('suggested_units')}  Leverage: {sig_obj.get('suggested_leverage')}x\nConfidence: {int(sig_obj.get('confidence',0)*100)}%\n\n🤖 Mentor: {ai_text}\n\nReply with: #link {sig_id} TP1  OR  #link {sig_id} SL after uploading PnL screenshot.")
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_pnl_{sig_id}"), types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_sig_{sig_id}"))
    try:
        if img:
            bot.send_photo(chat_id, img, caption=text, reply_markup=kb)
        else:
            bot.send_message(chat_id, text, reply_markup=kb)
    except Exception:
        traceback.print_exc()
        bot.send_message(chat_id, text, reply_markup=kb)
    return sig_id

# ---------- keyboard ----------
def main_keyboard():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("📈 Get Signal", callback_data="get_signal"),
           types.InlineKeyboardButton("🧠 AI Opinion", callback_data="ask_ai"))
    kb.add(types.InlineKeyboardButton("🔥 Trending Pairs", callback_data="trending"),
           types.InlineKeyboardButton("📰 Market News", callback_data="market_news"))
    kb.add(types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status"),
           types.InlineKeyboardButton("🧾 History", callback_data="history"))
    kb.add(types.InlineKeyboardButton("🔄 Refresh", callback_data="refresh_bot"),
           types.InlineKeyboardButton("⚙️ Bot Status", callback_data="bot_status"))
    return kb

# ---------- handlers ----------
@bot.message_handler(commands=['start','menu'])
def cmd_start(message):
    bot.send_message(message.chat.id, "Welcome back, boss 👑\nChoose an action:", reply_markup=main_keyboard())

@bot.callback_query_handler(func=lambda c: True)
def callback_handler(call):
    try:
        data = call.data; cid = call.message.chat.id
        if data == "get_signal":
            kb = types.InlineKeyboardMarkup()
            for p in PAIRS: kb.add(types.InlineKeyboardButton(p, callback_data=f"sigrun_{p}"))
            bot.send_message(cid, "Choose pair:", reply_markup=kb); return

        if data.startswith("sigrun_"):
            pair = data.split("_",1)[1]; bot.send_chat_action(cid, "typing"); bot.send_message(cid, f"Generating signal for {pair} — stay sharp 😤")
            sig = generate_signal_for(pair, DEFAULT_INTERVAL)
            if sig.get("error"): bot.send_message(cid, f"Signal error: {sig['error']}"); return
            record_and_send(sig, cid); return

        if data == "ask_ai":
            bot.send_message(cid, "🔥 Ask the AI mentor anything (start your message with AI: )"); return

        if data == "trending":
            try:
                s = get_session().get("https://api.binance.com/api/v3/ticker/24hr", timeout=8); s.raise_for_status(); tickers = s.json()
                out = ""
                for p in PAIRS:
                    t = next((x for x in tickers if x.get("symbol")==p), None)
                    if t: out += f"{p}: {float(t.get('priceChangePercent',0)):.2f}% Vol:{int(float(t.get('quoteVolume',0))):,}\n"
                bot.send_message(cid, "<b>🔥 Trending Pairs</b>\n\n" + out, parse_mode="HTML")
            except: bot.send_message(cid, "Failed to fetch trending pairs."); return

        if data == "market_news":
            try:
                r = get_session().get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN", timeout=8); r.raise_for_status(); js = r.json()
                items = js.get("Data", [])[:5]; text = "📰 Top News:\n"
                for it in items: text += f"• {it.get('title')}\n{it.get('url')}\n\n"
                bot.send_message(cid, text); return
            except: bot.send_message(cid, "Failed to fetch news."); return

        if data == "challenge_status":
            d = load_data(); c = d.get("challenge",{}); bot.send_message(cid, f"🏆 Challenge Bal: ${c.get('balance',CHALLENGE_START):.2f}\nWins:{c.get('wins',0)} Losses:{c.get('losses',0)} Target:${CHALLENGE_TARGET}"); return

        if data == "history":
            d = load_data(); recs = d.get("signals",[])[-30:]
            if not recs: bot.send_message(cid, "No history yet."); return
            txt = "Recent signals:\n"
            for r in reversed(recs[-20:]): s = r["signal"]; txt += f"{r['id']} {s['symbol']} {s['signal']} conf:{int(s.get('confidence',0)*100)}% result:{r.get('result') or '-'}\n"
            bot.send_message(cid, txt); return

        if data == "refresh_bot":
            bot.send_message(cid, "🔄 Refreshed ✅"); return

        if data == "bot_status":
            bot.send_message(cid, f"🤖 Bot Status\nTime: {datetime.utcnow().isoformat()}\nWatching: {', '.join(PAIRS)}"); return

        if data.startswith("link_pnl_"):
            sig_id = data.split("_",1)[1]; bot.send_message(cid, f"Upload your PnL screenshot then send: #link {sig_id} TP1 OR #link {sig_id} SL"); return

        if data.startswith("ai_sig_"):
            sig_id = data.split("_",1)[1]; d = load_data(); rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if not rec: bot.send_message(cid, "Signal not found."); return
            prompt = f"Give a motivational one-line plan for this signal:\n{json.dumps(rec['signal'], indent=2)}"
            analysis = ai_analysis_text(prompt); bot.send_message(cid, f"🤖 AI Analysis:\n{analysis}"); return

        bot.answer_callback_query(call.id, "Unknown action")
    except Exception:
        traceback.print_exc()
        try: bot.answer_callback_query(call.id, "Handler error")
        except: pass

# photo handler: save PnL screenshot
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id); downloaded = bot.download_file(file_info.file_path)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S"); fname = f"{UPLOAD_DIR}/{now}_{message.photo[-1].file_id}.jpg"
        with open(fname, "wb") as f: f.write(downloaded)
        d = load_data(); d["pnl"].append({"file": fname, "from": message.from_user.id, "time": now, "caption": message.caption, "linked": None}); save_data(d)
        bot.reply_to(message, "Screenshot saved. To link to a signal reply with: #link <signal_id> TP1 or #link <signal_id> SL")
    except Exception:
        traceback.print_exc(); bot.reply_to(message, "Failed to save screenshot.")

@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().startswith("#link"))
def link_pnl(message):
    try:
        parts = message.text.strip().split(); 
        if len(parts) < 3: bot.reply_to(message, "Usage: #link <signal_id> TP1 or SL"); return
        sig_id = parts[1]; tag = parts[2].upper()
        d = load_data(); pnl_item = None
        for p in reversed(d["pnl"]):
            if p.get("linked") is None and p["from"] == message.from_user.id: pnl_item = p; break
        if not pnl_item: bot.reply_to(message, "No unlinked screenshot found."); return
        pnl_item["linked"] = {"signal_id": sig_id, "result": tag, "linked_by": message.from_user.id}
        if message.from_user.id == ADMIN_ID:
            srec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if srec:
                risk_amt = srec.get("risk_amt", 0)
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
                    d["challenge"]["history"].append({"time": datetime.utcnow().isoformat(), "note": f"{sig_id} SL", "change": -loss})
                srec["result"] = tag
            save_data(d)
        else:
            save_data(d)
        bot.reply_to(message, f"Screenshot linked to {sig_id} as {tag}. Admin confirmation updates challenge.")
    except Exception:
        traceback.print_exc(); bot.reply_to(message, "Error linking screenshot.")

# AI one-shot handler: message starting with "AI:"
@bot.message_handler(func=lambda m: isinstance(m.text, str) and m.text.strip().upper().startswith("AI:"))
def handle_ai_text(message):
    prompt = message.text.strip()[3:].strip()
    if not prompt:
        bot.reply_to(message, "Usage: AI: <your question>")
        return
    bot.send_chat_action(message.chat.id, "typing")
    res = ai_analysis_text(prompt)
    bot.send_message(message.chat.id, f"🤖 AI:\n{res}")

# fallback text handler for pair names or menu
@bot.message_handler(func=lambda m: True)
def fallback(message):
    text = (message.text or "").strip()
    if text.lower() == "menu":
        bot.send_message(message.chat.id, "Menu", reply_markup=main_keyboard()); return
    if text.upper() in PAIRS:
        bot.send_chat_action(message.chat.id, "typing")
        sig = generate_signal_for(text.upper(), DEFAULT_INTERVAL)
        if sig.get("error"): bot.send_message(message.chat.id, f"Signal error: {sig['error']}"); return
        record_and_send(sig, message.chat.id); return
    bot.send_message(message.chat.id, "Tap a button to start", reply_markup=main_keyboard())

# ---------- background scanner (optional; admin toggles auto_scan flag in data.json) ----------
def scan_loop():
    while True:
        try:
            d = load_data()
            if d.get("auto_scan"):
                picks = []
                for p in PAIRS:
                    try:
                        sig = generate_signal_for(p, DEFAULT_INTERVAL)
                        if sig and sig.get("signal") in ("LONG","SHORT") and sig.get("confidence",0) > 0.4:
                            picks.append(sig)
                    except:
                        continue
                picks = sorted(picks, key=lambda x: x.get("confidence",0), reverse=True)[:3]
                for pick in picks:
                    record_and_send(pick, ADMIN_ID)
            time.sleep(60)
        except Exception:
            traceback.print_exc(); time.sleep(10)

scanner_thread = threading.Thread(target=scan_loop, daemon=True)
scanner_thread.start()

# ---------- start bot ----------
if __name__ == "__main__":
    print("🤖 Boss Destiny Trading Empire bot starting...")
    # ensure single instance: stop other polling instances before running
    bot.infinity_polling(timeout=60, long_polling_timeout=60)
