# ---------- bot.py (Part 1 of 3) ----------
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

from PIL import Image, ImageDraw, ImageFont

import telebot
from telebot import types

# plotting for candlesticks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------- CONFIG (env vars) -------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))          # your numeric Telegram id
CHANNEL_ID = os.getenv("CHANNEL_ID", None)         # optional channel (e.g. @yourchannel or -100...)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")   # optional for AI analysis

# Binance pairs (use BINANCE symbols like BTCUSDT, ETHUSDT)
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
SIGNAL_INTERVAL = os.getenv("SIGNAL_INTERVAL", "5m")   # label only for messages
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "5"))   # percent of balance risked per trade
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "0"))
SIGNAL_COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "10"))
CHALLENGE_TARGET = float(os.getenv("CHALLENGE_TARGET", "100"))
LOGO_PATH = os.getenv("LOGO_PATH", "bd_logo.png")

USE_WEBHOOK = os.getenv("USE_WEBHOOK", "0") in ("1", "true", "True", "yes")
APP_URL = os.getenv("APP_URL") or os.getenv("RENDER_EXTERNAL_URL")  # webhook url if used
PORT = int(os.getenv("PORT", "10000"))

DATA_FILE = "data.json"
UPLOAD_DIR = "uploads"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_24H = "https://api.binance.com/api/v3/ticker/24hr"

# ------------- sanity checks -------------
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required in environment variables.")
if ADMIN_ID == 0:
    raise RuntimeError("ADMIN_ID must be set to your numeric Telegram id.")

# ------------- bot init -------------
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

# ------------- storage helpers -------------
def atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def init_storage():
    if not os.path.exists(DATA_FILE):
        d = {
            "signals": [],       # {id, signal(dict), time, risk_amt, pos_size, user, result}
            "pnl": [],           # uploaded screenshots metadata
            "challenge": {"balance": CHALLENGE_START, "wins": 0, "losses": 0, "history": []},
            "stats": {"total_signals": 0, "wins": 0, "losses": 0},
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

# ------------- util helpers -------------
def nice_num(x, ndigits=8):
    try:
        return float(round(x, ndigits))
    except:
        return x

def send_logo_with_text(chat_id, text, reply_markup=None, reply_to=None, photo_bytes=None):
    """
    Sends logo (if exists) with caption text and optionally a chart image.
    If photo_bytes provided, logo is sent first (captioned) and then the chart image.
    """
    try:
        if photo_bytes:
            if os.path.exists(LOGO_PATH):
                with open(LOGO_PATH, "rb") as img:
                    bot.send_photo(chat_id, img, caption=text, reply_markup=reply_markup, reply_to_message_id=reply_to)
            bio = BytesIO(photo_bytes)
            bio.seek(0)
            bot.send_photo(chat_id, bio)
            return
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as img:
                bot.send_photo(chat_id, img, caption=text, reply_markup=reply_markup, reply_to_message_id=reply_to)
                return
    except Exception:
        pass
    bot.send_message(chat_id, text, reply_markup=reply_markup, reply_to_message_id=reply_to)

# ------------- Binance klines fetcher -------------
def fetch_klines_df(symbol="BTCUSDT", interval="5m", limit=300):
    """
    Returns DataFrame for klines from Binance with columns: open_time, open, high, low, close, volume
    """
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

# ------------- indicators -------------
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

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - macd_signal
    return macd_line, macd_signal, hist

# ------------- candlestick image -------------
def generate_candlestick_image(df, symbol):
    """
    Create a simple candlestick image for last ~60 candles and return bytes.
    """
    try:
        dfp = df.copy().tail(60)
        dates = mdates.date2num(dfp["open_time"].dt.to_pydatetime())
        o = dfp["open"].values
        h = dfp["high"].values
        l = dfp["low"].values
        c = dfp["close"].values

        fig, ax = plt.subplots(figsize=(8,4), dpi=100)
        if len(dates) > 1:
            width = (dates[1]-dates[0]) * 0.6
        else:
            width = 0.0005
        for i in range(len(dates)):
            color = "green" if c[i] >= o[i] else "red"
            ax.plot([dates[i], dates[i]], [l[i], h[i]], color=color, linewidth=0.8)
            rect = plt.Rectangle((dates[i]-width/2, min(o[i], c[i])), width, abs(c[i]-o[i]), color=color)
            ax.add_patch(rect)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b'))
        ax.set_title(symbol + " — last " + str(len(dfp)) + " candles")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print("generate_candlestick_image error:", e)
        return None
        # ---------- bot.py (Part 2 of 3) ----------
# ------------- PnL image generator -------------
def generate_pnl_image(signal_rec, srec, result_tag):
    """
    Creates a PnL summary image based on stored signal and returns path to saved PNG.
    signal_rec: dict inside srec['signal']
    srec: stored signal record (contains pos_size, risk_amt)
    result_tag: 'TP1', 'TP2' or 'SL'
    """
    try:
        entry = float(signal_rec.get("entry", 0))
        sl = float(signal_rec.get("sl", entry))
        tp1 = float(signal_rec.get("tp1", entry))
        tp2 = float(signal_rec.get("tp2", tp1))
        pos_size = float(srec.get("pos_size", 0.0))
        risk_amt = float(srec.get("risk_amt", 0.0))
        pair = signal_rec.get("symbol", "PAIR")
        tf = signal_rec.get("interval", "")
        side = signal_rec.get("signal", "HOLD")

        if result_tag.startswith("TP"):
            exit_price = tp1 if result_tag == "TP1" else tp2
        else:
            exit_price = sl

        if side == "SELL":
            pnl_units = (entry - exit_price) * pos_size
        else:
            pnl_units = (exit_price - entry) * pos_size
        pnl_usd = pnl_units
        pnl_pct = (pnl_usd / (risk_amt + 1e-9)) * 100 if risk_amt else 0.0

        # image canvas
        W, H = 1000, 600
        bg = (12, 12, 14)
        txt_color = (235, 235, 235)
        pos_color = (27, 153, 64)
        neg_color = (219, 64, 64)
        accent = pos_color if pnl_usd >= 0 else neg_color

        im = Image.new("RGB", (W, H), bg)
        draw = ImageDraw.Draw(im)

        try:
            font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
            font_m = ImageFont.truetype("DejaVuSans.ttf", 24)
            font_l = ImageFont.truetype("DejaVuSans.ttf", 40)
        except Exception:
            font_bold = ImageFont.load_default()
            font_m = ImageFont.load_default()
            font_l = ImageFont.load_default()

        # logo
        if os.path.exists(LOGO_PATH):
            try:
                logo = Image.open(LOGO_PATH).convert("RGBA")
                logo.thumbnail((120, 120))
                im.paste(logo, (30, 30), logo)
            except Exception:
                pass

        draw.text((170, 30), "Boss Destiny — Trade PnL Summary", font=font_bold, fill=txt_color)

        # left column
        left_x = 60
        y = 180
        spacing = 44
        draw.text((left_x, y), f"Pair: {pair}", font=font_l, fill=txt_color); y += spacing
        draw.text((left_x, y), f"TF: {tf}", font=font_m, fill=txt_color); y += spacing
        draw.text((left_x, y), f"Side: {side}", font=font_m, fill=txt_color); y += spacing
        draw.text((left_x, y), f"Entry: {entry:.8f}", font=font_m, fill=txt_color); y += spacing
        draw.text((left_x, y), f"Exit ({result_tag}): {exit_price:.8f}", font=font_m, fill=txt_color); y += spacing
        draw.text((left_x, y), f"Stop-loss: {sl:.8f}", font=font_m, fill=txt_color); y += spacing

        # right column
        right_x = 520
        y2 = 180
        draw.text((right_x, y2), f"Risk (per trade): ${risk_amt:.4f}", font=font_m, fill=txt_color); y2 += spacing
        draw.text((right_x, y2), f"Position size: {pos_size:.6f}", font=font_m, fill=txt_color); y2 += spacing
        draw.text((right_x, y2), f"P&L (USD): ${pnl_usd:.4f}", font=font_m, fill=accent); y2 += spacing
        draw.text((right_x, y2), f"P&L vs Risk: {pnl_pct:.2f}%", font=font_m, fill=accent); y2 += spacing
        draw.text((right_x, y2), f"Signal Confidence: {int(signal_rec.get('confidence',0)*100)}%", font=font_m, fill=txt_color); y2 += spacing

        # badge
        bx, by = 60, 420
        draw.rectangle((bx, by, bx+880, by+130), outline=accent, width=2)
        p_text = f"{'PROFIT' if pnl_usd>=0 else 'LOSS'}: ${pnl_usd:.4f} ({pnl_pct:.2f}%)"
        draw.text((bx+20, by+30), p_text, font=font_l, fill=accent)

        # footer
        draw.text((60, H-50), f"Signal ID: {srec.get('id','')}", font=font_m, fill=(150,150,150))
        draw.text((420, H-50), f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", font=font_m, fill=(150,150,150))

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        fname = os.path.join(UPLOAD_DIR, f"pnl_{srec.get('id','unknown')}_{int(time.time())}.png")
        im.save(fname, optimize=True)
        return fname
    except Exception as e:
        print("generate_pnl_image error:", e)
        return None

# ------------- risk sizing -------------
def compute_risk_and_size(entry, sl, balance, risk_percent):
    risk_amount = (balance * risk_percent) / 100.0
    diff = abs(entry - sl)
    if diff <= 1e-12:
        pos_size = 0.0
    else:
        pos_size = risk_amount / diff
    return round(risk_amount,8), round(float(pos_size),8)

# ------------- record & broadcast -------------
last_signal_time = {}   # symbol -> datetime

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
    rec = {"id": sig_id, "signal": sig, "time": datetime.utcnow().isoformat(),
           "risk_amt": risk_amt, "pos_size": pos_size, "user": user_id or ADMIN_ID, "result": None}
    d["signals"].append(rec)
    d["stats"]["total_signals"] = d["stats"].get("total_signals", 0) + 1
    save_data(d)
    last_signal_time[sig["symbol"]] = datetime.utcnow()

    stats = d.get("stats", {})
    wins = stats.get("wins", 0); total = stats.get("total_signals", 0)
    accuracy = (wins / total * 100) if total else 0.0

    # attempt to create chart image
    chart = None
    try:
        df = fetch_klines_df(sig["symbol"], interval=SIGNAL_INTERVAL, limit=120)
        chart = generate_candlestick_image(df, sig["symbol"])
    except Exception:
        chart = None

    text = (f"🔥 <b>Boss Destiny Signal</b> 🔥\n"
            f"ID: {sig_id}\nPair: {sig['symbol']} | TF: {sig['interval']}\nSignal: <b>{sig['signal']}</b>\nEntry: {sig['entry']}\nSL: {sig['sl']}\nTP1: {sig['tp1']} | TP2: {sig['tp2']}\n\n"
            f"💰 Capital (risk): ${risk_amt:.4f} ({RISK_PERCENT}% of ${balance:.2f})\n"
            f"📈 Position size (units): {pos_size}\n"
            f"🎯 Confidence: {int(sig['confidence']*100)}% | Accuracy: {accuracy:.1f}%\n"
            f"Reasons: {', '.join(sig['reasons']) if sig['reasons'] else 'None'}\n\nExecute manually on your exchange.")
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("📤 Post (Admin)", callback_data=f"post_{sig_id}"))
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_pnl_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 Ask AI", callback_data=f"ai_sig_{sig_id}"))

    target = chat_id or (CHANNEL_ID if CHANNEL_ID else ADMIN_ID)
    if chart:
        send_logo_with_text(target, text, reply_markup=kb, photo_bytes=chart)
    else:
        send_logo_with_text(target, text, reply_markup=kb)
    return sig_id

# ------------- scanner loop -------------
def scanner_loop():
    print("Scanner started for pairs:", PAIRS, "interval:", SIGNAL_INTERVAL)
    while True:
        try:
            for pair in PAIRS:
                try:
                    df = fetch_klines_df(pair, interval=SIGNAL_INTERVAL, limit=200)
                    sig = generate_signal_from_df(df, pair, SIGNAL_INTERVAL)
                    if sig.get("error"):
                        continue
                    if sig["signal"] in ("BUY", "SELL") and can_send_signal(pair):
                        # guard with RSI extremes
                        if (sig["signal"] == "BUY" and sig["rsi"] < 85) or (sig["signal"] == "SELL" and sig["rsi"] > 15):
                            record_and_send_signal(sig)
                            print("Auto-signal:", pair, sig["signal"])
                except Exception as e:
                    print("pair loop error", pair, e)
            # sleep aligned to interval (approx)
            sleep_secs = 60
            if SIGNAL_INTERVAL.endswith("m"):
                try:
                    m = int(SIGNAL_INTERVAL[:-1]); sleep_secs = max(60, m * 60)
                except:
                    sleep_secs = 60
            elif SIGNAL_INTERVAL.endswith("h"):
                try:
                    h = int(SIGNAL_INTERVAL[:-1]); sleep_secs = h * 3600
                except:
                    sleep_secs = 300
            time.sleep(sleep_secs)
        except Exception as e:
            print("scanner outer error:", e)
            time.sleep(10)

# ------------- AI analysis (optional) -------------
def ai_text_analysis(prompt):
    if not OPENAI_API_KEY:
        return "AI not configured."
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
        return f"AI error: {e}"

# ------------- small adapter: allow generate_signal_from_df (used by scanner) -------------
def generate_signal_from_df(df, symbol, interval_label=""):
    """
    Accept df (from fetch_klines_df) and compute same as generate_signal function.
    Returns dict or {"error": "..."}
    """
    try:
        if df is None or len(df) < 10:
            return {"error": "not enough data"}
        df2 = df.copy()
        df2["ema_fast"] = ema(df2["close"], EMA_FAST)
        df2["ema_slow"] = ema(df2["close"], EMA_SLOW)
        df2["rsi"] = rsi(df2["close"], RSI_PERIOD)
        macd_line, macd_signal, macd_hist = macd(df2["close"])
        df2["macd_line"] = macd_line; df2["macd_signal"] = macd_signal; df2["macd_hist"] = macd_hist

        last = df2.iloc[-1]; prev = df2.iloc[-2]
        signal = None; reasons = []; score = 0.0

        # EMA crossover
        if (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"]):
            signal = "BUY"; reasons.append("EMA cross (fast>slow)"); score += 0.30
        if (prev["ema_fast"] >= prev["ema_slow"]) and (last["ema_fast"] < last["ema_slow"]):
            signal = "SELL"; reasons.append("EMA cross (fast<slow)"); score += 0.30

        # MACD
        if last["macd_hist"] > 0:
            reasons.append("MACD hist positive"); score += 0.12
        else:
            score -= 0.05
        if (df2["macd_line"].iloc[-2] <= df2["macd_signal"].iloc[-2]) and (df2["macd_line"].iloc[-1] > df2["macd_signal"].iloc[-1]):
            reasons.append("MACD bullish cross"); score += 0.12
        if (df2["macd_line"].iloc[-2] >= df2["macd_signal"].iloc[-2]) and (df2["macd_line"].iloc[-1] < df2["macd_signal"].iloc[-1]):
            reasons.append("MACD bearish cross"); score += 0.12

        # wick rejection (simple)
        body = abs(last["close"] - last["open"])
        upper_wick = last["high"] - max(last["close"], last["open"])
        lower_wick = min(last["close"], last["open"]) - last["low"]
        if body > 0:
            upper_ratio = upper_wick / body
            lower_ratio = lower_wick / body
        else:
            upper_ratio = lower_ratio = 0
        if upper_ratio > 2 and upper_wick > lower_wick:
            reasons.append("Upper wick rejection"); score -= 0.1
            if not signal:
                signal = "SELL"
        if lower_ratio > 2 and lower_wick > upper_wick:
            reasons.append("Lower wick rejection"); score -= 0.1
            if not signal:
                signal = "BUY"

        # rsi
        r = last["rsi"]
        if signal == "BUY" and r > 80:
            reasons.append(f"RSI high ({r:.1f}) - caution"); score -= 0.15
        if signal == "SELL" and r < 20:
            reasons.append(f"RSI low ({r:.1f}) - caution"); score -= 0.15

        # volume
        if MIN_VOLUME and last["volume"] < MIN_VOLUME:
            reasons.append("Low volume"); score -= 0.05

        confidence = max(0.05, min(0.95, 0.5 + score))
        price = float(last["close"])
        if signal == "BUY":
            sl = float(df2["low"].iloc[-3])
            tp1 = price + (price - sl) * 1.5
            tp2 = price + (price - sl) * 3
        elif signal == "SELL":
            sl = float(df2["high"].iloc[-3])
            tp1 = price - (sl - price) * 1.5
            tp2 = price - (sl - price) * 3
        else:
            sl = price * 0.995
            tp1 = price * 1.005
            tp2 = price * 1.01

        return {
            "symbol": symbol.upper(),
            "interval": interval_label,
            "timestamp": datetime.utcnow().isoformat(),
            "signal": signal or "HOLD",
            "entry": nice_num(price),
            "sl": nice_num(sl),
            "tp1": nice_num(tp1),
            "tp2": nice_num(tp2),
            "rsi": round(float(r),2),
            "volume": float(last.get("volume", 0)),
            "reasons": reasons,
            "confidence": round(confidence,2)
        }
    except Exception as e:
        return {"error": str(e)}
        # ---------- bot.py (Part 3 of 3) ----------
# ------------- Telegram handlers -------------
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    data = call.data; user_id = call.from_user.id; chat_id = call.message.chat.id
    d = load_data()

    if data == "get_signal":
        kb = types.InlineKeyboardMarkup()
        for p in PAIRS:
            kb.add(types.InlineKeyboardButton(p, callback_data=f"signal_pair_{p}"))
        bot.send_message(chat_id, "Choose pair:", reply_markup=kb)
        return

    if data.startswith("signal_pair_"):
        pair = data.split("_",2)[2]
        try:
            df = fetch_klines_df(pair, interval=SIGNAL_INTERVAL, limit=200)
            sig = generate_signal_from_df(df, pair, SIGNAL_INTERVAL)
            if sig.get("error"):
                bot.send_message(chat_id, f"Error: {sig['error']}")
                return
            sig_id = record_and_send_signal(sig, chat_id=chat_id, user_id=user_id)
            bot.answer_callback_query(call.id, "Signal generated.")
        except Exception as e:
            bot.send_message(chat_id, f"Error fetching market data: {e}")
        return

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
                bot.send_message(CHANNEL_ID, f"📢 Official Signal:\n\n{rec['signal']}")
            except Exception as e:
                bot.send_message(user_id, f"Failed to post to channel: {e}")
        bot.send_message(user_id, f"Signal {sig_id} confirmed and (if channel set) posted.")
        bot.answer_callback_query(call.id, "Posted.")
        return

    if data.startswith("link_pnl_"):
        sig_id = data.split("_",1)[1]
        bot.send_message(chat_id, f"Reply with your screenshot image, then send: #link {sig_id} TP1 OR #link {sig_id} SL")
        return

    if data.startswith("ai_sig_"):
        sig_id = data.split("_",1)[1]
        rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
        if not rec:
            bot.answer_callback_query(call.id, "Signal not found.")
            return
        prompt = f"Analyze this signal and market context:\n{json.dumps(rec['signal'], indent=2)}\nProvide rationale, risk controls, and 2 alternative exits."
        out = ai_text_analysis(prompt)
        bot.send_message(chat_id, f"🤖 AI Analysis for {sig_id}:\n\n{out}")
        bot.answer_callback_query(call.id, "AI sent.")
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
        bot.send_message(chat_id, "Send the chart image in this chat (photo). The bot will save it.")
        return

    if data == "pnl_upload":
        bot.send_message(chat_id, "Upload your PnL screenshot now; then link with: #link <signal_id> TP1 or SL")
        return

    if data == "ask_ai":
        bot.send_message(chat_id, "Type your market question starting with: AI: <your question>")
        return

# photo handler
@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded = bot.download_file(file_info.file_path)
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    fname = os.path.join(UPLOAD_DIR, f"{now}_{message.photo[-1].file_id}.jpg")
    with open(fname, "wb") as f:
        f.write(downloaded)
    d = load_data()
    d["pnl"].append({"file": fname, "from": message.from_user.id, "time": now, "caption": message.caption})
    save_data(d)
    bot.reply_to(message, "Saved screenshot. To link to a signal: send #link <signal_id> TP1 or SL")

# #link handler
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
        bot.reply_to(message, "No unlinked screenshot found. Upload screenshot first.")
        return
    pnl_item["linked"] = {"signal_id": sig_id, "result": tag, "linked_by": message.from_user.id}
    # admin confirmation updates challenge
    if message.from_user.id == ADMIN_ID:
        srec = next((s for s in d["signals"] if s["id"]==sig_id), None)
        if srec:
            entry = float(srec["signal"].get("entry", 0))
            sl = float(srec["signal"].get("sl", entry))
            tp1 = float(srec["signal"].get("tp1", entry))
            tp2 = float(srec["signal"].get("tp2", tp1))
            pos_size = float(srec.get("pos_size", 0.0))
            if tag.startswith("TP"):
                exit_price = tp1 if tag == "TP1" else tp2
                side = srec["signal"].get("signal","BUY")
                pnl_units = (exit_price - entry) * pos_size if side != "SELL" else (entry - exit_price) * pos_size
                d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) + pnl_units
                d["challenge"]["wins"] = d["challenge"].get("wins",0) + 1
                d["stats"]["wins"] = d["stats"].get("wins",0) + 1
                d["challenge"]["history"].append({"time": datetime.utcnow().isoformat(), "note": f"{sig_id} {tag}", "change": float(pnl_units)})
            elif tag == "SL":
                exit_price = sl
                side = srec["signal"].get("signal","BUY")
                pnl_units = (exit_price - entry) * pos_size if side != "SELL" else (entry - exit_price) * pos_size
                d["challenge"]["balance"] = d["challenge"].get("balance", CHALLENGE_START) + pnl_units
                d["challenge"]["losses"] = d["challenge"].get("losses",0) + 1
                d["stats"]["losses"] = d["stats"].get("losses",0) + 1
                d["challenge"]["history"].append({"time": datetime.utcnow().isoformat(), "note": f"{sig_id} SL", "change": float(pnl_units)})
            srec["result"] = tag
            save_data(d)
            # generate PnL image and send
            try:
                pnl_path = generate_pnl_image(srec["signal"], srec, tag)
                if pnl_path and os.path.exists(pnl_path):
                    with open(pnl_path, "rb") as fh:
                        caption = f"PnL for {sig_id} — {tag}\nBalance: ${d['challenge']['balance']:.2f}"
                        bot.send_photo(message.chat.id, fh, caption=caption)
                        if CHANNEL_ID:
                            fh.seek(0)
                            bot.send_photo(CHANNEL_ID, open(pnl_path, "rb"), caption=f"[Official] {sig_id} — {tag}")
            except Exception as e:
                print("PnL image error:", e)
    else:
        save_data(d)
    bot.reply_to(message, f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance.")

# main menu builder
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

# text handler
@bot.message_handler(func=lambda m: True)
def all_messages(message):
    text = (message.text or "").strip()
    d = load_data()
    if message.from_user.id not in d.get("users", []):
        d["users"].append(message.from_user.id); save_data(d)

    if text.lower() == "menu":
        send_logo_with_text(message.chat.id, "Boss Destiny Menu", reply_markup=main_menu())
        return

    if text.startswith("AI:"):
        prompt = text[3:].strip()
        ans = ai_text_analysis(prompt)
        send_logo_with_text(message.chat.id, f"🤖 AI Answer:\n\n{ans}")
        return

    if text.lower().startswith("price "):
        # e.g., "price BTCUSDT"
        parts = text.split()
        if len(parts) >= 2:
            symbol = parts[1].upper()
            try:
                r = requests.get(BINANCE_TICKER_24H, params={"symbol": symbol}, timeout=10)
                if r.status_code == 200:
                    obj = r.json()
                    price = float(obj.get("lastPrice", 0))
                    change = float(obj.get("priceChangePercent", 0))
                    vol = float(obj.get("volume", 0))
                    send_logo_with_text(message.chat.id, f"{symbol} price: ${price:.4f}\n24h change: {change:.2f}%\nVolume: {vol}")
                    return
            except Exception:
                pass
        bot.reply_to(message, "Usage: price BTCUSDT")
        return

    # quick pair analysis: e.g., "BTCUSDT"
    if text.upper() in [p.upper() for p in PAIRS]:
        symbol = text.upper()
        try:
            df = fetch_klines_df(symbol, interval=SIGNAL_INTERVAL, limit=200)
            sig = generate_signal_from_df(df, symbol, SIGNAL_INTERVAL)
            if sig.get("error"):
                bot.reply_to(message, f"Error: {sig['error']}")
                return
            send_logo_with_text(message.chat.id, f"Quick analysis for {symbol}:\nSignal: {sig['signal']}\nEntry: {sig['entry']}\nSL: {sig['sl']}\nTP1: {sig['tp1']}\nConfidence: {int(sig['confidence']*100)}%")
        except Exception as e:
            bot.reply_to(message, f"Error fetching data: {e}")
        return

    send_logo_with_text(message.chat.id, "Tap a button to start:", reply_markup=main_menu())

# ------------- webhook endpoints (Flask) -------------
from flask import Flask, request
app = Flask(__name__)

@app.route("/" + BOT_TOKEN, methods=["POST"])
def telegram_webhook():
    json_str = request.get_data().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "Boss Destiny Trading Bot (Binance) is running.", 200

# ------------- bootstrap -------------
def set_webhook():
    if not APP_URL:
        print("APP_URL not set. Cannot set webhook.")
        return False
    webhook_url = APP_URL.rstrip("/") + "/" + BOT_TOKEN
    try:
        bot.remove_webhook()
    except Exception:
        pass
    try:
        bot.set_webhook(url=webhook_url)
        print("Webhook set to:", webhook_url)
        return True
    except Exception as e:
        print("Failed setting webhook:", e)
        return False

def start_polling_safe():
    while True:
        try:
            print("Starting polling...")
            bot.infinity_polling(timeout=60, long_polling_timeout=60)
        except Exception as e:
            print("Polling error:", e)
            time.sleep(5)

if __name__ == "__main__":
    # start scanner thread
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    print("Scanner thread started.")
    if USE_WEBHOOK and APP_URL:
        set_webhook()
        print("Starting Flask server on port", PORT)
        app.run(host="0.0.0.0", port=PORT)
    else:
        # small Flask binder to satisfy platforms that probe port, then start polling
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT), daemon=True).start()
        start_polling_safe()
