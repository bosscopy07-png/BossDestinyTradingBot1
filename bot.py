# ---------- bot.py (PART 1 of 3) ----------
import os
import json
import time
import math
import threading
import requests
from datetime import datetime, timedelta
from io import BytesIO
import traceback

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import telebot
from telebot import types

# optional plotting for candlesticks
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ------------- CONFIG -------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
CHANNEL_ID = os.getenv("CHANNEL_ID")  # optional
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,SOLUSDT").split(",")
SIGNAL_INTERVAL = os.getenv("SIGNAL_INTERVAL", "5m")
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "5"))  # percent of balance risked per trade
CHALLENGE_START = float(os.getenv("CHALLENGE_START", "10"))
CHALLENGE_TARGET = float(os.getenv("CHALLENGE_TARGET", "100"))
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "0"))
SIGNAL_COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "30"))
LOGO_PATH = os.getenv("LOGO_PATH", "bd_logo.png")
DATA_FILE = "data.json"
UPLOAD_DIR = "uploads"

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_24H = "https://api.binance.com/api/v3/ticker/24hr"

# interval normalization (fixes bad inputs like "1hrs")
VALID_INTERVALS = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","2h":"2h","4h":"4h","6h":"6h","12h":"12h","1d":"1d"}
def normalize_interval(s):
    s2 = s.strip().lower()
    if s2.endswith("hrs"): s2 = s2.replace("hrs","h")
    if s2.endswith("hours"): s2 = s2.replace("hours","h")
    if s2.endswith("hour"): s2 = s2.replace("hour","h")
    if s2.endswith("mins"): s2 = s2.replace("mins","m")
    if s2.endswith("min"): s2 = s2.replace("min","m")
    if s2 in VALID_INTERVALS: return VALID_INTERVALS[s2]
    if s2.endswith(("m","h","d")): return s2
    return SIGNAL_INTERVAL

# sanity
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is required.")
if ADMIN_ID == 0:
    raise RuntimeError("ADMIN_ID environment variable is required (your numeric Telegram id).")

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
            "signals": [],   # stored signals
            "pnl": [],       # uploaded screenshots metadata
            "challenge": {"balance": CHALLENGE_START, "wins": 0, "losses": 0, "history": []},
            "stats": {"total_signals": 0, "wins": 0, "losses": 0},
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

# ------------- small utils -------------
def nice(x, nd=8):
    try:
        return float(round(x, nd))
    except:
        return x

def log_exc(e):
    print("ERROR:", e)
    traceback.print_exc()

def send_logo(chat_id, text, reply_markup=None, photo_bytes=None, reply_to=None):
    try:
        # if a chart provided, send logo caption first (if available) then chart
        if photo_bytes:
            if os.path.exists(LOGO_PATH):
                with open(LOGO_PATH, "rb") as logo_f:
                    bot.send_photo(chat_id, logo_f, caption=text, reply_markup=reply_markup, reply_to_message_id=reply_to)
            bio = BytesIO(photo_bytes); bio.seek(0)
            bot.send_photo(chat_id, bio)
            return
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as logo_f:
                bot.send_photo(chat_id, logo_f, caption=text, reply_markup=reply_markup, reply_to_message_id=reply_to)
                return
    except Exception as e:
        log_exc(e)
    bot.send_message(chat_id, text, reply_markup=reply_markup, reply_to_message_id=reply_to)

# ------------- Binance fetcher with retries -------------
def fetch_klines_df(symbol="BTCUSDT", interval="1h", limit=300):
    interval = normalize_interval(interval)
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    tries = 0
    while tries < 3:
        try:
            r = requests.get(BINANCE_KLINES, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
            df = pd.DataFrame(data, columns=cols)
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            return df
        except requests.HTTPError as http_e:
            tries += 1
            print(f"fetch klines HTTP error {http_e} attempt {tries} for {symbol} {interval}")
            time.sleep(1 + tries)
        except Exception as e:
            tries += 1
            print(f"fetch klines error {e} attempt {tries} for {symbol} {interval}")
            time.sleep(1 + tries)
    raise RuntimeError(f"Failed to fetch klines for {symbol} interval {interval} after retries")

def fetch_ticker_24h(symbol="BTCUSDT"):
    try:
        r = requests.get(BINANCE_TICKER_24H, params={"symbol": symbol.upper()}, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("ticker24h error", e)
        return {}

# ------------- indicators -------------
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

def macd(series, f=12, s=26, sig=9):
    ef = series.ewm(span=f, adjust=False).mean()
    es = series.ewm(span=s, adjust=False).mean()
    mc = ef - es
    msig = mc.ewm(span=sig, adjust=False).mean()
    return mc, msig, mc - msig

# ------------- candlestick chart generation (optional) -------------
def gen_candle_img(df, symbol):
    if not MATPLOTLIB_AVAILABLE:
        return None
    try:
        dfp = df.copy().tail(60)
        dates = mdates.date2num(dfp["open_time"].dt.to_pydatetime())
        o = dfp["open"].values; h = dfp["high"].values; l = dfp["low"].values; c = dfp["close"].values
        fig, ax = plt.subplots(figsize=(8,4), dpi=100)
        width = (dates[1]-dates[0])*0.6 if len(dates)>1 else 0.0005
        for i in range(len(dates)):
            color = "green" if c[i] >= o[i] else "red"
            ax.plot([dates[i],dates[i]],[l[i],h[i]], color=color, linewidth=0.8)
            rect = plt.Rectangle((dates[i]-width/2, min(o[i],c[i])), width, abs(c[i]-o[i]), color=color)
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
        print("gen_candle_img error", e)
        return None
        # ---------- bot.py (PART 2 of 3) ----------
# ------------- signal engine -------------
def generate_signal_from_df(df, symbol, interval_label=""):
    try:
        if df is None or len(df) < 20:
            return {"error": "not enough data"}
        df2 = df.copy()
        df2["ema_fast"] = ema(df2["close"], EMA_FAST)
        df2["ema_slow"] = ema(df2["close"], EMA_SLOW)
        df2["rsi"] = rsi(df2["close"], RSI_PERIOD)
        mc, msig, mhist = macd(df2["close"])
        df2["macd_line"], df2["macd_signal"], df2["macd_hist"] = mc, msig, mhist

        last = df2.iloc[-1]; prev = df2.iloc[-2]
        signal = None; reasons=[]; score = 0.0

        # EMA cross
        if (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"]):
            signal = "BUY"; reasons.append("EMA cross up"); score += 0.30
        if (prev["ema_fast"] >= prev["ema_slow"]) and (last["ema_fast"] < last["ema_slow"]):
            signal = "SELL"; reasons.append("EMA cross down"); score += 0.30

        # MACD hist
        if last["macd_hist"] > 0:
            score += 0.12; reasons.append("MACD positive")
        else:
            score -= 0.05

        # wick rejection
        body = abs(last["close"] - last["open"])
        upper_wick = last["high"] - max(last["close"], last["open"])
        lower_wick = min(last["close"], last["open"]) - last["low"]
        upper_ratio = (upper_wick / (body+1e-9)) if body>0 else 0
        lower_ratio = (lower_wick / (body+1e-9)) if body>0 else 0
        if upper_ratio > 2 and upper_wick > lower_wick:
            reasons.append("Upper wick rejection"); score -= 0.10
            if not signal: signal = "SELL"
        if lower_ratio > 2 and lower_wick > upper_wick:
            reasons.append("Lower wick rejection"); score -= 0.10
            if not signal: signal = "BUY"

        # RSI guard
        r = last["rsi"]
        if signal == "BUY" and r > 80:
            reasons.append(f"RSI high {r:.1f}"); score -= 0.15
        if signal == "SELL" and r < 20:
            reasons.append(f"RSI low {r:.1f}"); score -= 0.15

        # volume
        vol = float(last.get("volume",0))
        if MIN_VOLUME and vol < MIN_VOLUME:
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
            "time": datetime.utcnow().isoformat(),
            "signal": signal or "HOLD",
            "entry": nice(price),
            "sl": nice(sl),
            "tp1": nice(tp1),
            "tp2": nice(tp2),
            "rsi": round(float(r),2),
            "volume": vol,
            "reasons": reasons,
            "confidence": round(confidence,2)
        }
    except Exception as e:
        log_exc(e)
        return {"error": str(e)}

# ------------- risk sizing & recording -------------
def compute_risk_size(entry, sl, balance, risk_percent):
    risk_amount = (balance * risk_percent)/100.0
    diff = abs(entry - sl)
    if diff <= 1e-12:
        pos_size = 0.0
    else:
        pos_size = risk_amount / diff
    return round(risk_amount,8), round(float(pos_size),8)

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
    risk_amt, pos_size = compute_risk_size(sig["entry"], sig["sl"], balance, RISK_PERCENT)
    rec = {"id":sig_id, "signal":sig, "time": datetime.utcnow().isoformat(), "risk_amt":risk_amt, "pos_size":pos_size, "user":user_id or ADMIN_ID, "result":None}
    d["signals"].append(rec)
    d["stats"]["total_signals"] = d["stats"].get("total_signals",0) + 1
    save_data(d)
    last_signal_time[sig["symbol"]] = datetime.utcnow()

    # accuracy
    stats = d.get("stats",{})
    wins = stats.get("wins",0); total = stats.get("total_signals",0)
    accuracy = (wins/total*100) if total else 0.0

    # chart bytes optional
    chart = None
    try:
        df = fetch_klines_df(sig["symbol"], interval=SIGNAL_INTERVAL, limit=120)
        chart = gen_candle_img(df, sig["symbol"]) if MATPLOTLIB_AVAILABLE else None
    except Exception:
        chart = None

    text = (f"🔥 <b>Boss Destiny Signal</b> 🔥\nID: {sig_id}\nPair: {sig['symbol']} | TF: {sig['interval']}\nSignal: <b>{sig['signal']}</b>\nEntry: {sig['entry']}\nSL: {sig['sl']}\nTP1: {sig['tp1']} | TP2: {sig['tp2']}\n\n"
            f"💰 Risk per trade: ${risk_amt:.4f} ({RISK_PERCENT}% of ${balance:.2f})\n"
            f"📈 Pos size: {pos_size}\n"
            f"🎯 Conf: {int(sig['confidence']*100)}% | Accuracy: {accuracy:.1f}%\nReasons: {', '.join(sig['reasons']) if sig['reasons'] else 'None'}")

    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("📤 Post (Admin)", callback_data=f"post_{sig_id}"))
    kb.add(types.InlineKeyboardButton("📸 Link PnL", callback_data=f"link_pnl_{sig_id}"))
    kb.add(types.InlineKeyboardButton("🤖 AI Analysis", callback_data=f"ai_sig_{sig_id}"))
    kb.add(types.InlineKeyboardButton("📊 Add to Portfolio", callback_data=f"add_port_{sig_id}"))

    target = chat_id or (CHANNEL_ID if CHANNEL_ID else ADMIN_ID)
    send_logo(target, text, reply_markup=kb, photo_bytes=chart)
    return sig_id

# ------------- scan & allocation -------------
def scan_and_get_top4(pairs=None, interval=SIGNAL_INTERVAL):
    pairs = pairs or PAIRS
    picks = []
    for p in pairs:
        try:
            df = fetch_klines_df(p, interval=interval, limit=200)
            sig = generate_signal_from_df(df, p, interval)
            if "error" in sig:
                continue
            picks.append(sig)
        except Exception as e:
            print(f"scan error {p}", e)
    picks_sorted = sorted(picks, key=lambda x: x.get("confidence",0), reverse=True)
    top4 = picks_sorted[:4]
    d = load_data(); d["last_scan"] = {"time": datetime.utcnow().isoformat(), "picks": top4}; save_data(d)
    return top4

def suggest_allocation_for_picks(top4, balance):
    suggestions = []
    total_conf = sum([t.get("confidence",0) for t in top4]) or 1.0
    for t in top4:
        conf = t.get("confidence",0)
        allocated_capital = (conf / total_conf) * balance
        risk_amt = (balance * RISK_PERCENT)/100.0
        entry = t["entry"]; sl = t["sl"]
        diff = abs(entry - sl) if abs(entry - sl) > 1e-12 else 1e-12
        pos_size = risk_amt / diff
        suggestions.append({"signal": t, "allocated_capital": round(allocated_capital,4), "risk_amt":round(risk_amt,6), "pos_size":round(pos_size,6)})
    return suggestions

# ------------- scanner loop (auto) -------------
def scanner_loop():
    print("Scanner started (auto_scan)")
    while True:
        d = load_data()
        if d.get("auto_scan", False):
            try:
                top = scan_and_get_top4(PAIRS, SIGNAL_INTERVAL)
                if top:
                    balance = d["challenge"].get("balance", CHALLENGE_START)
                    sug = suggest_allocation_for_picks(top, balance)
                    text = f"🤖 Auto-scan Top {len(sug)} picks (Balance ${balance:.2f}):\n"
                    for s in sug:
                        sig = s["signal"]
                        text += f"- {sig['symbol']} {sig['signal']} conf:{int(sig['confidence']*100)}% entry:{sig['entry']} SL:{sig['sl']} alloc:${s['allocated_capital']:.2f}\n"
                    send_logo(ADMIN_ID, text)
                time.sleep(60)
            except Exception as e:
                print("scanner error:", e)
                time.sleep(10)
        else:
            time.sleep(5)

# ------------- AI wrapper -------------
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
            max_tokens=500,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log_exc(e)
        return f"AI error: {e}"
        # ---------- bot.py (PART 3 of 3) ----------
# ------------- Telegram callback handlers -------------
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    data = call.data; user_id = call.from_user.id; chat_id = call.message.chat.id
    d = load_data()

    if data == "get_signal":
        kb = types.InlineKeyboardMarkup()
        for p in PAIRS:
            kb.add(types.InlineKeyboardButton(p, callback_data=f"signal_pair_{p}"))
        kb.add(types.InlineKeyboardButton("Scan Top 4", callback_data="scan_top4"))
        bot.send_message(chat_id, "Choose pair or scan top picks:", reply_markup=kb)
        return

    if data.startswith("signal_pair_"):
        pair = data.split("_",2)[2]
        try:
            df = fetch_klines_df(pair, interval=SIGNAL_INTERVAL, limit=300)
            sig = generate_signal_from_df(df, pair, SIGNAL_INTERVAL)
            if sig.get("error"):
                bot.send_message(chat_id, f"Error: {sig['error']}")
                return
            record_and_send(sig, chat_id=chat_id, user_id=user_id)
            bot.answer_callback_query(call.id, "Signal sent.")
        except Exception as e:
            bot.send_message(chat_id, f"Error fetching market data: {e}")
        return

    if data == "scan_top4":
        try:
            top4 = scan_and_get_top4(PAIRS, SIGNAL_INTERVAL)
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
        ld = load_data()
        picks = ld.get("last_scan", {}).get("picks", [])
        pick = next((p for p in picks if p["symbol"].upper() == sym.upper()), None)
        if not pick:
            bot.answer_callback_query(call.id, "Pick not found in last scan.")
            return
        record_and_send(pick, chat_id=chat_id, user_id=user_id)
        bot.answer_callback_query(call.id, "Pick sent.")
        return

    if data == "alloc_top4":
        ld = load_data()
        picks = ld.get("last_scan", {}).get("picks", [])
        if not picks:
            bot.answer_callback_query(call.id, "No last scan data.")
            return
        bal = ld["challenge"].get("balance", CHALLENGE_START)
        sug = suggest_allocation_for_picks(picks, bal)
        txt = f"Allocation suggestions (balance ${bal:.2f}):\n"
        for s in sug:
            txt += f"- {s['signal']['symbol']} {s['signal']['signal']} conf:{int(s['signal']['confidence']*100)}% alloc:${s['allocated_capital']:.2f} risk:${s['risk_amt']:.4f} pos:{s['pos_size']}\n"
        send_logo(chat_id, txt)
        bot.answer_callback_query(call.id, "Allocation shown.")
        return

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

    if data.startswith("link_pnl_"):
        sig_id = data.split("_",1)[1]
        bot.send_message(chat_id, f"Reply to this message with your screenshot, then send: #link {sig_id} TP1 or #link {sig_id} SL")
        return

    if data.startswith("ai_sig_"):
        sig_id = data.split("_",1)[1]
        rec = next((s for s in d["signals"] if s["id"]==sig_id), None)
        if not rec:
            bot.answer_callback_query(call.id, "Signal not found.")
            return
        prompt = f"Analyze this trading signal and market context:\n{json.dumps(rec['signal'], indent=2)}\nProvide rationale, risk controls, and two alternative exits."
        out = ai_text_analysis(prompt)
        bot.send_message(chat_id, f"🤖 AI:\n\n{out}")
        bot.answer_callback_query(call.id, "AI sent.")
        return

    if data == "challenge_status":
        c = d["challenge"]
        wins = c.get("wins",0); losses = c.get("losses",0); bal = c.get("balance", CHALLENGE_START)
        total = wins + losses
        acc = (wins/total*100) if total else 0.0
        txt = f"🏆 Challenge\nBalance: ${bal:.2f}\nWins: {wins} Losses: {losses}\nAccuracy: {acc:.1f}%\nTarget: ${CHALLENGE_TARGET}"
        send_logo(chat_id, txt, reply_markup=main_menu())
        return

    if data == "send_chart_info":
        bot.send_message(chat_id, "Send chart image (photo).")
        return

    if data == "pnl_upload":
        bot.send_message(chat_id, "Upload your PnL screenshot now; then link it with: #link <signal_id> TP1 or SL")
        return

    if data == "ask_ai":
        bot.send_message(chat_id, "Type your market question starting with: AI: <your question>")
        return

    if data == "history":
        recent = d.get("signals", [])[-10:]
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
        d = load_data()
        rows = []
        for r in d.get("signals", []):
            s = r["signal"]
            rows.append({
                "id": r["id"], "symbol": s["symbol"], "signal": s["signal"], "entry": s["entry"],
                "sl": s["sl"], "tp1": s["tp1"], "confidence": s.get("confidence",0), "time": r.get("time"), "result": r.get("result")
            })
        if not rows:
            bot.send_message(chat_id, "No records to export.")
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
        d = load_data()
        d["auto_scan"] = not d.get("auto_scan", False)
        save_data(d)
        bot.answer_callback_query(call.id, f"Auto-scan set to {d['auto_scan']}")
        bot.send_message(user_id, f"Auto-scan is now {d['auto_scan']}")
        return

    bot.answer_callback_query(call.id, "Unknown action.")

# ------------- photo upload handler -------------
@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        fname = os.path.join(UPLOAD_DIR, f"{now}_{message.photo[-1].file_id}.jpg")
        with open(fname,"wb") as f:
            f.write(downloaded)
        d = load_data()
        d["pnl"].append({"file": fname, "from": message.from_user.id, "time": now, "caption": message.caption, "linked": None})
        save_data(d)
        bot.reply_to(message, "Saved screenshot. To link: send #link <signal_id> TP1 or SL")
    except Exception as e:
        log_exc(e)
        bot.reply_to(message, "Failed to save image.")

# ------------- #link handler -------------
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
        if message.from_user.id == ADMIN_ID:
            srec = next((s for s in d["signals"] if s["id"]==sig_id), None)
            if srec:
                entry = float(srec["signal"].get("entry",0))
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
            try:
                from_path = pnl_item["file"]
                bot.send_photo(message.chat.id, open(from_path,"rb"), caption=f"Linked {sig_id} as {tag}. Balance: ${d['challenge']['balance']:.2f}")
            except Exception as e:
                log_exc(e)
        else:
            save_data(d)
        bot.reply_to(message, f"Linked screenshot to {sig_id} as {tag}. Admin confirmation updates balance.")
    except Exception as e:
        log_exc(e)
        bot.reply_to(message, "Error linking screenshot.")

# ------------- main menu builder -------------
def main_menu():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📈 Get Signal", callback_data="get_signal"),
        types.InlineKeyboardButton("🔎 Scan Top 4", callback_data="scan_top4"),
        types.InlineKeyboardButton("📊 My Challenge", callback_data="challenge_status"),
        types.InlineKeyboardButton("⚙️ Risk Settings", callback_data="risk_settings"),
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

# ------------- message handler -------------
@bot.message_handler(func=lambda m: True)
def all_messages(message):
    text = (message.text or "").strip()
    d = load_data()
    if message.from_user.id not in d.get("users", []):
        d["users"].append(message.from_user.id); save_data(d)

    if text.lower() == "menu":
        send_logo(message.chat.id, "Boss Destiny Menu", reply_markup=main_menu())
        return

    if text.startswith("AI:"):
        prompt = text[3:].strip()
        out = ai_text_analysis(prompt)
        send_logo(message.chat.id, f"🤖 AI:\n\n{out}")
        return

    if text.lower().startswith("price "):
        parts = text.split()
        if len(parts) >= 2:
            sym = parts[1].upper()
            try:
                t = fetch_ticker_24h(sym)
                if not t:
                    bot.reply_to(message, f"No ticker for {sym}")
                    return
                price = float(t.get("lastPrice",0)); change = float(t.get("priceChangePercent",0)); vol = float(t.get("volume",0))
                send_logo(message.chat.id, f"{sym} price: ${price:.6f}\n24h change: {change:.2f}%\nVolume: {vol}")
                return
            except Exception as e:
                bot.reply_to(message, f"Ticker error: {e}")
                return
        bot.reply_to(message, "Usage: price BTCUSDT")
        return

    if text.upper() in [p.upper() for p in PAIRS]:
        sym = text.upper()
        try:
            df = fetch_klines_df(sym, interval=SIGNAL_INTERVAL, limit=300)
            sig = generate_signal_from_df(df, sym, SIGNAL_INTERVAL)
            if sig.get("error"):
                bot.reply_to(message, f"Error: {sig['error']}")
                return
            send_logo(message.chat.id, f"Quick analysis for {sym}:\nSignal: {sig['signal']}\nEntry: {sig['entry']}\nSL: {sig['sl']}\nTP1: {sig['tp1']}\nConf: {int(sig['confidence']*100)}%")
            return
        except Exception as e:
            bot.reply_to(message, f"Error fetching: {e}")
            return

    send_logo(message.chat.id, "Tap a button:", reply_markup=main_menu())

# ------------- startup -------------
def start_bot():
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    print("Scanner thread started.")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

if __name__ == "__main__":
    print("Starting Boss Destiny Trading Bot (improved)...")
    start_bot()
