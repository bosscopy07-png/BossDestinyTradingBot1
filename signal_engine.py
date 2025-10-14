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
