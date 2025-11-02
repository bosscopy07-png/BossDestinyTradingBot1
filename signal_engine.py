# signal_engine.py
import os
import traceback
from datetime import datetime
from math import isfinite
import numpy as np
import pandas as pd
from market_providers import fetch_klines_multi

# ---------- indicators ----------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
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

def bollinger_bands(series, period=20, std_factor=2):
    sma_val = sma(series, period)
    std = series.rolling(period).std()
    upper = sma_val + std_factor * std
    lower = sma_val - std_factor * std
    return upper, lower

# ---------- main generator ----------
def generate_signal(symbol="BTCUSDT", interval="1h"):
    """
    Return a signal dict with: symbol, interval, signal(LONG/SHORT/HOLD), entry, sl, tp1, confidence, reasons
    Uses multi-exchange public data (defaults to binance).
    """
    try:
        df = fetch_klines_multi(symbol=symbol, interval=interval, limit=500, exchange="binance")
        if df is None or len(df) < 30:
            return {"error": "Insufficient data for signal generation"}

        # config
        fast_ema = int(os.getenv("EMA_FAST", "9"))
        slow_ema = int(os.getenv("EMA_SLOW", "21"))
        rsi_period = int(os.getenv("RSI_PERIOD", "14"))
        bb_period = int(os.getenv("BB_PERIOD", "20"))
        bb_std = float(os.getenv("BB_STD", "2"))

        df["close"] = df["close"].astype(float)
        df["low"] = df["low"].astype(float)
        df["high"] = df["high"].astype(float)

        df["ema_fast"] = ema(df["close"], fast_ema)
        df["ema_slow"] = ema(df["close"], slow_ema)
        df["rsi"] = rsi(df["close"], rsi_period)
        macd_val, macd_sig, macd_hist = macd(df["close"])
        df["macd"] = macd_val
        df["macd_signal"] = macd_sig
        df["macd_hist"] = macd_hist
        df["bb_upper"], df["bb_lower"] = bollinger_bands(df["close"], bb_period, bb_std)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        signal = "HOLD"
        reasons = []
        score = 0.5

        # EMA cross
        if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
            signal = "LONG"
            reasons.append("EMA cross up")
            score += 0.14
        elif prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
            signal = "SHORT"
            reasons.append("EMA cross down")
            score += 0.14

        # MACD
        if last["macd_hist"] > 0 and signal == "LONG":
            score += 0.08; reasons.append("MACD positive")
        if last["macd_hist"] < 0 and signal == "SHORT":
            score += 0.08; reasons.append("MACD negative")

        # RSI sanity
        r = float(last["rsi"])
        if signal == "LONG" and r > 80:
            reasons.append("High RSI - caution"); score -= 0.08
        if signal == "SHORT" and r < 20:
            reasons.append("Low RSI - caution"); score -= 0.08

        # Bollinger context
        if signal == "LONG" and last["close"] < last["bb_lower"]:
            reasons.append("Price near lower BB"); score += 0.03
        if signal == "SHORT" and last["close"] > last["bb_upper"]:
            reasons.append("Price near upper BB"); score += 0.03

        price = float(last["close"])
        if signal == "LONG":
            sl = float(df["low"].iloc[-3])
            tp1 = price + (price - sl) * 1.5
        elif signal == "SHORT":
            sl = float(df["high"].iloc[-3])
            tp1 = price - (sl - price) * 1.5
        else:
            sl = price * 0.995
            tp1 = price * 1.005

        # risk sizing
        try:
            from storage import load_data
            d = load_data()
            balance = d.get("challenge", {}).get("balance", float(os.getenv("CHALLENGE_START", "100.0")))
        except Exception:
            balance = float(os.getenv("CHALLENGE_START", "100.0"))

        risk_pct = float(os.getenv("RISK_PERCENT", "1"))
        risk_usd = round((balance * risk_pct) / 100.0, 8)
        diff = abs(price - sl) if abs(price - sl) > 1e-12 else 1e-12
        units = round(risk_usd / diff, 8)

        confidence = max(0.05, min(0.99, score))

        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "timestamp": datetime.utcnow().isoformat(),
            "signal": "BUY" if signal == "LONG" else "SELL" if signal == "SHORT" else "HOLD",
            "entry": round(price, 8),
            "sl": round(sl, 8) if isfinite(sl) else None,
            "tp1": round(tp1, 8) if isfinite(tp1) else None,
            "confidence": round(confidence, 2),
            "reasons": reasons,
            "suggested_risk_usd": risk_usd,
            "suggested_units": units
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
