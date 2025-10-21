import os
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from market_providers import fetch_klines_multi as fetch_klines_df

# ======================================================
# CORE INDICATORS
# ======================================================
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


# ======================================================
# ADVANCED SIGNAL GENERATION
# ======================================================
def generate_signal(symbol="BTCUSDT", interval="1h"):
    """Enhanced adaptive trading signal generator"""
    try:
        df = fetch_klines_df(symbol, interval, limit=300)
        if df is None or len(df) < 30:
            return {"error": "Insufficient data for signal generation"}

        # --- Dynamic Settings ---
        fast_ema = int(os.getenv("EMA_FAST", "9"))
        slow_ema = int(os.getenv("EMA_SLOW", "21"))
        rsi_period = int(os.getenv("RSI_PERIOD", "14"))
        bb_period = int(os.getenv("BB_PERIOD", "20"))
        bb_std = float(os.getenv("BB_STD", "2"))

        # --- Compute Indicators ---
        df["ema_fast"] = ema(df["close"], fast_ema)
        df["ema_slow"] = ema(df["close"], slow_ema)
        df["rsi"] = rsi(df["close"], rsi_period)
        df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
        df["bb_upper"], df["bb_lower"] = bollinger_bands(df["close"], bb_period, bb_std)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        signal = "HOLD"
        reasons = []
        score = 0.5

        # --- EMA Cross ---
        if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
            signal = "LONG"
            reasons.append("EMA crossover (bullish)")
            score += 0.25
        elif prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
            signal = "SHORT"
            reasons.append("EMA crossover (bearish)")
            score += 0.25

        # --- MACD Confirmation ---
        if last["macd_hist"] > 0 and signal == "LONG":
            score += 0.1
            reasons.append("MACD supports bullish momentum")
        elif last["macd_hist"] < 0 and signal == "SHORT":
            score += 0.1
            reasons.append("MACD supports bearish momentum")

        # --- RSI Logic ---
        rsi_val = float(last["rsi"])
        if rsi_val > 80:
            reasons.append("Overbought RSI")
            if signal == "LONG": score -= 0.1
        elif rsi_val < 20:
            reasons.append("Oversold RSI")
            if signal == "SHORT": score -= 0.1

        # --- Bollinger Bounce ---
        if signal == "LONG" and last["close"] < last["bb_lower"]:
            score += 0.05
            reasons.append("Price near lower BB (potential rebound)")
        if signal == "SHORT" and last["close"] > last["bb_upper"]:
            score += 0.05
            reasons.append("Price near upper BB (potential pullback)")

        # --- Volatility Awareness ---
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        volatility = (atr / last["close"]) * 100
        if volatility < 1.0:
            reasons.append("Low volatility environment — weak momentum")
            score -= 0.05
        elif volatility > 5.0:
            reasons.append("High volatility — signals may be unstable")
            score -= 0.03

        # --- Multi-Timeframe Confirmation ---
        try:
            higher_tf = {"15m": "1h", "1h": "4h", "4h": "1d"}.get(interval, "4h")
            df_htf = fetch_klines_df(symbol, higher_tf, limit=200)
            if df_htf is not None and len(df_htf) > 50:
                df_htf["ema_fast"] = ema(df_htf["close"], fast_ema)
                df_htf["ema_slow"] = ema(df_htf["close"], slow_ema)
                last_htf = df_htf.iloc[-1]
                if last_htf["ema_fast"] > last_htf["ema_slow"] and signal == "LONG":
                    score += 0.15
                    reasons.append(f"Higher timeframe ({higher_tf}) confirms uptrend")
                elif last_htf["ema_fast"] < last_htf["ema_slow"] and signal == "SHORT":
                    score += 0.15
                    reasons.append(f"Higher timeframe ({higher_tf}) confirms downtrend")
                else:
                    reasons.append(f"Higher timeframe ({higher_tf}) neutral")
        except Exception:
            reasons.append("Higher timeframe confirmation skipped")

        # --- Price Levels ---
        price = float(last["close"])
        if signal == "LONG":
            sl = float(df["low"].iloc[-3])
            tp1 = price + (price - sl) * (1.5 if volatility < 3 else 1.2)
        elif signal == "SHORT":
            sl = float(df["high"].iloc[-3])
            tp1 = price - (sl - price) * (1.5 if volatility < 3 else 1.2)
        else:
            sl = price * 0.995
            tp1 = price * 1.005

        # --- Risk Management ---
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

        confidence = max(0.05, min(0.98, score))

        # --- Final Output ---
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "timestamp": datetime.utcnow().isoformat(),
            "signal": signal,
            "entry": round(price, 8),
            "sl": round(sl, 8),
            "tp1": round(tp1, 8),
            "confidence": round(confidence, 2),
            "volatility": round(volatility, 2),
            "reasons": reasons,
            "suggested_risk_usd": risk_usd,
            "suggested_units": units
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ======================================================
# BACKWARD COMPATIBILITY
# ======================================================
def generate_signal_for(symbol, interval):
    """Alias for older bots or scripts"""
    return generate_signal(symbol, interval)
