# signal_engine.py
import os
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger("signal_engine")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(ch)

# Prefer market_providers.fetch_klines_multi if available (multi-exchange)
try:
    from market_providers import fetch_klines_multi
    HAS_MP = True
except Exception:
    fetch_klines_multi = None
    HAS_MP = False

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# default exchanges to scan (order: prefer reliable first)
DEFAULT_EXCHANGES = ["binance", "bybit", "kucoin", "okx"]

# ---------------- basic indicators ----------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    mc = ef - es
    msig = mc.ewm(span=signal, adjust=False).mean()
    hist = mc - msig
    return mc, msig, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ---------------- fetch fallback (Binance) ----------------
def _fetch_klines_binance(symbol: str, interval: str, limit: int = 300) -> Optional[pd.DataFrame]:
    try:
        s = requests.Session()
        r = s.get(BINANCE_KLINES, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
        r.raise_for_status()
        raw = r.json()
        cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(raw, columns=cols)
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df
    except Exception:
        LOG.exception("Binance klines fetch failed for %s %s", symbol, interval)
        return None

# ---------------- scoring helper for a single df ----------------
def _score_dataframe(df: pd.DataFrame) -> (float, List[str]):
    reasons = []
    try:
        if df is None or len(df) < 30:
            return 0.0, ["insufficient_data"]
        closes = df["close"].astype(float)
        # EMAs
        ema_fast = ema(closes, 9)
        ema_slow = ema(closes, 21)
        prev_ef = float(ema_fast.iloc[-2])
        prev_es = float(ema_slow.iloc[-2])
        cur_ef = float(ema_fast.iloc[-1])
        cur_es = float(ema_slow.iloc[-1])

        score = 0.5
        if prev_ef <= prev_es and cur_ef > cur_es:
            score += 0.18
            reasons.append("ema_cross_up")
        elif prev_ef >= prev_es and cur_ef < cur_es:
            score -= 0.18
            reasons.append("ema_cross_down")

        # MACD
        _, _, macd_hist = macd(closes)
        mh = float(macd_hist.iloc[-1])
        if mh > 0:
            score += 0.08
            reasons.append("macd_pos")
        else:
            score -= 0.03

        # RSI
        r = float(rsi(closes).iloc[-1])
        if r < 30:
            score += 0.06
            reasons.append("rsi_oversold")
        elif r > 70:
            score -= 0.06
            reasons.append("rsi_overbought")
        else:
            score += 0.02

        # momentum short
        lookback = min(4, len(closes)-1)
        ret = (closes.iloc[-1] - closes.iloc[-1-lookback]) / (closes.iloc[-1-lookback] + 1e-12)
        if ret > 0.01:
            score += 0.04
            reasons.append("momentum_pos")
        elif ret < -0.01:
            score -= 0.04
            reasons.append("momentum_neg")

        # simple engulfing
        try:
            if len(df) >= 2:
                prev = df.iloc[-2]
                last = df.iloc[-1]
                prev_body = abs(prev["close"] - prev["open"])
                last_body = abs(last["close"] - last["open"])
                if prev["close"] < prev["open"] and last["close"] > last["open"] and last_body > prev_body:
                    score += 0.12
                    reasons.append("bull_engulfing")
                elif prev["close"] > prev["open"] and last["close"] < last["open"] and last_body > prev_body:
                    score -= 0.12
                    reasons.append("bear_engulfing")
        except Exception:
            pass

        # clamp to 0..0.98
        score = max(0.0, min(0.98, score))
        return round(score, 4), reasons or ["none"]
    except Exception:
        LOG.exception("Error scoring dataframe")
        return 0.0, ["error_scoring"]

# ---------------- analyze one exchange ----------------
def _analyze_on_exchange(symbol: str, interval: str, exchange: str) -> Dict[str, Any]:
    """
    Attempt to fetch klines for (symbol, interval, exchange) and return structured analysis.
    """
    try:
        # fetch using market_providers if possible
        df = None
        if HAS_MP and fetch_klines_multi:
            try:
                df = fetch_klines_multi(symbol, interval, limit=300, exchange=exchange)
            except Exception:
                LOG.debug("fetch_klines_multi failed for %s on %s, falling back", symbol, exchange)
                df = None

        if df is None:
            # fallback to Binance for any exchange request (best-effort)
            df = _fetch_klines_binance(symbol, interval, limit=300)

        if df is None or df.empty or "close" not in df or len(df) < 30:
            return {"exchange": exchange, "error": "insufficient_data"}

        df = df.copy()
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)

        score, reasons = _score_dataframe(df)
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2])

        # direction decision
        if score >= 0.65:
            signal = "LONG" if last > prev else "SHORT"
        else:
            signal = "HOLD"

        # compute ATR-based SL/TP
        atr_val = float(atr(df).iloc[-1]) if len(df) > 14 else (df["high"].max() - df["low"].min()) * 0.01
        if signal == "LONG":
            sl = round(max(0.0, last - 1.5 * atr_val), 8)
            tp1 = round(last + 1.5 * atr_val, 8)
        elif signal == "SHORT":
            sl = round(last + 1.5 * atr_val, 8)
            tp1 = round(max(0.0, last - 1.5 * atr_val), 8)
        else:
            sl = None
            tp1 = None

        return {
            "exchange": exchange,
            "symbol": symbol.upper(),
            "interval": interval,
            "signal": signal,
            "entry": round(last, 8),
            "sl": sl,
            "tp1": tp1,
            "confidence": float(score),
            "reasons": reasons
        }
    except Exception:
        LOG.exception("analyze_on_exchange failed for %s %s %s", symbol, interval, exchange)
        return {"exchange": exchange, "error": "analysis_failed"}

# ---------------- main multi-exchange generator ----------------
def generate_signal_multi(symbol: str = "BTCUSDT", interval: str = "1h",
                          exchanges: List[str] = None) -> Dict[str, Any]:
    """
    Analyze across multiple exchanges and return best result plus per-exchange breakdown.
    Structure:
    {
      "symbol","interval","best": {...}, "per_exchange": [...], "timestamp": ISO
    }
    """
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES

    per = []
    best = None
    try:
        for ex in exchanges:
            try:
                res = _analyze_on_exchange(symbol, interval, ex)
                per.append(res)
                if isinstance(res, dict) and res.get("confidence") is not None and res.get("signal") in ("LONG","SHORT"):
                    if best is None or res["confidence"] > best.get("confidence", -1):
                        best = res
            except Exception:
                LOG.exception("sub-analysis exception for %s on %s", symbol, ex)
                per.append({"exchange": ex, "error": "exception"})

        # if no strong result but some holds, pick the highest confidence hold (for diagnostics)
        if best is None:
            # pick highest confidence among per entries (even if HOLD)
            scored = [p for p in per if isinstance(p, dict) and p.get("confidence") is not None]
            if scored:
                candidate = max(scored, key=lambda x: x.get("confidence", 0.0))
                best = candidate

        if best is None:
            return {
                "symbol": symbol.upper(),
                "interval": interval,
                "best": None,
                "per_exchange": per,
                "timestamp": datetime.utcnow().isoformat(),
                "error": "no_data_on_any_exchange"
            }

        # shape final combined dict (compatible with bot_runner expectations)
        out = {
            "symbol": best.get("symbol", symbol.upper()),
            "interval": best.get("interval", interval),
            "signal": best.get("signal", "HOLD"),
            "entry": best.get("entry"),
            "sl": best.get("sl"),
            "tp1": best.get("tp1"),
            "confidence": float(best.get("confidence", 0.0)),
            "reasons": best.get("reasons", []),
            "source_exchange": best.get("exchange"),
            "per_exchange": per,
            "timestamp": datetime.utcnow().isoformat()
        }
        return out

    except Exception:
        LOG.exception("generate_signal_multi crashed for %s %s", symbol, interval)
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "signal": "HOLD",
            "entry": None,
            "sl": None,
            "tp1": None,
            "confidence": 0.0,
            "reasons": ["error"],
            "per_exchange": per,
            "timestamp": datetime.utcnow().isoformat(),
            "error": "internal_exception"
        }

# Keep older single-call API name generate_signal for backward compat
def generate_signal(symbol: str = "BTCUSDT", interval: str = "1h") -> Dict[str, Any]:
    return generate_signal_multi(symbol, interval, exchanges=DEFAULT_EXCHANGES)

# quick local test
if __name__ == "__main__":
    print("Signal engine multi-exchange selftest")
    res = generate_signal_multi("BTCUSDT", "1h")
    import json
    print(json.dumps(res, indent=2, default=str))
