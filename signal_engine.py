# signal_engine.py
import logging
import traceback
from typing import Dict, Any

log = logging.getLogger("signal_engine")
log.setLevel(logging.INFO)

# Prefer the advanced multi-tf analyzer in market_providers if available
try:
    from market_providers import analyze_pair_multi_timeframes, fetch_klines_multi
except Exception:
    analyze_pair_multi_timeframes = None
    fetch_klines_multi = None
    log.exception("market_providers not available in signal_engine")

# Simple fallback engine (if advanced analyzer not present)
def _fallback_simple_signal(symbol: str, interval: str) -> Dict[str, Any]:
    """
    Basic indicator-based fallback: loads 200 candles from Binance (via fetch_klines_multi if available)
    and uses simple MA/RSI cross logic to return a signal dict.
    """
    try:
        df = None
        if fetch_klines_multi:
            df = fetch_klines_multi(symbol, interval, limit=200, exchange="binance")
        # if no df returned -> error
        if df is None or df.empty:
            return {"error": "insufficient_data"}

        # compute simple indicators
        close = df["close"].astype(float)
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rs = gain / (loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))

        # decide
        if ma20 > ma50 and rsi < 70:
            sig = "LONG"
        elif ma20 < ma50 and rsi > 30:
            sig = "SHORT"
        else:
            sig = "HOLD"

        last = float(close.iloc[-1])
        sl = round(last * (0.995 if sig != "SHORT" else 1.005), 8)
        tp1 = round(last * (1.005 if sig != "SHORT" else 0.995), 8)
        confidence = 0.5
        return {
            "symbol": symbol,
            "interval": interval,
            "signal": sig,
            "entry": last,
            "sl": sl,
            "tp1": tp1,
            "confidence": confidence,
            "reasons": ["fallback_simple"]
        }
    except Exception:
        log.exception("fallback signal generation failed")
        return {"error": "engine_failed"}

def generate_signal(symbol: str, interval: str = "1h") -> Dict[str, Any]:
    """
    Unified interface used by bot_runner. Returns a dict with:
    { symbol, interval, signal, entry, sl, tp1, confidence, reasons } or {'error':...}
    """
    try:
        symbol = str(symbol).upper().replace("/", "").replace("-", "")
        # prefer analyze_pair_multi_timeframes for stronger signals
        if analyze_pair_multi_timeframes:
            try:
                # analyze using the requested interval + 1h and 4h context
                tfs = [interval]
                if "1h" not in tfs: tfs.append("1h")
                if "4h" not in tfs: tfs.append("4h")
                res = analyze_pair_multi_timeframes(symbol, timeframes=tfs, exchange="binance")
                if res.get("error"):
                    return {"error": res.get("error")}
                combined = res.get("combined_score", 0.0)
                combined_signal = res.get("combined_signal", "HOLD")
                # prefer the exact interval info if available otherwise 1h
                info = res.get("analysis", {}).get(interval) or res.get("analysis", {}).get("1h") or next(iter(res.get("analysis", {}).values()))
                return {
                    "symbol": symbol,
                    "interval": interval,
                    "signal": info.get("signal", combined_signal),
                    "entry": info.get("close"),
                    "sl": info.get("sl"),
                    "tp1": info.get("tp1"),
                    "confidence": float(res.get("combined_score", 0.0)),
                    "reasons": info.get("reasons", [])
                }
            except Exception:
                log.exception("analyze_pair_multi_timeframes failed, falling back")
                # fall through to fallback
        # fallback:
        return _fallback_simple_signal(symbol, interval)
    except Exception:
        log.exception("generate_signal top-level failed")
        return {"error": "generate_failed"}
