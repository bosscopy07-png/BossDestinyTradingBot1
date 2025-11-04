import os
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

import numpy as np
import pandas as pd
import requests

# ---------------- Logging ----------------
LOG = logging.getLogger("signal_engine")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(ch)

# ---------------- Optional providers ----------------
try:
    from market_providers import fetch_klines_multi
    HAS_MP = True
except Exception:
    fetch_klines_multi = None
    HAS_MP = False

try:
    from ai_client import ai_analysis_text
    HAS_AI = True
except Exception:
    ai_analysis_text = None
    HAS_AI = False

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
DEFAULT_EXCHANGES = ["binance", "bybit", "kucoin", "okx"]

# ---------------- Indicators ----------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
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

# ---------------- Binance fallback ----------------
def _fetch_klines_binance(symbol: str, interval: str, limit: int = 300) -> Optional[pd.DataFrame]:
    try:
        s = requests.Session()
        r = s.get(BINANCE_KLINES, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if not isinstance(raw, list) or len(raw) == 0:
            LOG.debug("Binance returned empty klines for %s %s", symbol, interval)
            return None
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"]
        df = pd.DataFrame(raw, columns=cols)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df[["open_time", "open", "high", "low", "close", "volume"]]
    except requests.exceptions.HTTPError as e:
        LOG.warning("Binance HTTP error for %s %s: %s", symbol, interval, e)
        return None
    except Exception:
        LOG.exception("Binance klines fetch failed for %s %s", symbol, interval)
        return None

# ---------------- Scoring ----------------
def _score_dataframe(df: pd.DataFrame) -> (float, List[str]):
    reasons: List[str] = []
    try:
        if df is None or len(df) < 30:
            return 0.0, ["insufficient_data"]
        closes = df["close"].astype(float)
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

        _, _, macd_hist = macd(closes)
        mh = float(macd_hist.iloc[-1])
        if mh > 0:
            score += 0.08
            reasons.append("macd_pos")
        else:
            score -= 0.03

        r = float(rsi(closes).iloc[-1])
        if r < 30:
            score += 0.06
            reasons.append("rsi_oversold")
        elif r > 70:
            score -= 0.06
            reasons.append("rsi_overbought")
        else:
            score += 0.02

        lookback = min(4, len(closes) - 1)
        ret = (closes.iloc[-1] - closes.iloc[-1 - lookback]) / (closes.iloc[-1 - lookback] + 1e-12)
        if ret > 0.01:
            score += 0.04
            reasons.append("momentum_pos")
        elif ret < -0.01:
            score -= 0.04
            reasons.append("momentum_neg")

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

        score = max(0.0, min(0.98, score))
        return round(score, 4), (reasons or ["none"])
    except Exception:
        LOG.exception("Error scoring dataframe")
        return 0.0, ["error_scoring"]

# ---------------- Analyze one exchange ----------------
def _analyze_on_exchange(symbol: str, interval: str, exchange: str) -> Dict[str, Any]:
    try:
        df = None
        if HAS_MP and fetch_klines_multi:
            try:
                df = fetch_klines_multi(symbol, interval, limit=300, exchange=exchange)
            except Exception:
                LOG.debug("fetch_klines_multi failed for %s on %s", symbol, exchange)
                df = None

        if df is None:
            df = _fetch_klines_binance(symbol, interval, limit=300)

        if df is None or df.empty or "close" not in df or len(df) < 30:
            return {"exchange": exchange, "error": "insufficient_data"}

        df = df.copy()
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)

        score, reasons = _score_dataframe(df)
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2])

        if score >= 0.65:
            signal = "LONG" if last > prev else "SHORT"
        else:
            signal = "HOLD"

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
            "reasons": reasons,
        }
    except Exception:
        LOG.exception("analyze_on_exchange failed for %s %s %s", symbol, interval, exchange)
        return {"exchange": exchange, "error": "analysis_failed"}

# ---------------- Multi-exchange signal ----------------
def generate_signal_multi(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    exchanges: List[str] = None,
    min_confidence_for_signal: float = 0.65,
    use_ai_explain: bool = False,
) -> Dict[str, Any]:
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES

    per: List[Dict[str, Any]] = []
    strong_candidates: List[Dict[str, Any]] = []
    try:
        for ex in exchanges:
            try:
                res = _analyze_on_exchange(symbol, interval, ex)
                per.append(res)
                if isinstance(res, dict) and res.get("confidence") is not None:
                    if res.get("signal") in ("LONG", "SHORT") and res.get("confidence", 0.0) >= min_confidence_for_signal:
                        strong_candidates.append(res)
            except Exception:
                LOG.exception("sub-analysis exception for %s on %s", symbol, ex)
                per.append({"exchange": ex, "error": "exception"})

        votes = {"LONG": 0, "SHORT": 0, "HOLD": 0}
        confs: List[float] = []
        for p in per:
            if not isinstance(p, dict) or p.get("confidence") is None:
                continue
            v = p.get("signal", "HOLD")
            votes[v] = votes.get(v, 0) + 1
            confs.append(float(p.get("confidence", 0.0)))

        best: Optional[Dict[str, Any]] = None
        if strong_candidates:
            best = max(strong_candidates, key=lambda x: x.get("confidence", 0.0))
        else:
            scored = [p for p in per if isinstance(p, dict) and p.get("confidence") is not None]
            if scored:
                best = max(scored, key=lambda x: x.get("confidence", 0.0))

        if best is None:
            return {
                "symbol": symbol.upper(),
                "interval": interval,
                "best": None,
                "per_exchange": per,
                "timestamp": datetime.utcnow().isoformat(),
                "error": "no_data_on_any_exchange",
            }

        avg_conf = float(np.mean(confs)) if confs else float(best.get("confidence", 0.0))
        top_vote = max(votes, key=lambda k: votes[k])
        consensus = votes[top_vote] >= 2 and top_vote in ("LONG", "SHORT")
        consensus_details = {"votes": votes, "avg_confidence": avg_conf}

        out: Dict[str, Any] = {
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
            "consensus": consensus,
            "consensus_details": consensus_details,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if use_ai_explain and HAS_AI:
            try:
                text_for_ai = (
                    f"Symbol {out['symbol']} interval {out['interval']}. Signal {out['signal']} "
                    f"confidence {out['confidence']:.2f}. Reasons: {out['reasons']}. Votes: {votes}."
                )
                ai_text = ai_analysis_text(text_for_ai)
                out["ai_explanation"] = ai_text
            except Exception:
                LOG.exception("AI explanation failed")

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
            "error": "internal_exception",
        }

def generate_signal(symbol: str = "BTCUSDT", interval: str = "1h") -> Dict[str, Any]:
    return generate_signal_multi(symbol, interval, exchanges=DEFAULT_EXCHANGES)

# ---------------- Auto scanner ----------------
_scanner_thread: Optional[threading.Thread] = None
_scanner_stop_evt = threading.Event()
_scanner_lock = threading.Lock()
_send_callback: Optional[Callable[[Dict[str, Any]], None]] = None

def register_send_callback(fn: Callable[[Dict[str, Any]], None]) -> None:
    global _send_callback
    _send_callback = fn
    LOG.info("signal_engine: send callback registered: %s", fn)

def _scanner_loop(pairs: List[str], interval: str, exchanges: List[str], min_confidence: float, poll_seconds: int, use_ai: bool):
    LOG.info("signal_engine: scanner started for %s (interval=%s) poll=%s", pairs, interval, poll_seconds)
    while not _scanner_stop_evt.is_set():
        try:
            for p in pairs:
                res = generate_signal_multi(p, interval, exchanges=exchanges, min_confidence_for_signal=min_confidence, use_ai_explain=use_ai)
                if not isinstance(res, dict):
                    continue
                conf = float(res.get("confidence", 0.0))
                is_strong = (conf >= min_confidence) and (res.get("signal") in ("LONG", "SHORT"))
                consensus = bool(res.get("consensus"))
                if is_strong and (consensus or conf >= min_confidence):
                    LOG.info("signal_engine: strong signal detected %s %s %.3f", p, res.get("signal"), conf)
                    if _send_callback:
                        try:
                            _send_callback(res)
                        except Exception:
                            LOG.exception("send_callback failed")
                    else:
                        LOG.warning("signal_engine: no send_callback registered to deliver signal")
            for _ in range(int(poll_seconds)):
                if _scanner_stop_evt.is_set():
                    break
                time.sleep(1)
        except Exception:
            LOG.exception("scanner loop exception, continuing")
    LOG.info("signal_engine: scanner stopped")

def start_auto_scanner(pairs: List[str], interval: str = "1h", exchanges: List[str] = None, min_confidence: float = 0.85, poll_seconds: int = 60 * 5, use_ai: bool = False) -> bool:
    global _scanner_thread, _scanner_stop_evt
    with _scanner_lock:
        if _scanner_thread and _scanner_thread.is_alive():
            LOG.warning("signal_engine: scanner already running")
            return False
        _scanner_stop_evt.clear()
        exs = exchanges or DEFAULT_EXCHANGES
        _scanner_thread = threading.Thread(target=_scanner_loop, args=(pairs, interval, exs, min_confidence, poll_seconds, use_ai), daemon=True)
        _scanner_thread.start()
        LOG.info("signal_engine: scanner started")
        return True

def stop_auto_scanner(timeout: int = 5) -> bool:
    global _scanner_thread, _scanner_stop_evt
    with _scanner_lock:
        if not _scanner_thread or not _scanner_thread.is_alive():
            return False
        _scanner_stop_evt.set()
        _scanner_thread.join(timeout)
        if _scanner_thread.is_alive():
            LOG.warning("signal_engine: scanner thread did not stop within timeout")
        else:
            LOG.info("signal_engine: scanner stopped cleanly")
        return True

# ---------------- Export ----------------
__all__ = ["generate_signal_multi", "generate_signal", "start_auto_scanner", "stop_auto_scanner", "register_send_callback"]

# ---------------- Self-test ----------------
if __name__ == "__main__":
    print("Signal engine multi-exchange selftest")
    res = generate_signal_multi("BTCUSDT", "1h")
    import json
    print(json.dumps(res, indent=2, default=str))
