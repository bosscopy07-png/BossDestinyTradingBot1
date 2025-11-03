# market_providers.py
import os
import time
import traceback
import json
from io import BytesIO
from typing import List, Tuple, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional ccxt (preferred if available for multi-exchange OHLCV)
try:
    import ccxt
    HAS_CCXT = True
except Exception:
    HAS_CCXT = False

# Use your image util (must provide create_brand_image(lines, chart_img_bytes=None, title=None, subtitle=None))
try:
    from image_utils import create_brand_image
except Exception:
    create_brand_image = None  # fallbacks will be textual only

# QuickChart for lightweight charts (no matplotlib dependency)
QUICKCHART_URL = "https://quickchart.io/chart"

# Exchanges endpoints (public)
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"
BYBIT_KLINES = "https://api.bybit.com/v2/public/kline/list"  # older public; used only if needed
BYBIT_TICKER = "https://api.bybit.com/v2/public/tickers"
KUCOIN_KLINES = "https://api.kucoin.com/api/v1/market/candles"  # needs type param
KUCOIN_TICKER = "https://api.kucoin.com/api/v1/market/allTickers"
OKX_KLINES = "https://www.okx.com/api/v5/market/candles"
OKX_TICKER = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"

# default timeframes for multi-tf analysis
DEFAULT_TFS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# network session helper
def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "DestinyTradingEmpireBot/1.0"})
    proxy = os.getenv("PROXY_URL")
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=frozenset(["GET", "POST"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# -------------------------
# Utility: quickchart sparkline
# -------------------------
def _quickchart_sparkline(values: List[float], width: int = 800, height: int = 240, label: str = "") -> Optional[bytes]:
    """
    Build a simple sparkline PNG using QuickChart. Returns bytes or None.
    """
    try:
        cfg = {
            "type": "line",
            "data": {
                "labels": list(range(len(values))),
                "datasets": [
                    {"label": label, "data": values, "fill": False, "pointRadius": 0}
                ]
            },
            "options": {
                "plugins": {"legend": {"display": False}},
                "elements": {"line": {"tension": 0}},
                "scales": {"x": {"display": False}, "y": {"display": True}}
            }
        }
        params = {"c": json.dumps(cfg), "width": width, "height": height, "devicePixelRatio": 2}
        sess = get_session()
        r = sess.get(QUICKCHART_URL, params=params, timeout=12)
        r.raise_for_status()
        return r.content
    except Exception:
        traceback.print_exc()
        return None

# -------------------------
# Ticker fetchers (fast)
# -------------------------
def fetch_binance_tickers() -> List[Dict[str, Any]]:
    s = get_session()
    r = s.get(BINANCE_TICKER, timeout=8)
    r.raise_for_status()
    return r.json()

def fetch_bybit_tickers() -> List[Dict[str, Any]]:
    s = get_session()
    r = s.get(BYBIT_TICKER, timeout=8)
    r.raise_for_status()
    return r.json().get("result", [])

def fetch_kucoin_tickers() -> List[Dict[str, Any]]:
    s = get_session()
    r = s.get(KUCOIN_TICKER, timeout=8)
    r.raise_for_status()
    return r.json().get("data", {}).get("ticker", [])

def fetch_okx_tickers() -> List[Dict[str, Any]]:
    s = get_session()
    r = s.get(OKX_TICKER, timeout=8)
    r.raise_for_status()
    return r.json().get("data", [])

# -------------------------
# Trending helpers
# -------------------------
def fetch_trending_pairs_text(pairs: Optional[List[str]] = None) -> str:
    """
    Lightweight text listing of trending pairs (from Binance as fallback).
    """
    try:
        if pairs is None:
            pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
        s = get_session()
        r = s.get(BINANCE_TICKER, timeout=8)
        r.raise_for_status()
        tickers = r.json()
        out = []
        for p in pairs:
            t = next((x for x in tickers if x.get("symbol") == p), None)
            if t:
                out.append(f"{p}: {float(t.get('priceChangePercent',0)):.2f}%  vol:{int(float(t.get('quoteVolume',0))):,}")
        if not out:
            return "No trending data."
        return "📈 Trending Pairs (binance):\n" + "\n".join(out)
    except Exception:
        traceback.print_exc()
        return "Failed to fetch trending pairs."

def fetch_trending_pairs_branded(limit: int = 10) -> Tuple[Optional[BytesIO], str]:
    """
    Scan multiple exchanges, collate a top movers list and return (image_buf, caption).
    Uses create_brand_image(lines, chart_img_bytes, title, subtitle) if available.
    """
    rows = []
    try:
        # Binance
        try:
            for item in fetch_binance_tickers():
                sym = item.get("symbol")
                pct = float(item.get("priceChangePercent", 0)) if item.get("priceChangePercent") is not None else 0.0
                vol = float(item.get("quoteVolume", 0)) if item.get("quoteVolume") else 0.0
                rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "Binance"})
        except Exception:
            pass

        # Bybit
        try:
            for item in fetch_bybit_tickers():
                sym = item.get("symbol")
                pct = float(item.get("price_24h_pcnt", 0)) * 100 if item.get("price_24h_pcnt") is not None else 0.0
                vol = float(item.get("quote_volume", 0)) if item.get("quote_volume") else 0.0
                rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "Bybit"})
        except Exception:
            pass

        # KuCoin
        try:
            for item in fetch_kucoin_tickers():
                sym = item.get("symbol").replace("-", "")
                pct = float(item.get("changeRate", 0)) * 100 if item.get("changeRate") else 0.0
                vol = float(item.get("volValue", 0)) if item.get("volValue") else 0.0
                rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "KuCoin"})
        except Exception:
            pass

        # OKX
        try:
            for item in fetch_okx_tickers():
                sym = item.get("instId", "").replace("-", "")
                pct = float(item.get("change24h", 0)) if item.get("change24h") else 0.0
                vol = float(item.get("volCcy24h", 0)) if item.get("volCcy24h") else 0.0
                rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "OKX"})
        except Exception:
            pass

        if not rows:
            return None, "No exchange data available."

        df = pd.DataFrame(rows)
        df = df[df["vol"] > 0]
        df = df.sort_values("change", ascending=False).head(limit)

        lines = [f"🔥 Multi-Exchange Trending Pairs ({len(df)} picks)"]
        for _, r in df.iterrows():
            lines.append(f"{r['symbol']:<12} | {r['change']:+6.2f}% | vol: {int(r['vol']):,} | {r['exchange']}")

        # create a small sparkline from top symbol close prices (best-effort)
        chart_bytes = None
        try:
            top_sym = df.iloc[0]["symbol"]
            kdf = fetch_klines_multi(top_sym, "1h", limit=48)
            if isinstance(kdf, pd.DataFrame) and "close" in kdf:
                closes = kdf["close"].tolist()[-48:]
                chart_bytes = _quickchart_sparkline(closes, label=top_sym)
        except Exception:
            pass

        if create_brand_image:
            buf = create_brand_image(lines, chart_img=chart_bytes)
            return buf, "Top Multi-Exchange Trending Pairs"
        else:
            return None, "\n".join(lines)

    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {e}"

# -------------------------
# Kline fetcher (multi-exchange)
# -------------------------
def _parse_binance_klines(raw) -> pd.DataFrame:
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def fetch_klines_multi(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 300, exchange: str = "binance") -> Optional[pd.DataFrame]:
    """
    Robust multi-exchange klines getter. Primary: Binance public. If ccxt is available, use ccxt for the chosen exchange.
    Returns pandas DataFrame with columns: open_time, open, high, low, close, volume
    """
    try:
        exchange = (exchange or "binance").lower()
        sess = get_session()

        # Binance direct
        if exchange == "binance" or not HAS_CCXT:
            url = BINANCE_KLINES
            r = sess.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
            r.raise_for_status()
            return _parse_binance_klines(r.json())

        # If ccxt is available, use it for bybit/kucoin/okx
        if HAS_CCXT:
            try:
                ex = None
                if exchange == "bybit":
                    ex = ccxt.bybit()
                elif exchange == "kucoin":
                    ex = ccxt.kucoin()
                elif exchange == "okx":
                    ex = ccxt.okx()
                else:
                    ex = ccxt.binance()
                raw = ex.fetch_ohlcv(symbol.upper(), timeframe=interval, limit=limit)
                df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume"])
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                for c in ["open","high","low","close","volume"]:
                    df[c] = df[c].astype(float)
                return df
            except Exception:
                # fallback to Binance REST if ccxt or specific exchange call fails
                pass

        # fallback: binance
        url = BINANCE_KLINES
        r = sess.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
        r.raise_for_status()
        return _parse_binance_klines(r.json())

    except Exception:
        traceback.print_exc()
        return None

# -------------------------
# Indicators
# -------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
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

# -------------------------
# Candle pattern detector
# -------------------------
def detect_candle_pattern_from_row(row: pd.Series) -> Optional[str]:
    """
    Basic single-candle patterns:
     - Hammer (bullish)
     - Shooting star (bearish)
     - Bullish/Bearish Engulfing requires prev row comparison (handled elsewhere)
    """
    try:
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        body = abs(c - o)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        # hammer: small body near top? (bullish)
        if body / (h - l + 1e-12) < 0.35 and (lower_wick > 2 * body):
            return "HAMMER"
        # shooting star: small body near bottom? (bearish)
        if body / (h - l + 1e-12) < 0.35 and (upper_wick > 2 * body):
            return "SHOOTING_STAR"
    except Exception:
        pass
    return None

def detect_engulfing(df: pd.DataFrame) -> Optional[str]:
    """
    Last two candles: detect engulfing
    """
    try:
        if len(df) < 2:
            return None
        prev = df.iloc[-2]
        last = df.iloc[-1]
        prev_body = abs(prev["close"] - prev["open"])
        last_body = abs(last["close"] - last["open"])
        # bullish engulfing: prev down, last up and last body engulfs prev body
        if prev["close"] < prev["open"] and last["close"] > last["open"] and last_body > prev_body and last["close"] > prev["open"]:
            return "BULL_ENGULFING"
        if prev["close"] > prev["open"] and last["close"] < last["open"] and last_body > prev_body and last["close"] < prev["open"]:
            return "BEAR_ENGULFING"
    except Exception:
        pass
    return None

# -------------------------
# Momentum + pattern scoring
# -------------------------
def score_momentum_and_candle(df: pd.DataFrame) -> Tuple[float, List[str]]:
    """
    Score based on:
     - EMA slope (fast vs slow)
     - RSI extremes
     - MACD hist sign
     - ATR (volatility)
     - Candle pattern (hammer, engulfing, shooting star)
    Returns (confidence [0..1], reasons list)
    """
    reasons = []
    try:
        if df is None or len(df) < 30:
            return 0.0, ["insufficient_data"]
        closes = df["close"]
        # EMAs
        ema_fast = ema(closes, 9).iloc[-1]
        ema_slow = ema(closes, 21).iloc[-1]
        prev_ema_fast = ema(closes, 9).iloc[-2]
        prev_ema_slow = ema(closes, 21).iloc[-2]
        ema_signal = 0.0
        if ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow:
            ema_signal += 0.25
            reasons.append("ema_cross_up")
        elif ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow:
            ema_signal -= 0.25
            reasons.append("ema_cross_down")

        # RSI
        r = float(rsi(closes).iloc[-1])
        if r < 30:
            reasons.append("rsi_oversold")
            r_score = 0.15
        elif r > 70:
            reasons.append("rsi_overbought")
            r_score = -0.12
        else:
            r_score = 0.05

        # MACD
        macd_val, macd_sig, macd_hist = macd(closes)
        mh = macd_hist.iloc[-1]
        if mh > 0:
            reasons.append("macd_pos")
            macd_score = 0.12
        else:
            macd_score = -0.03

        # ATR - volatility filter (higher ATR => conservative)
        atr_val = float(atr(df).iloc[-1])
        if atr_val is None or np.isnan(atr_val):
            atr_score = 0.0
        else:
            # low volatility adds some confidence to trend signals
            atr_score = 0.05 if atr_val < (closes.std()) else -0.01

        # Candle patterns
        cand_pat = detect_engulfing(df)
        single_pat = detect_candle_pattern_from_row(df.iloc[-1])
        if cand_pat:
            if "BULL" in cand_pat:
                reasons.append(cand_pat.lower())
                pat_score = 0.25
            else:
                reasons.append(cand_pat.lower())
                pat_score = -0.2
        elif single_pat:
            if single_pat == "HAMMER":
                reasons.append("hammer")
                pat_score = 0.18
            elif single_pat == "SHOOTING_STAR":
                reasons.append("shooting_star")
                pat_score = -0.18
            else:
                pat_score = 0.0
        else:
            pat_score = 0.0

        # Momentum (recent returns)
        ret_3 = (closes.iloc[-1] - closes.iloc[-4]) / (closes.iloc[-4] + 1e-12)
        if ret_3 > 0.01:
            momentum = 0.08
            reasons.append("short_momentum_pos")
        elif ret_3 < -0.01:
            momentum = -0.08
            reasons.append("short_momentum_neg")
        else:
            momentum = 0.0

        # Aggregate
        score = 0.5 + ema_signal + r_score + macd_score + atr_score + pat_score + momentum
        # clamp 0..1
        score = max(0.0, min(1.0, score))
        return round(score, 4), reasons
    except Exception:
        traceback.print_exc()
        return 0.0, ["error_scoring"]

# -------------------------
# Multi-timeframe analysis orchestration
# -------------------------
def analyze_pair_multi_timeframes(symbol: str, timeframes: Optional[List[str]] = None, exchange: str = "binance") -> Dict[str, Any]:
    """
    Analyze a symbol across multiple timeframes; returns a structured dict:
    {
        "symbol": symbol,
        "analysis": {
            tf: { "score": 0..1, "signal": "LONG/SHORT/HOLD", "reasons": [...], "close": val, "sl": val, "tp1": val }
        },
        "combined_score": 0..1,
        "combined_signal": "STRONG_LONG"/"STRONG_SHORT"/"HOLD",
        "prices": {...}
    }
    """
    if timeframes is None:
        timeframes = DEFAULT_TFS

    results = {}
    scores = []
    closes = {}
    try:
        for tf in timeframes:
            df = fetch_klines_multi(symbol, tf, limit=200, exchange=exchange)
            if df is None or df.empty or "close" not in df:
                results[tf] = {"error": "no_data"}
                continue
            # ensure numeric
            df = df.copy()
            df["close"] = df["close"].astype(float)
            score, reasons = score_momentum_and_candle(df)
            # determine directional preference
            last = df.iloc[-1]["close"]
            prev = df.iloc[-2]["close"]
            if score > 0.55:
                # decide long or short by net momentum & candle sentiment
                sdir = "LONG" if last > prev else "SHORT"
            else:
                sdir = "HOLD"
            # compute simple SL/TP using ATR
            atr_val = float(atr(df).iloc[-1]) if len(df) > 15 else (max(df["high"])-min(df["low"]))*0.005
            if sdir == "LONG":
                sl = last - 1.5 * atr_val
                tp1 = last + 1.5 * atr_val
            elif sdir == "SHORT":
                sl = last + 1.5 * atr_val
                tp1 = last - 1.5 * atr_val
            else:
                sl = last * 0.995
                tp1 = last * 1.005
            results[tf] = {
                "score": round(score, 3),
                "signal": sdir,
                "reasons": reasons,
                "close": float(last),
                "sl": float(round(sl, 8)),
                "tp1": float(round(tp1, 8))
            }
            scores.append(score)
            closes[tf] = float(last)

        # combine scores across TFs: weighted (higher TFs more weight)
        weights = []
        # assign weights proportional to timeframe importance
        tf_weights_map = {"1m": 0.5, "5m": 0.7, "15m": 0.9, "30m": 1.0, "1h": 1.4, "4h": 1.6, "1d": 2.0}
        total_weight = 0.0
        sum_prod = 0.0
        for tf in timeframes:
            w = tf_weights_map.get(tf, 1.0)
            total_weight += w
            s = results.get(tf, {}).get("score", 0.0)
            sum_prod += s * w

        combined_score = sum_prod / (total_weight + 1e-12) if total_weight > 0 else 0.0
        # direction majority
        long_count = sum(1 for tf in timeframes if results.get(tf, {}).get("signal") == "LONG")
        short_count = sum(1 for tf in timeframes if results.get(tf, {}).get("signal") == "SHORT")

        if combined_score >= 0.9 and long_count > short_count:
            combined_signal = "STRONG_LONG"
        elif combined_score >= 0.9 and short_count > long_count:
            combined_signal = "STRONG_SHORT"
        elif combined_score >= 0.65 and long_count > short_count:
            combined_signal = "LONG"
        elif combined_score >= 0.65 and short_count > long_count:
            combined_signal = "SHORT"
        else:
            combined_signal = "HOLD"

        return {
            "symbol": symbol.upper(),
            "exchange": exchange,
            "analysis": results,
            "combined_score": round(combined_score, 4),
            "combined_signal": combined_signal,
            "prices": closes
        }

    except Exception:
        traceback.print_exc()
        return {"error": "analysis_failed"}

# -------------------------
# Detect strong signals among many pairs
# -------------------------
def detect_strong_signals(pairs: Optional[List[str]] = None,
                          timeframes: Optional[List[str]] = None,
                          exchange: str = "binance",
                          min_confidence: float = 0.90) -> List[Dict[str, Any]]:
    """
    Scan a list of pairs and return candidates with combined_score >= min_confidence.
    Returns list of dicts with symbol, combined_score, combined_signal, sl, tp1, reasons, image (BytesIO or None)
    """
    if pairs is None:
        pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT").split(",")
    if timeframes is None:
        timeframes = DEFAULT_TFS

    out = []
    for p in pairs:
        try:
            res = analyze_pair_multi_timeframes(p, timeframes, exchange=exchange)
            if not isinstance(res, dict) or res.get("error"):
                continue
            score = float(res.get("combined_score", 0.0))
            if score >= min_confidence and res.get("combined_signal") in ("STRONG_LONG", "STRONG_SHORT"):
                # prepare image summary
                closes = []
                # pick intermediate tf (1h) if available else largest available
                try:
                    kdf = fetch_klines_multi(p, "1h", limit=80, exchange=exchange)
                    if isinstance(kdf, pd.DataFrame) and "close" in kdf:
                        closes = kdf["close"].tolist()[-60:]
                except Exception:
                    pass
                chart_bytes = _quickchart_sparkline(closes, label=p) if closes else None
                # build message lines
                lines = [
                    f"🔥 {res['combined_signal']} detected",
                    f"Pair: {p} ({exchange})",
                    f"Score: {score:.3f}  Signal: {res['combined_signal']}",
                ]
                # attach tf breakdown
                for tf, tinfo in res.get("analysis", {}).items():
                    if isinstance(tinfo, dict) and "score" in tinfo:
                        lines.append(f"{tf}: {tinfo['signal']} ({tinfo['score']*100:.0f}%)")
                # SL/TP from 1h analysis if present else pick any
                try:
                    oneh = res["analysis"].get("1h") or next(iter(res["analysis"].values()))
                    sl = oneh.get("sl")
                    tp1 = oneh.get("tp1")
                except Exception:
                    sl, tp1 = None, None

                # build image via create_brand_image if available
                img_buf = None
                try:
                    if create_brand_image:
                        img_buf = create_brand_image(lines, chart_img=chart_bytes)
                except Exception:
                    traceback.print_exc()
                    img_buf = None

                out.append({
                    "symbol": p.upper(),
                    "exchange": exchange,
                    "combined_score": round(score, 4),
                    "combined_signal": res.get("combined_signal"),
                    "sl": sl,
                    "tp1": tp1,
                    "analysis": res,
                    "image": img_buf,
                    "caption_lines": lines
                })
        except Exception:
            traceback.print_exc()
            continue
    # sort descending by confidence
    out = sorted(out, key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return out

# -------------------------
# Helper: generate branded signal image for a single signal dict
# -------------------------
def generate_branded_signal_image(signal_item: Dict[str, Any]) -> Tuple[Optional[BytesIO], str]:
    """
    Build an image + caption for a signal item returned from detect_strong_signals or analyze_pair_multi_timeframes.
    Returns (img_buf or None, caption_text)
    """
    try:
        symbol = signal_item.get("symbol")
        caption_lines = signal_item.get("caption_lines") or []
        # add SL/TP lines
        if signal_item.get("sl") is not None and signal_item.get("tp1") is not None:
            caption_lines.append(f"Entry: {signal_item['analysis'].get('1h',{}).get('close', 'N/A')}")
            caption_lines.append(f"SL: {signal_item['sl']}  TP1: {signal_item['tp1']}")
        caption_text = "\n".join(caption_lines)
        img_buf = signal_item.get("image")
        # fallback: if img_buf None, create brand image with caption lines only
        if img_buf is None and create_brand_image:
            img_buf = create_brand_image(caption_lines)
        return img_buf, caption_text
    except Exception:
        traceback.print_exc()
        return None, "Signal"

# -------------------------
# Quick test when run directly
# -------------------------
if __name__ == "__main__":
    print("Market providers self test")
    try:
        pairs = ["BTCUSDT", "ETHUSDT"]
        strong = detect_strong_signals(pairs=pairs, timeframes=["15m", "1h", "4h"], min_confidence=0.75)
        print("Found strong signals:", len(strong))
        for s in strong:
            print(s["symbol"], s["combined_score"], s["combined_signal"])
    except Exception as e:
        print("Selftest failed:", e)
        
