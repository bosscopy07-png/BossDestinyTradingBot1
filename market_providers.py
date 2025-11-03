# market_providers.py
"""
Unified multi-exchange market data provider used by bot_runner.py.

Features:
- Multi-exchange klines via ccxt (preferred) or REST fallbacks (Binance, Bybit, KuCoin, OKX).
- Tick data / trending helpers.
- Multi-timeframe analysis orchestrator and strong-signal detector.
- Image creation via `create_brand_image` when available (image_utils.py).
"""

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
import logging

logging.getLogger("urllib3").setLevel(logging.WARNING)

# optional ccxt for robust multi-exchange support
try:
    import ccxt
    HAS_CCXT = True
except Exception:
    ccxt = None
    HAS_CCXT = False

# optional image util
try:
    from image_utils import create_brand_image
except Exception:
    create_brand_image = None

# QuickChart for sparkline fallback
QUICKCHART_URL = "https://quickchart.io/chart"

# Public REST endpoints
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"
BYBIT_TICKER = "https://api.bybit.com/v2/public/tickers"
BYBIT_KLINES_V5 = "https://api.bybit.com/v5/market/kline"
KUCOIN_KLINES = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_TICKER = "https://api.kucoin.com/api/v1/market/allTickers"
OKX_KLINES = "https://www.okx.com/api/v5/market/candles"
OKX_TICKER = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"

# default timeframes
DEFAULT_TFS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# session helper
def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "DestinyTradingEmpireBot/1.0"})
    proxy = os.getenv("PROXY_URL")
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    retries = Retry(total=3, backoff_factor=0.6,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=frozenset(["GET", "POST"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# ---------- QuickChart sparkline ----------
def _quickchart_sparkline(values: List[float], width: int = 800, height: int = 240, label: str = "") -> Optional[bytes]:
    try:
        cfg = {
            "type": "line",
            "data": {
                "labels": list(range(len(values))),
                "datasets": [{"label": label, "data": values, "fill": False, "pointRadius": 0}]
            },
            "options": {"plugins": {"legend": {"display": False}}, "elements": {"line": {"tension": 0}}}
        }
        params = {"c": json.dumps(cfg), "width": width, "height": height, "devicePixelRatio": 2}
        r = get_session().get(QUICKCHART_URL, params=params, timeout=12)
        r.raise_for_status()
        return r.content
    except Exception:
        traceback.print_exc()
        return None

# ---------- Ticker fetchers ----------
def fetch_binance_tickers() -> List[Dict[str, Any]]:
    sess = get_session()
    r = sess.get(BINANCE_TICKER, timeout=8)
    r.raise_for_status()
    return r.json()

def fetch_bybit_tickers() -> List[Dict[str, Any]]:
    sess = get_session()
    r = sess.get(BYBIT_TICKER, timeout=8)
    r.raise_for_status()
    # bybit v2 returns list
    return r.json()

def fetch_kucoin_tickers() -> List[Dict[str, Any]]:
    sess = get_session()
    r = sess.get(KUCOIN_TICKER, timeout=8)
    r.raise_for_status()
    return r.json().get("data", {}).get("ticker", [])

def fetch_okx_tickers() -> List[Dict[str, Any]]:
    sess = get_session()
    r = sess.get(OKX_TICKER, timeout=8)
    r.raise_for_status()
    return r.json().get("data", [])

# ---------- Trending helpers ----------
def fetch_trending_pairs_text(pairs: Optional[List[str]] = None) -> str:
    try:
        if pairs is None:
            pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
        tickers = fetch_binance_tickers()
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
    rows = []
    try:
        # gather from multiple exchanges (best-effort)
        try:
            for item in fetch_binance_tickers():
                sym = item.get("symbol")
                pct = float(item.get("priceChangePercent", 0) or 0)
                vol = float(item.get("quoteVolume", 0) or 0)
                rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "Binance"})
        except Exception:
            logging.debug("binance tickers failed", exc_info=True)

        try:
            by = fetch_bybit_tickers()
            # bybit returns dict/list forms; try common shapes
            if isinstance(by, dict) and "result" in by:
                cand = by.get("result", [])
            elif isinstance(by, list):
                cand = by
            else:
                cand = []
            for item in cand:
                # adapt possible key names
                sym = item.get("symbol") or item.get("symbolName") or item.get("s")
                pct = float(item.get("price_24h_pcnt", 0) or 0) * (100 if abs(float(item.get("price_24h_pcnt", 0) or 0)) < 2 else 1)
                vol = float(item.get("quote_volume", 0) or 0)
                if sym:
                    rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "Bybit"})
        except Exception:
            logging.debug("bybit tickers failed", exc_info=True)

        try:
            for item in fetch_kucoin_tickers():
                sym = item.get("symbol") if isinstance(item, dict) else None
                if not sym:
                    continue
                pct = float(item.get("changeRate", 0) or 0) * 100 if item.get("changeRate") else 0.0
                vol = float(item.get("volValue", 0) or 0)
                rows.append({"symbol": sym.replace("-", ""), "change": pct, "vol": vol, "exchange": "KuCoin"})
        except Exception:
            logging.debug("kucoin tickers failed", exc_info=True)

        try:
            for item in fetch_okx_tickers():
                sym = item.get("instId", "")
                pct = float(item.get("change24h", 0) or 0)
                vol = float(item.get("volCcy24h", 0) or 0)
                rows.append({"symbol": sym.replace("-", ""), "change": pct, "vol": vol, "exchange": "OKX"})
        except Exception:
            logging.debug("okx tickers failed", exc_info=True)

        if not rows:
            return None, "No exchange data available."

        df = pd.DataFrame(rows)
        df = df[df["vol"] > 0]
        df = df.sort_values("change", ascending=False).head(limit)

        lines = [f"🔥 Multi-Exchange Trending Pairs ({len(df)} picks)"]
        for _, r in df.iterrows():
            lines.append(f"{r['symbol']:<12} | {r['change']:+6.2f}% | vol: {int(r['vol']):,} | {r['exchange']}")

        chart_bytes = None
        try:
            top_sym = df.iloc[0]["symbol"]
            kdf = fetch_klines_multi(top_sym, "1h", limit=48)
            if isinstance(kdf, pd.DataFrame) and "close" in kdf:
                chart_bytes = _quickchart_sparkline(kdf["close"].tolist()[-48:], label=top_sym)
        except Exception:
            logging.debug("sparkline failed", exc_info=True)

        if create_brand_image:
            buf = create_brand_image(lines, chart_img=chart_bytes)
            return buf, "Top Multi-Exchange Trending Pairs"
        else:
            return None, "\n".join(lines)
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {e}"

# ---------- Kline parsing helpers ----------
def _parse_binance_klines(raw) -> Optional[pd.DataFrame]:
    try:
        cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(raw, columns=cols)
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df[["open_time","open","high","low","close","volume"]]
    except Exception:
        traceback.print_exc()
        return None

def _parse_generic_ohlcv_list(arr) -> Optional[pd.DataFrame]:
    try:
        # common list-of-lists: [ts, open, high, low, close, volume]
        if not arr or not isinstance(arr, list):
            return None
        # detect length: if each element is list-like and len >=6
        if isinstance(arr[0], list) and len(arr[0]) >= 6:
            df = pd.DataFrame(arr)
            df = df.iloc[:, :6]
            df.columns = ["open_time","open","high","low","close","volume"]
            # detect timestamp unit: seconds vs ms
            ts = float(df["open_time"].iloc[0])
            unit = "ms" if ts > 1e12 else "s"
            df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit=unit)
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            return df[["open_time","open","high","low","close","volume"]]
        return None
    except Exception:
        traceback.print_exc()
        return None

# ---------- Normalize symbol mapping ----------
def _normalize_for_exchange(symbol: str, exchange: str) -> str:
    s = symbol.upper().replace("/", "").replace("-", "")
    if exchange in ("okx", "kucoin"):
        if s.endswith("USDT"):
            return s[:-4] + "-USDT"
        if s.endswith("USD"):
            return s[:-3] + "-USD"
    return s

def _kucoin_interval_map(interval: str) -> str:
    m = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1hour","4h":"4hour","1d":"1day"}
    return m.get(interval, interval)

# ---------- Fetch klines multi-exchange ----------
def fetch_klines_multi(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 300, exchange: str = "binance") -> Optional[pd.DataFrame]:
    try:
        exchange = (exchange or "binance").lower()
        sess = get_session()

        # 1) try ccxt (preferred)
        if HAS_CCXT:
            try:
                ccxt_map = {"okx":"okx", "kucoin":"kucoin", "bybit":"bybit", "binance":"binance"}
                ccxt_id = ccxt_map.get(exchange, exchange)
                ex = getattr(ccxt, ccxt_id)()
                ex.load_markets()
                # ccxt expects "BTC/USDT"
                cc_sym = symbol.replace("USDT", "/USDT").replace("USD", "/USD")
                ohlcv = ex.fetch_ohlcv(cc_sym, timeframe=interval, limit=int(limit))
                if ohlcv and isinstance(ohlcv, list):
                    df = pd.DataFrame(ohlcv, columns=["open_time","open","high","low","close","volume"])
                    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                    for c in ["open","high","low","close","volume"]:
                        df[c] = df[c].astype(float)
                    return df[["open_time","open","high","low","close","volume"]]
            except Exception:
                logging.debug("ccxt fetch failed", exc_info=True)

        # 2) exchange-specific REST fallbacks
        if exchange == "binance":
            url = BINANCE_KLINES
            r = sess.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
            r.raise_for_status()
            return _parse_binance_klines(r.json())

        if exchange == "bybit":
            url = BYBIT_KLINES_V5
            params = {"category": "spot", "symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
            r = sess.get(url, params=params, timeout=10)
            r.raise_for_status()
            j = r.json()
            arr = j.get("result", {}).get("list") or j.get("result")
            df = _parse_generic_ohlcv_list(arr)
            if df is not None:
                return df

        if exchange == "kucoin":
            ku_sym = _normalize_for_exchange(symbol, "kucoin")
            params = {"symbol": ku_sym, "type": _kucoin_interval_map(interval)}
            r = sess.get(KUCOIN_KLINES, params=params, timeout=10)
            r.raise_for_status()
            j = r.json()
            arr = j.get("data")
            df = _parse_generic_ohlcv_list(arr)
            if df is not None:
                return df

        if exchange == "okx":
            ok_sym = _normalize_for_exchange(symbol, "okx")
            params = {"instId": ok_sym, "bar": interval, "limit": int(limit)}
            r = sess.get(OKX_KLINES, params=params, timeout=10)
            r.raise_for_status()
            j = r.json()
            arr = j.get("data")
            df = _parse_generic_ohlcv_list(arr)
            if df is not None:
                return df

        # last resort: try Binance again
        try:
            url = BINANCE_KLINES
            r = sess.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
            r.raise_for_status()
            return _parse_binance_klines(r.json())
        except Exception:
            logging.debug("final binance fallback failed", exc_info=True)

    except Exception:
        traceback.print_exc()

    return None

# ---------- Indicators & scoring (kept similar to previous) ----------
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

# candle pattern helpers
def detect_candle_pattern_from_row(row: pd.Series) -> Optional[str]:
    try:
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        body = abs(c - o)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        if body / (h - l + 1e-12) < 0.35 and (lower_wick > 2 * body):
            return "HAMMER"
        if body / (h - l + 1e-12) < 0.35 and (upper_wick > 2 * body):
            return "SHOOTING_STAR"
    except Exception:
        pass
    return None

def detect_engulfing(df: pd.DataFrame) -> Optional[str]:
    try:
        if len(df) < 2:
            return None
        prev = df.iloc[-2]
        last = df.iloc[-1]
        prev_body = abs(prev["close"] - prev["open"])
        last_body = abs(last["close"] - last["open"])
        if prev["close"] < prev["open"] and last["close"] > last["open"] and last_body > prev_body and last["close"] > prev["open"]:
            return "BULL_ENGULFING"
        if prev["close"] > prev["open"] and last["close"] < last["open"] and last_body > prev_body and last["close"] < prev["open"]:
            return "BEAR_ENGULFING"
    except Exception:
        pass
    return None

# scoring function
def score_momentum_and_candle(df: pd.DataFrame) -> Tuple[float, List[str]]:
    reasons = []
    try:
        if df is None or len(df) < 30:
            return 0.0, ["insufficient_data"]
        closes = df["close"]
        ema_fast = ema(closes, 9).iloc[-1]
        ema_slow = ema(closes, 21).iloc[-1]
        prev_ema_fast = ema(closes, 9).iloc[-2]
        prev_ema_slow = ema(closes, 21).iloc[-2]
        ema_signal = 0.0
        if ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow:
            ema_signal += 0.25; reasons.append("ema_cross_up")
        elif ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow:
            ema_signal -= 0.25; reasons.append("ema_cross_down")

        r = float(rsi(closes).iloc[-1])
        if r < 30:
            reasons.append("rsi_oversold"); r_score = 0.15
        elif r > 70:
            reasons.append("rsi_overbought"); r_score = -0.12
        else:
            r_score = 0.05

        macd_val, macd_sig, macd_hist = macd(closes)
        mh = macd_hist.iloc[-1]
        if mh > 0:
            reasons.append("macd_pos"); macd_score = 0.12
        else:
            macd_score = -0.03

        atr_val = float(atr(df).iloc[-1]) if len(df) > 15 else (max(df["high"])-min(df["low"]))*0.005
        atr_score = 0.05 if atr_val < (closes.std()) else -0.01

        cand_pat = detect_engulfing(df)
        single_pat = detect_candle_pattern_from_row(df.iloc[-1])
        if cand_pat:
            if "BULL" in cand_pat:
                reasons.append(cand_pat.lower()); pat_score = 0.25
            else:
                reasons.append(cand_pat.lower()); pat_score = -0.2
        elif single_pat:
            if single_pat == "HAMMER":
                reasons.append("hammer"); pat_score = 0.18
            elif single_pat == "SHOOTING_STAR":
                reasons.append("shooting_star"); pat_score = -0.18
            else:
                pat_score = 0.0
        else:
            pat_score = 0.0

        ret_3 = (closes.iloc[-1] - closes.iloc[-4]) / (closes.iloc[-4] + 1e-12)
        if ret_3 > 0.01:
            momentum = 0.08; reasons.append("short_momentum_pos")
        elif ret_3 < -0.01:
            momentum = -0.08; reasons.append("short_momentum_neg")
        else:
            momentum = 0.0

        score = 0.5 + ema_signal + r_score + macd_score + atr_score + pat_score + momentum
        score = max(0.0, min(1.0, score))
        return round(score, 4), reasons
    except Exception:
        traceback.print_exc()
        return 0.0, ["error_scoring"]

# ---------- Adaptive timeframe helper ----------
def get_timeframes(adapt: bool = False, volatility: Optional[float] = None) -> List[str]:
    """
    If adapt==False returns DEFAULT_TFS.
    If adapt==True, chooses a smaller or larger set depending on volatility:
      - high volatility -> include shorter TFs
      - low volatility -> focus on higher TFs
    """
    if not adapt:
        return DEFAULT_TFS
    # simple heuristic: volatility > threshold -> include 1m,5m
    try:
        if volatility is None:
            # fallback: include full set
            return DEFAULT_TFS
        if volatility > 0.02:
            return ["1m", "5m", "15m", "30m", "1h"]
        elif volatility > 0.005:
            return ["5m", "15m", "30m", "1h", "4h"]
        else:
            return ["15m", "30m", "1h", "4h", "1d"]
    except Exception:
        return DEFAULT_TFS

# ---------- Multi-timeframe analysis ----------
def analyze_pair_multi_timeframes(symbol: str, timeframes: Optional[List[str]] = None, exchange: str = "binance", adapt_timeframes: bool = False) -> Dict[str, Any]:
    if timeframes is None:
        timeframes = DEFAULT_TFS
    if adapt_timeframes:
        # estimate short-term volatility from 1h binance data as heuristic
        try:
            base_df = fetch_klines_multi(symbol, "1h", limit=48, exchange="binance")
            vol = None
            if isinstance(base_df, pd.DataFrame):
                vol = base_df["close"].pct_change().std()
            timeframes = get_timeframes(adapt=True, volatility=vol)
        except Exception:
            timeframes = DEFAULT_TFS

    results = {}
    closes = {}
    try:
        tf_weights_map = {"1m": 0.5, "5m": 0.7, "15m": 0.9, "30m": 1.0, "1h": 1.4, "4h": 1.6, "1d": 2.0}
        total_weight = 0.0
        sum_prod = 0.0

        for tf in timeframes:
            df = fetch_klines_multi(symbol, tf, limit=200, exchange=exchange)
            if df is None or df.empty or "close" not in df:
                results[tf] = {"error": "no_data"}
                continue
            df = df.copy()
            df["close"] = df["close"].astype(float)
            score, reasons = score_momentum_and_candle(df)
            last = df.iloc[-1]["close"]
            prev = df.iloc[-2]["close"]
            if score > 0.55:
                sdir = "LONG" if last > prev else "SHORT"
            else:
                sdir = "HOLD"
            atr_val = float(atr(df).iloc[-1]) if len(df) > 15 else max(df["high"]) - min(df["low"])
            if sdir == "LONG":
                sl = last - 1.5 * atr_val
                tp1 = last + 1.5 * atr_val
            elif sdir == "SHORT":
                sl = last + 1.5 * atr_val
                tp1 = last - 1.5 * atr_val
            else:
                sl = last * 0.995
                tp1 = last * 1.005

            results[tf] = {"score": round(score, 3), "signal": sdir, "reasons": reasons,
                           "close": float(last), "sl": float(round(sl, 8)), "tp1": float(round(tp1, 8))}
            weight = tf_weights_map.get(tf, 1.0)
            total_weight += weight
            sum_prod += results[tf]["score"] * weight
            closes[tf] = float(last)

        combined_score = sum_prod / (total_weight + 1e-12) if total_weight > 0 else 0.0
        long_count = sum(1 for tf in results if results.get(tf, {}).get("signal") == "LONG")
        short_count = sum(1 for tf in results if results.get(tf, {}).get("signal") == "SHORT")

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

        return {"symbol": symbol.upper(), "exchange": exchange, "analysis": results,
                "combined_score": round(combined_score, 4), "combined_signal": combined_signal, "prices": closes}
    except Exception:
        traceback.print_exc()
        return {"error": "analysis_failed"}

# ---------- Detect strong signals among pairs ----------
def detect_strong_signals(pairs: Optional[List[str]] = None,
                          timeframes: Optional[List[str]] = None,
                          exchange: str = "binance",
                          min_confidence: float = 0.90,
                          adapt_timeframes: bool = False) -> List[Dict[str, Any]]:
    if pairs is None:
        pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT").split(",")
    if timeframes is None:
        timeframes = DEFAULT_TFS

    out = []
    for p in pairs:
        try:
            res = analyze_pair_multi_timeframes(p, timeframes, exchange=exchange, adapt_timeframes=adapt_timeframes)
            if not isinstance(res, dict) or res.get("error"):
                continue
            score = float(res.get("combined_score", 0.0))
            if score >= min_confidence and res.get("combined_signal") in ("STRONG_LONG", "STRONG_SHORT"):
                closes = []
                try:
                    kdf = fetch_klines_multi(p, "1h", limit=80, exchange=exchange)
                    if isinstance(kdf, pd.DataFrame) and "close" in kdf:
                        closes = kdf["close"].tolist()[-60:]
                except Exception:
                    pass
                chart_bytes = _quickchart_sparkline(closes, label=p) if closes else None
                lines = [f"🔥 {res['combined_signal']} detected", f"Pair: {p} ({exchange})", f"Score: {score:.3f}  Signal: {res['combined_signal']}"]
                for tf, tinfo in res.get("analysis", {}).items():
                    if isinstance(tinfo, dict) and "score" in tinfo:
                        lines.append(f"{tf}: {tinfo['signal']} ({tinfo['score']*100:.0f}%)")
                try:
                    oneh = res["analysis"].get("1h") or next(iter(res["analysis"].values()))
                    sl = oneh.get("sl")
                    tp1 = oneh.get("tp1")
                except Exception:
                    sl, tp1 = None, None
                img_buf = None
                try:
                    if create_brand_image:
                        img_buf = create_brand_image(lines, chart_img=chart_bytes)
                except Exception:
                    traceback.print_exc()
                    img_buf = None
                out.append({"symbol": p.upper(), "exchange": exchange, "combined_score": round(score, 4),
                            "combined_signal": res.get("combined_signal"), "sl": sl, "tp1": tp1,
                            "analysis": res, "image": img_buf, "caption_lines": lines})
        except Exception:
            traceback.print_exc()
            continue

    out = sorted(out, key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return out

# ---------- Helper: generate image + caption ----------
def generate_branded_signal_image(signal_item: Dict[str, Any]) -> Tuple[Optional[BytesIO], str]:
    try:
        caption_lines = signal_item.get("caption_lines") or []
        if signal_item.get("sl") is not None and signal_item.get("tp1") is not None:
            entry = signal_item.get("analysis", {}).get("1h", {}).get("close", "N/A")
            caption_lines.append(f"Entry: {entry}")
            caption_lines.append(f"SL: {signal_item['sl']}  TP1: {signal_item['tp1']}")
        caption_text = "\n".join(caption_lines)
        img_buf = signal_item.get("image")
        if img_buf is None and create_brand_image:
            img_buf = create_brand_image(caption_lines)
        return img_buf, caption_text
    except Exception:
        traceback.print_exc()
        return None, "Signal"

# ---------- Self-test ----------
if __name__ == "__main__":
    print("Market providers quick self-test")
    try:
        pairs = ["BTCUSDT", "ETHUSDT"]
        for p in pairs:
            d = fetch_klines_multi(p, "1h", limit=50)
            print(p, "ok" if isinstance(d, pd.DataFrame) else "no_data")
        strong = detect_strong_signals(pairs=pairs, timeframes=["15m", "1h", "4h"], min_confidence=0.75)
        print("Found strong signals:", len(strong))
        for s in strong:
            print(s["symbol"], s["combined_score"], s["combined_signal"])
    except Exception as e:
        print("Selftest failed:", e)
        
