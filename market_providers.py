# market_providers.py
"""
Unified multi-exchange market data provider for trading bots.

Features:
- Multi-exchange klines via ccxt (preferred) or REST fallbacks (Binance, Bybit, KuCoin, OKX).
- Ticker fetching & trending pairs helpers.
- Multi-timeframe analysis and strong-signal scoring.
- QuickChart sparkline fallback and optional branded images.
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
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger("market_providers")
logger.setLevel(logging.INFO)

# Optional ccxt for multi-exchange support
try:
    import ccxt
    HAS_CCXT = True
except Exception:
    ccxt = None
    HAS_CCXT = False

# Optional branding image util
try:
    from image_utils import create_brand_image
except Exception:
    create_brand_image = None

# QuickChart fallback
QUICKCHART_URL = "https://quickchart.io/chart"

# REST endpoints
EXCHANGE_ENDPOINTS = {
    "binance_klines": "https://api.binance.com/api/v3/klines",
    "binance_ticker": "https://api.binance.com/api/v3/ticker/24hr",
    "bybit_ticker": "https://api.bybit.com/v2/public/tickers",
    "bybit_klines": "https://api.bybit.com/v5/market/kline",
    "kucoin_klines": "https://api.kucoin.com/api/v1/market/candles",
    "kucoin_ticker": "https://api.kucoin.com/api/v1/market/allTickers",
    "okx_klines": "https://www.okx.com/api/v5/market/candles",
    "okx_ticker": "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
}

# Default timeframes
DEFAULT_TFS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# ------------------------------
# HTTP session helper
# ------------------------------
def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "DestinyTradingEmpireBot/1.0"})
    proxy = os.getenv("PROXY_URL")
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# ------------------------------
# QuickChart sparkline
# ------------------------------
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
        logger.exception("QuickChart sparkline generation failed")
        return None

# ------------------------------
# Ticker fetchers
# ------------------------------
def fetch_binance_tickers() -> List[Dict[str, Any]]:
    try:
        r = get_session().get(EXCHANGE_ENDPOINTS["binance_ticker"], timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        logger.exception("Binance ticker fetch failed")
        return []

def fetch_bybit_tickers() -> List[Dict[str, Any]]:
    try:
        r = get_session().get(EXCHANGE_ENDPOINTS["bybit_ticker"], timeout=8)
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else r.json().get("result", [])
    except Exception:
        logger.exception("Bybit ticker fetch failed")
        return []

def fetch_kucoin_tickers() -> List[Dict[str, Any]]:
    try:
        r = get_session().get(EXCHANGE_ENDPOINTS["kucoin_ticker"], timeout=8)
        r.raise_for_status()
        return r.json().get("data", {}).get("ticker", [])
    except Exception:
        logger.exception("KuCoin ticker fetch failed")
        return []

def fetch_okx_tickers() -> List[Dict[str, Any]]:
    try:
        r = get_session().get(EXCHANGE_ENDPOINTS["okx_ticker"], timeout=8)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        logger.exception("OKX ticker fetch failed")
        return []

# ------------------------------
# Trending helpers
# ------------------------------
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
        return "📈 Trending Pairs (Binance):\n" + "\n".join(out) if out else "No trending data."
    except Exception:
        logger.exception("Trending pairs fetch failed")
        return "Failed to fetch trending pairs."

def fetch_trending_pairs_branded(limit: int = 10) -> Tuple[Optional[BytesIO], str]:
    rows = []
    try:
        # Aggregate tickers from multiple exchanges
        for fetcher, exch in [(fetch_binance_tickers, "Binance"), 
                              (fetch_bybit_tickers, "Bybit"),
                              (fetch_kucoin_tickers, "KuCoin"),
                              (fetch_okx_tickers, "OKX")]:
            try:
                for t in fetcher():
                    sym = t.get("symbol") or t.get("symbolName") or t.get("instId") or ""
                    sym = sym.replace("-", "").upper()
                    change = float(t.get("priceChangePercent") or t.get("price_24h_pcnt") or t.get("changeRate",0) or t.get("change24h",0) or 0)
                    vol = float(t.get("quoteVolume") or t.get("volValue") or t.get("volCcy24h",0) or 0)
                    if sym: rows.append({"symbol": sym, "change": change, "vol": vol, "exchange": exch})
            except Exception:
                logger.debug(f"{exch} ticker aggregation failed", exc_info=True)

        if not rows:
            return None, "No exchange data available."

        df = pd.DataFrame(rows)
        df = df[df["vol"] > 0].sort_values("change", ascending=False).head(limit)

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
            logger.debug("Sparkline generation failed", exc_info=True)

        if create_brand_image:
            buf = create_brand_image(lines, chart_img=chart_bytes)
            return buf, "Top Multi-Exchange Trending Pairs"
        else:
            return None, "\n".join(lines)
    except Exception:
        logger.exception("Branded trending pairs generation failed")
        return None, "Error generating trending pairs"

# ------------------------------
# Kline parsing helpers
# ------------------------------
def _parse_ohlcv_list(arr: List) -> Optional[pd.DataFrame]:
    try:
        if not arr or not isinstance(arr, list): return None
        df = pd.DataFrame(arr)
        df = df.iloc[:, :6]
        df.columns = ["open_time","open","high","low","close","volume"]
        ts = float(df["open_time"].iloc[0])
        unit = "ms" if ts > 1e12 else "s"
        df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit=unit)
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df[["open_time","open","high","low","close","volume"]]
    except Exception:
        logger.exception("OHLCV parsing failed")
        return None

def _parse_binance_klines(raw) -> Optional[pd.DataFrame]:
    try:
        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df[["open_time","open","high","low","close","volume"]]
    except Exception:
        logger.exception("Binance kline parsing failed")
        return None

# ------------------------------
# Symbol normalization
# ------------------------------
def _normalize_for_exchange(symbol: str, exchange: str) -> str:
    s = symbol.upper().replace("/", "").replace("-", "")
    if exchange.lower() in ("okx","kucoin"):
        if s.endswith("USDT"): return s[:-4] + "-USDT"
        if s.endswith("USD"): return s[:-3] + "-USD"
    return s

def _kucoin_interval_map(interval: str) -> str:
    m = {"1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1hour","4h":"4hour","1d":"1day"}
    return m.get(interval, interval)

# ------------------------------
# Fetch klines multi-exchange
# ------------------------------
def fetch_klines_multi(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 300, exchange: str = "binance") -> Optional[pd.DataFrame]:
    try:
        exchange = (exchange or "binance").lower()
        sess = get_session()

        # CCXT preferred
        if HAS_CCXT:
            try:
                ex = getattr(ccxt, exchange)()
                ex.load_markets()
                cc_sym = symbol.replace("USDT","/USDT").replace("USD","/USD")
                ohlcv = ex.fetch_ohlcv(cc_sym, timeframe=interval, limit=int(limit))
                return _parse_ohlcv_list(ohlcv)
            except Exception:
                logger.debug("CCXT fetch failed", exc_info=True)

        # REST fallback per exchange
        if exchange == "binance":
            r = sess.get(EXCHANGE_ENDPOINTS["binance_klines"], params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
            r.raise_for_status()
            return _parse_binance_klines(r.json())

        if exchange == "bybit":
            r = sess.get(EXCHANGE_ENDPOINTS["bybit_klines"], params={"category":"spot","symbol":symbol,"interval":interval,"limit":limit}, timeout=10)
            r.raise_for_status()
            arr = r.json().get("result", {}).get("list") or r.json().get("result")
            return _parse_ohlcv_list(arr)

        if exchange == "kucoin":
            ku_sym = _normalize_for_exchange(symbol, "kucoin")
            r = sess.get(EXCHANGE_ENDPOINTS["kucoin_klines"], params={"symbol":ku_sym,"type":_kucoin_interval_map(interval)}, timeout=10)
            r.raise_for_status()
            return _parse_ohlcv_list(r.json().get("data"))

        if exchange == "okx":
            ok_sym = _normalize_for_exchange(symbol, "okx")
            r = sess.get(EXCHANGE_ENDPOINTS["okx_klines"], params={"instId":ok_sym,"bar":interval,"limit":limit}, timeout=10)
            r.raise_for_status()
            return _parse_ohlcv_list(r.json().get("data"))

    except Exception:
        logger.exception("fetch_klines_multi failed")

    return None
