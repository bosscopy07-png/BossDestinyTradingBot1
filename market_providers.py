# market_providers.py
import os
import time
import traceback
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO

# Prefer ccxt if available for many exchanges; otherwise fallback to REST public endpoints
try:
    import ccxt
    HAS_CCXT = True
except Exception:
    HAS_CCXT = False

# small helper session with retry
def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "BossDestinyBot/2.0"})
    proxy = os.getenv("PROXY_URL")
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504], allowed_methods=frozenset(["GET","POST"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# ---- public tickers (fast) ----
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"
BYBIT_TICKER = "https://api.bybit.com/v2/public/tickers"
KUCOIN_TICKER = "https://api.kucoin.com/api/v1/market/allTickers"
OKX_TICKER = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"

def fetch_binance_tickers():
    s = get_session()
    r = s.get(BINANCE_TICKER, timeout=8)
    r.raise_for_status()
    return r.json()

def fetch_bybit_tickers():
    s = get_session()
    r = s.get(BYBIT_TICKER, timeout=8)
    r.raise_for_status()
    return r.json().get("result", [])

def fetch_kucoin_tickers():
    s = get_session()
    r = s.get(KUCOIN_TICKER, timeout=8)
    r.raise_for_status()
    return r.json().get("data", {}).get("ticker", [])

def fetch_okx_tickers():
    s = get_session()
    r = s.get(OKX_TICKER, timeout=8)
    r.raise_for_status()
    return r.json().get("data", [])

# build a compact text trending quickly
def fetch_trending_pairs_text(pairs=None, exchanges=None):
    # simple textual fallback for bot messages
    try:
        if pairs is None:
            pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
        out = []
        s = get_session()
        r = s.get(BINANCE_TICKER, timeout=8)
        r.raise_for_status()
        tickers = r.json()
        for p in pairs:
            t = next((x for x in tickers if x.get("symbol") == p), None)
            if t:
                out.append(f"{p}: {float(t.get('priceChangePercent',0)):.2f}% vol:{int(float(t.get('quoteVolume',0))):,}")
        return "📈 Trending Pairs:\n" + "\n".join(out)
    except Exception:
        traceback.print_exc()
        return "Failed to fetch trending pairs."

# ----- branded image generator function (uses image_utils.create_brand_image) -----
from image_utils import create_brand_image

def fetch_trending_pairs_branded(limit=10):
    """Query multiple exchanges and return a branded image + caption."""
    rows = []
    try:
        # Binance
        try:
            b = fetch_binance_tickers()
            for item in b:
                sym = item.get("symbol")
                pct = float(item.get("priceChangePercent", 0))
                vol = float(item.get("quoteVolume", 0))
                rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "Binance"})
        except Exception:
            pass

        # Bybit
        try:
            by = fetch_bybit_tickers()
            for item in by:
                sym = item.get("symbol")
                pct = float(item.get("price_24h_pcnt", 0)) * 100 if item.get("price_24h_pcnt") is not None else 0
                vol = float(item.get("quote_volume", 0)) if item.get("quote_volume") else 0
                rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "Bybit"})
        except Exception:
            pass

        # KuCoin
        try:
            k = fetch_kucoin_tickers()
            for item in k:
                sym = item.get("symbol").replace("-", "")
                pct = float(item.get("changeRate", 0)) * 100 if item.get("changeRate") else 0
                vol = float(item.get("volValue", 0)) if item.get("volValue") else 0
                rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "KuCoin"})
        except Exception:
            pass

        # OKX
        try:
            o = fetch_okx_tickers()
            for item in o:
                sym = item.get("instId").replace("-", "")
                pct = float(item.get("change24h", 0)) if item.get("change24h") else 0
                vol = float(item.get("volCcy24h", 0)) if item.get("volCcy24h") else 0
                rows.append({"symbol": sym, "change": pct, "vol": vol, "exchange": "OKX"})
        except Exception:
            pass

        if not rows:
            return None, "No exchange data available."

        # convert to DataFrame and pick top movers
        df = pd.DataFrame(rows)
        df = df[df["vol"] > 0]
        df = df.sort_values("change", ascending=False).head(limit)

        lines = [f"🔥 Multi-Exchange Trending Pairs ({len(df)} picks)"]
        for _, r in df.iterrows():
            lines.append(f"{r['symbol']:<10} | {r['change']:+.2f}% | vol:{int(r['vol']):,} | {r['exchange']}")

        img_buf = create_brand_image(lines)
        caption = "Top Multi-Exchange Trending Pairs"
        return img_buf, caption

    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {e}"

# ---- klines / candle data (multi-exchange where possible) ----
import pandas as pd
def _parse_binance_klines(raw):
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def fetch_klines_multi(symbol="BTCUSDT", interval="1h", limit=300, exchange="binance"):
    """Return pandas DataFrame for the chosen exchange if available (public)."""
    try:
        exchange = exchange.lower()
        sess = get_session()
        if exchange == "binance":
            url = "https://api.binance.com/api/v3/klines"
            r = sess.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
            r.raise_for_status()
            return _parse_binance_klines(r.json())
        # bybit, kucoin, okx public klines can be added similarly (for brevity they fallback to Binance)
        # If ccxt is available we will use it:
        if HAS_CCXT:
            ex = ccxt.binance() if exchange == "binance" else ccxt.bybit() if exchange == "bybit" else ccxt.okx()
            raw = ex.fetch_ohlcv(symbol.upper(), timeframe=interval, limit=limit)
            df = pd.DataFrame(raw, columns=["open_time","open","high","low","close","volume"])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            return df
        # fallback: call binance for symbol
        url = "https://api.binance.com/api/v3/klines"
        r = sess.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
        r.raise_for_status()
        return _parse_binance_klines(r.json())
    except Exception:
        traceback.print_exc()
        return None
