# market_providers.py
import os
import time
import traceback
from datetime import datetime
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
COINGECKO_MAP = {"BTCUSDT":"bitcoin","ETHUSDT":"ethereum","BNBUSDT":"binancecoin","SOLUSDT":"solana","XRPUSDT":"ripple","DOGEUSDT":"dogecoin"}

def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent":"BossDestiny/1.0"})
    proxy = os.getenv("PROXY_URL")
    if proxy:
        s.proxies.update({"http":proxy,"https":proxy})
    retries = Retry(total=3, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=frozenset(["GET","POST"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def normalize_interval(s):
    if not s: return "1h"
    s2 = s.strip().lower()
    return s2 if s2 in {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"} else "1h"

def fetch_klines_binance(symbol="BTCUSDT", interval="1h", limit=300):
    sess = get_session()
    params = {"symbol": symbol.upper(), "interval": normalize_interval(interval), "limit": int(limit)}
    r = sess.get(BINANCE_KLINES, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def fetch_klines_coingecko(symbol="BTCUSDT", interval="1h", limit=300):
    coin = COINGECKO_MAP.get(symbol.upper())
    if not coin:
        raise RuntimeError("Coingecko fallback not available for symbol " + symbol)
    sess = get_session()
    r = sess.get(COINGECKO_MARKET_CHART.format(id=coin), params={"vs_currency":"usd","days":7}, timeout=10)
    r.raise_for_status()
    js = r.json()
    prices = js.get("prices", [])[-limit:]
    rows = []
    for p in prices:
        ts = int(p[0]); price = float(p[1])
        rows.append([pd.to_datetime(ts, unit="ms"), price, price, price, price, 0.0])
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume"])
    return df

def fetch_klines_df(symbol="BTCUSDT", interval="1h", limit=300):
    errs = []
    try:
        return fetch_klines_binance(symbol, interval, limit)
    except Exception as e:
        errs.append("Binance:" + str(e))
        time.sleep(0.3)
    try:
        return fetch_klines_coingecko(symbol, interval, limit)
    except Exception as e:
        errs.append("Coingecko:" + str(e))
    raise RuntimeError("All providers failed: " + " | ".join(errs))

def fetch_trending_pairs():
    try:
        sess = get_session()
        r = sess.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8)
        r.raise_for_status()
        tickers = r.json()
        pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
        out = []
        for p in pairs:
            t = next((x for x in tickers if x.get("symbol") == p), None)
            if t:
                out.append(f"{p}: {float(t.get('priceChangePercent', 0)):.2f}% vol:{int(float(t.get('quoteVolume', 0))):,}")
        return "Trending Pairs:\n" + "\n".join(out)
    except Exception:
        traceback.print_exc()
        return "Failed to fetch trending pairs."
