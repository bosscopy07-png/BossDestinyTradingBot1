import os
import time
import traceback
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ✅ Import your brand image generator
from image_utils import create_brand_image

# ===============================
# API ENDPOINTS
# ===============================
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"
BYBIT_TICKER = "https://api.bybit.com/v5/market/tickers?category=spot"
KUCOIN_TICKER = "https://api.kucoin.com/api/v1/market/allTickers"
OKX_TICKER = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"

# ===============================
# Session Setup
# ===============================
def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "BossDestiny/1.0"})
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

# ===============================
# Exchange Fetchers
# ===============================
def fetch_binance_pairs():
    s = get_session()
    r = s.get(BINANCE_TICKER, timeout=8)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    df["exchange"] = "Binance"
    df["symbol"] = df["symbol"].astype(str)
    df["priceChangePercent"] = df["priceChangePercent"].astype(float)
    df["volume"] = df["quoteVolume"].astype(float)
    return df[["symbol", "priceChangePercent", "volume", "exchange"]]


def fetch_bybit_pairs():
    s = get_session()
    r = s.get(BYBIT_TICKER, timeout=8)
    r.raise_for_status()
    data = r.json().get("result", {}).get("list", [])
    rows = []
    for x in data:
        rows.append({
            "symbol": x["symbol"].replace("/", ""),
            "priceChangePercent": float(x.get("price24hPcnt", 0)) * 100,
            "volume": float(x.get("turnover24h", 0)),
            "exchange": "Bybit"
        })
    return pd.DataFrame(rows)


def fetch_kucoin_pairs():
    s = get_session()
    r = s.get(KUCOIN_TICKER, timeout=8)
    r.raise_for_status()
    data = r.json().get("data", {}).get("ticker", [])
    rows = []
    for x in data:
        rows.append({
            "symbol": x["symbol"].replace("-", ""),
            "priceChangePercent": float(x.get("changeRate", 0)) * 100,
            "volume": float(x.get("volValue", 0)),
            "exchange": "KuCoin"
        })
    return pd.DataFrame(rows)


def fetch_okx_pairs():
    s = get_session()
    r = s.get(OKX_TICKER, timeout=8)
    r.raise_for_status()
    data = r.json().get("data", [])
    rows = []
    for x in data:
        rows.append({
            "symbol": x["instId"].replace("-", ""),
            "priceChangePercent": float(x.get("change24h", 0)),
            "volume": float(x.get("volCcy24h", 0)),
            "exchange": "OKX"
        })
    return pd.DataFrame(rows)


# ===============================
# Multi-Exchange Trending (Branded)
# ===============================
def fetch_trending_pairs_branded(limit=10):
    """Return a Boss Destiny–branded trending pairs image."""
    dfs = []
    for f in [fetch_binance_pairs, fetch_bybit_pairs, fetch_kucoin_pairs, fetch_okx_pairs]:
        try:
            dfs.append(f())
        except Exception as e:
            print(f"[WARN] {f.__name__} failed: {e}")
            traceback.print_exc()
            continue

    if not dfs:
        return None, "⚠️ All exchanges failed to respond."

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["volume"] > 0]
    df = df.sort_values("priceChangePercent", ascending=False).head(limit)

    lines = [f"🔥 Multi-Exchange Trending Pairs ({len(df)} top picks)"]
    for _, row in df.iterrows():
        lines.append(
            f"{row['symbol']:<10} | {row['priceChangePercent']:+.2f}% | Vol: {int(row['volume']):,} | {row['exchange']}"
        )

    # Create branded image
    img_buf = create_brand_image(lines)
    return img_buf, "Top Multi-Exchange Trending Pairs"


# ===============================
# Standalone Test
# ===============================
if __name__ == "__main__":
    img_buf, caption = fetch_trending_pairs_branded(10)
    if img_buf:
        with open("trending_pairs.png", "wb") as f:
            f.write(img_buf.getbuffer())
        print("✅ Branded trending pairs image saved as trending_pairs.png")
    else:
        print("❌ Failed to generate trending pairs image")
