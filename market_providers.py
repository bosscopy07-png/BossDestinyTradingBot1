import os
import time
import traceback
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
import matplotlib.pyplot as plt

# ✅ Import brand image generator
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
    s.headers.update({"User-Agent": "BossDestinyBot/2.0"})
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
# Combined & Trending Analysis
# ===============================
def fetch_all_pairs():
    """Combine data from all exchanges safely."""
    dfs = []
    for fetcher in [fetch_binance_pairs, fetch_bybit_pairs, fetch_kucoin_pairs, fetch_okx_pairs]:
        try:
            dfs.append(fetcher())
        except Exception as e:
            print(f"⚠️ Error fetching from {fetcher.__name__}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def get_trending_pairs(top_n=10):
    """Rank top and bottom movers by percentage change."""
    df = fetch_all_pairs()
    if df.empty:
        return None, None

    df = df.sort_values("priceChangePercent", ascending=False)
    top_gainers = df.head(top_n).reset_index(drop=True)
    top_losers = df.tail(top_n).reset_index(drop=True)
    return top_gainers, top_losers


def fetch_trending_pairs_branded(top_n=10):
    """Return image + text summary for Telegram."""
    top_gainers, top_losers = get_trending_pairs(top_n)
    if top_gainers is None:
        return None, "⚠️ Failed to fetch trending pairs."

    # Prepare message
    msg = "📈 **Top Gainers (24h)**\n"
    for _, row in top_gainers.iterrows():
        msg += f"{row['symbol']}: +{row['priceChangePercent']:.2f}%  | 💰 Vol: {row['volume']:.0f} ({row['exchange']})\n"

    msg += "\n📉 **Top Losers (24h)**\n"
    for _, row in top_losers.iterrows():
        msg += f"{row['symbol']}: {row['priceChangePercent']:.2f}%  | 💰 Vol: {row['volume']:.0f} ({row['exchange']})\n"

    # Create branded chart image
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_gainers["symbol"], top_gainers["priceChangePercent"], color="green", label="Gainers")
    ax.barh(top_losers["symbol"], top_losers["priceChangePercent"], color="red", label="Losers")
    ax.set_xlabel("% Change")
    ax.set_title("Top Gainers & Losers (24h)")
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    plt.close(fig)

    # Add brand overlay
    branded_img = create_brand_image(buf, title="Market Movers", subtitle="by BossDestinyBot 🚀")

    return branded_img, msg


# ===============================
# Kline (Chart Data) – Multi-Exchange
# ===============================

def fetch_klines_binance(symbol, interval="1h", limit=100):
    """Fetch klines from Binance."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


def fetch_klines_bybit(symbol, interval="60", limit=100):
    """Fetch klines from Bybit."""
    url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json().get("result", {}).get("list", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def fetch_klines_kucoin(symbol, interval="1hour", limit=100):
    """Fetch klines from KuCoin."""
    url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def fetch_klines_okx(symbol, interval="1H", limit=100):
    """Fetch klines from OKX."""
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={interval}&limit={limit}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def fetch_klines_multi(symbols, interval="1h", limit=100):
    """
    Fetch klines from multiple exchanges for multiple symbols.
    Auto-fallback between Binance, Bybit, KuCoin, and OKX.
    """
    klines_data = {}
    for symbol in symbols:
        fetched = False
        for fetcher in [
            ("Binance", fetch_klines_binance),
            ("Bybit", fetch_klines_bybit),
            ("KuCoin", fetch_klines_kucoin),
            ("OKX", fetch_klines_okx)
        ]:
            exchange, func = fetcher
            try:
                df = func(symbol, interval=interval, limit=limit)
                if not df.empty:
                    klines_data[symbol] = {"exchange": exchange, "data": df}
                    fetched = True
                    break
            except Exception as e:
                print(f"⚠️ {exchange} klines failed for {symbol}: {e}")
        if not fetched:
            klines_data[symbol] = {"exchange": None, "data": pd.DataFrame()}
    return klines_data
