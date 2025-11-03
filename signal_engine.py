import requests
import pandas as pd
import numpy as np
import time

# -----------------------------
# CONFIG
# -----------------------------
PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "MATICUSDT", "ADAUSDT"]
TIMEFRAME = "15m"  # 1m, 5m, 15m, 1h, 4h, etc.
LIMIT = 200        # Number of candles to fetch

# Primary + fallback exchange URLs
EXCHANGES = {
    "binance": "https://api.binance.com/api/v3/klines?symbol={pair}&interval={tf}&limit={limit}",
    "bybit": "https://api.bybit.com/v5/market/kline?category=spot&symbol={pair}&interval={tf}&limit={limit}",
    "kucoin": "https://api.kucoin.com/api/v1/market/candles?type={tf}&symbol={pair}"
}

# -----------------------------
# CORE FUNCTIONS
# -----------------------------
def fetch_candles(pair, tf=TIMEFRAME, limit=LIMIT):
    for name, url in EXCHANGES.items():
        try:
            endpoint = url.format(pair=pair, tf=tf, limit=limit)
            print(f"Fetching {pair} from {name}...")
            r = requests.get(endpoint, timeout=10)
            r.raise_for_status()
            data = r.json()

            # Normalize data structure across exchanges
            if name == "binance":
                df = pd.DataFrame(data, columns=[
                    "time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"
                ])
                df["close"] = df["close"].astype(float)
            elif name == "bybit":
                df = pd.DataFrame(data["result"]["list"], columns=[
                    "time", "open", "high", "low", "close", "volume", "turnover"
                ])
                df["close"] = df["close"].astype(float)
            elif name == "kucoin":
                df = pd.DataFrame(data["data"], columns=[
                    "time", "open", "close", "high", "low", "volume", "turnover"
                ])
                df["close"] = df["close"].astype(float)
            else:
                continue

            # Sort by time ascending
            df = df.iloc[::-1].reset_index(drop=True)
            if len(df) > 10:
                return df
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")
            continue
    return None


def calculate_indicators(df):
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["RSI"] = compute_rsi(df["close"], 14)
    df["MACD"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def generate_signal(df, pair=None):
    if df is None or len(df) < 50:
        return "Error: Insufficient data"

    last = df.iloc[-1]

    # Signal logic
    if last["MA20"] > last["MA50"] and last["RSI"] < 70 and last["MACD"] > last["Signal"]:
        return f"{pair or ''} → STRONG BUY 📈"
    elif last["MA20"] < last["MA50"] and last["RSI"] > 30 and last["MACD"] < last["Signal"]:
        return f"{pair or ''} → STRONG SELL 📉"
    else:
        return f"{pair or ''} → NO CLEAR SIGNAL ⚖️"


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def get_signals():
    results = {}
    for pair in PAIRS:
        df = fetch_candles(pair)
        if df is not None:
            df = calculate_indicators(df)
            signal = generate_signal(df)
            results[pair] = signal
            print(f"{pair}: {signal}")
        else:
            results[pair] = "Error: No market data"
    return results


def get_latest_signals():
    """Wrapper for bot_runner to fetch all signals."""
    return get_signals()


# -----------------------------
# MAIN LOOP (for standalone testing)
# -----------------------------
if __name__ == "__main__":
    while True:
        signals = get_signals()
        print("✅ Signals updated:", signals)
        time.sleep(60 * 15)
