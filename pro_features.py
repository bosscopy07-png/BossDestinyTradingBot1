import os
import traceback
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from market_providers import get_session, fetch_klines_multi
from ai_client import ai_analysis_text

QUICKCHART_URL = "https://quickchart.io/chart"
DEFAULT_EXCHANGES = ["binance", "bybit", "kucoin", "okx", "ftx"]

# -------------------- TOP MOVERS --------------------
def top_gainers_pairs(pairs=None, exchanges=None, limit=5):
    if pairs is None:
        pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT").split(",")
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES

    all_rows = []
    try:
        sess = get_session()
        for ex in exchanges:
            tickers = []

            # ----- BINANCE -----
            if ex == "binance":
                r = sess.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
                r.raise_for_status()
                tickers = r.json()

            # ----- BYBIT -----
            elif ex == "bybit":
                r = sess.get("https://api.bybit.com/v2/public/tickers", timeout=10)
                r.raise_for_status()
                tickers = r.json().get("result", [])

            # ----- KUCOIN -----
            elif ex == "kucoin":
                r = sess.get("https://api.kucoin.com/api/v1/market/allTickers", timeout=10)
                r.raise_for_status()
                tickers = r.json().get("data", {}).get("ticker", [])

            # ----- OKX -----
            elif ex == "okx":
                r = sess.get("https://www.okx.com/api/v5/market/tickers?instType=SPOT", timeout=10)
                r.raise_for_status()
                tickers = r.json().get("data", [])

            # ----- FTX -----
            elif ex == "ftx":
                r = sess.get("https://ftx.com/api/markets", timeout=10)
                r.raise_for_status()
                tickers = r.json().get("result", [])

            # ---- Process tickers for pairs ----
            for p in pairs:
                t = None
                # Exchange-specific parsing
                try:
                    if ex == "binance":
                        t = next((x for x in tickers if x.get("symbol") == p), None)
                        if t:
                            all_rows.append({
                                "symbol": p,
                                "exchange": ex,
                                "change": float(t.get("priceChangePercent", 0)),
                                "vol": float(t.get("quoteVolume", 0)),
                                "lastPrice": float(t.get("lastPrice", 0))
                            })
                    elif ex == "bybit":
                        t = next((x for x in tickers if x.get("symbol") == p), None)
                        if t:
                            all_rows.append({
                                "symbol": p,
                                "exchange": ex,
                                "change": float(t.get("price_24h_pcnt", 0))*100,
                                "vol": float(t.get("quote_volume", 0)),
                                "lastPrice": float(t.get("last_price", 0))
                            })
                    elif ex == "kucoin":
                        t = next((x for x in tickers if x.get("symbol") == p), None)
                        if t:
                            all_rows.append({
                                "symbol": p,
                                "exchange": ex,
                                "change": float(t.get("changeRate", 0))*100,
                                "vol": float(t.get("volValue", 0)),
                                "lastPrice": float(t.get("last", 0))
                            })
                    elif ex == "okx":
                        t = next((x for x in tickers if x.get("instId") == p), None)
                        if t:
                            all_rows.append({
                                "symbol": p,
                                "exchange": ex,
                                "change": float(t.get("sodUtc0Pcnt", 0))*100,
                                "vol": float(t.get("volCcy24h", 0)),
                                "lastPrice": float(t.get("last", 0))
                            })
                    elif ex == "ftx":
                        t = next((x for x in tickers if x.get("name") == p), None)
                        if t:
                            all_rows.append({
                                "symbol": p,
                                "exchange": ex,
                                "change": float(t.get("change24h", 0))*100,
                                "vol": float(t.get("quoteVolume24h", 0)),
                                "lastPrice": float(t.get("last", 0))
                            })
                except Exception:
                    continue

        if not all_rows:
            return "No data available."

        gainers = sorted(all_rows, key=lambda x: x["change"], reverse=True)[:limit]
        losers = sorted(all_rows, key=lambda x: x["change"])[:limit]

        txt = "📈 Top Gainers:\n" + "\n".join(
            f"{g['symbol']}({g['exchange']}): {g['change']:.2f}% | vol:{int(g['vol']):,} | price:{g['lastPrice']}" for g in gainers
        )
        txt += "\n\n📉 Top Losers:\n" + "\n".join(
            f"{l['symbol']}({l['exchange']}): {l['change']:.2f}% | vol:{int(l['vol']):,} | price:{l['lastPrice']}" for l in losers
        )
        return txt

    except Exception as e:
        traceback.print_exc()
        return f"Error fetching top movers: {e}"


# -------------------- FEAR & GREED INDEX --------------------
def fear_and_greed_index():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=10)
        r.raise_for_status()
        j = r.json()
        if "data" in j and j["data"]:
            val = int(j["data"][0]["value"])
            txt = j["data"][0]["value_classification"]
            return f"😐 Fear & Greed Index: {val} — {txt}"
        return "F&G data unavailable"
    except Exception as e:
        traceback.print_exc()
        return f"Error fetching F&G: {e}"


# -------------------- MULTI-EXCHANGE PRICE --------------------
def get_multi_exchange_price(symbol, exchanges=None):
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES
    prices = {}
    for ex in exchanges:
        try:
            df = fetch_klines_df_multi(symbol=symbol, exchange=ex, interval="1h", limit=1)
            prices[ex] = df['close'].iloc[-1]
        except Exception:
            prices[ex] = None
    avg_price = np.mean([p for p in prices.values() if p is not None])
    return {"avg_price": avg_price, "prices": prices}


# -------------------- TECHNICAL INDICATORS --------------------
def compute_indicators(df):
    df = df.copy()
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_upper'] = df['close'].rolling(20).mean() + 2*df['close'].rolling(20).std()
    df['BB_lower'] = df['close'].rolling(20).mean() - 2*df['close'].rolling(20).std()
    return df

# -------------------- FUTURES SIGNAL --------------------
def futures_leverage_suggestion(symbol, df=None):
    try:
        if df is None:
            df = fetch_klines_df_multi(symbol=symbol, exchange=DEFAULT_EXCHANGES[0], interval="1h", limit=120)
        returns = df['close'].pct_change().dropna()
        daily_vol = returns.std() * (24**0.5)
        if daily_vol < 0.01:
            lev = 20
        elif daily_vol < 0.02:
            lev = 10
        elif daily_vol < 0.04:
            lev = 5
        else:
            lev = 2
        return {"vol": round(daily_vol,5), "leverage": lev}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def multi_exchange_futures_signal(symbol, exchanges=None):
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES
    try:
        data = get_multi_exchange_price(symbol, exchanges)
        df = fetch_klines_df_multi(symbol=symbol, exchange=exchanges[0], interval="1h", limit=50)
        df = compute_indicators(df)
        last = df.iloc[-1]

        signal_type = "HODL"
        if last['close'] > last['SMA_10'] and last['RSI'] < 70 and last['MACD'] > last['MACD_signal']:
            signal_type = "LONG"
        elif last['close'] < last['SMA_10'] and last['RSI'] > 30 and last['MACD'] < last['MACD_signal']:
            signal_type = "SHORT"

        entry = data['avg_price']
        if signal_type == "LONG":
            sl = entry * 0.995
            tp1 = entry * 1.005
        elif signal_type == "SHORT":
            sl = entry * 1.005
            tp1 = entry * 0.995
        else:
            sl = tp1 = None

        leverage = futures_leverage_suggestion(symbol, df).get('leverage')

        return {
            "symbol": symbol,
            "signal": signal_type,
            "entry": round(entry,2),
            "sl": round(sl,2) if sl else None,
            "tp1": round(tp1,2) if tp1 else None,
            "leverage": leverage,
            "prices": data['prices']
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# -------------------- CANDLESTICK PATTERN --------------------
def detect_candle_pattern(symbol, interval="1h", points=50, exchanges=None):
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES
    try:
        df = fetch_klines_df_multi(symbol=symbol, exchange=exchanges[0], interval=interval, limit=points)
        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        if last_candle['close'] > last_candle['open'] and body/last_candle['close'] > 0.005:
            return "BULLISH"
        elif last_candle['close'] < last_candle['open'] and body/last_candle['open'] > 0.005:
            return "BEARISH"
        return "HODL"
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# -------------------- SUPPORT & RESISTANCE --------------------
def support_resistance(symbol, interval="1h", points=100, exchanges=None):
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES
    try:
        df = fetch_klines_df_multi(symbol=symbol, exchange=exchanges[0], interval=interval, limit=points)
        pivot_high = df['high'].rolling(5, center=True).max()
        pivot_low = df['low'].rolling(5, center=True).min()
        support = round(pivot_low.min(), 2)
        resistance = round(pivot_high.max(), 2)
        return {"support": support, "resistance": resistance}
    except Exception as e:
        # ✅ Proper logging with traceback
        print(f"[support_resistance] Error for {symbol}: {e}")
        traceback.print_exc()
        return {"error": str(e)}
        
def quickchart_price_image(symbol, interval="1h", points=30, exchanges=None):
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES
    try:
        df = fetch_klines_multi(symbol=symbol, exchange=exchanges[0], interval=interval, limit=points)
        closes = df['close'].tolist()[-points:]
        labels = [str(i + 1) for i in range(len(closes))]
        chart_cfg = {
            "type": "line",
            "data": {"labels": labels, "datasets":[{"label": symbol, "data": closes, "fill": False, "borderColor": "blue"}]},
            "options": {"plugins":{"legend":{"display": False}}, "scales":{"x":{"display": True}, "y":{"display": True}}}
        }
        params = {"c": json.dumps(chart_cfg), "width": 800, "height": 360, "devicePixelRatio": 2}
        r = requests.get(QUICKCHART_URL, params=params, timeout=15)
        r.raise_for_status()
        return r.content
    except Exception as e:
        traceback.print_exc()
        return None
