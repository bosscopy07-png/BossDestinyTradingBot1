# pro_features.py
import os
import traceback
import requests
import json
from market_providers import fetch_trending_pairs_text, fetch_trending_pairs_branded
from ai_client import ai_analysis_text
from market_providers import fetch_klines_multi

QUICKCHART_URL = "https://quickchart.io/chart"

def top_gainers_pairs(limit=5):
    try:
        img, caption = fetch_trending_pairs_branded(limit=limit)
        if img:
            return caption
        return fetch_trending_pairs_text()
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"

def fear_and_greed_index():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
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

def futures_leverage_suggestion(symbol, df=None):
    try:
        if df is None:
            df = fetch_klines_multi(symbol=symbol, interval="1h", limit=120, exchange="binance")
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

def quickchart_price_image(symbol, interval="1h", points=30, exchange="binance"):
    # lightweight quickchart fetcher for images (public)
    try:
        df = fetch_klines_multi(symbol=symbol, interval=interval, limit=points, exchange=exchange)
        closes = df['close'].tolist()[-points:]
        labels = [str(i+1) for i in range(len(closes))]
        cfg = {
            "type":"line",
            "data":{"labels":labels,"datasets":[{"label":symbol,"data":closes,"fill":False}]},
            "options":{"plugins":{"legend":{"display":False}}}
        }
        params = {"c": json.dumps(cfg), "width":800, "height":360}
        r = requests.get(QUICKCHART_URL, params=params, timeout=12)
        r.raise_for_status()
        return r.content
    except Exception:
        traceback.print_exc()
        return None

def ai_market_brief_text(symbol):
    try:
        data = fetch_klines_multi(symbol=symbol, interval="1h", limit=50, exchange="binance")
        prompt = f"Give a short market brief for {symbol}. Last close {data['close'].iloc[-1]} and trend summary."
        return ai_analysis_text(prompt)
    except Exception:
        traceback.print_exc()
        return "AI brief not available"
