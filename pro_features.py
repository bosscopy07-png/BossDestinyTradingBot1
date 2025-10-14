# pro_features.py
import os, time, traceback, json, requests
from datetime import datetime, timedelta
from market_providers import get_session
from storage import load_data
from ai_client import ai_analysis_text

# QuickChart base
QUICKCHART_URL = "https://quickchart.io/chart"

def top_gainers_pairs(pairs=None, limit=5):
    """
    Return text list of top gainers/losers among provided pairs using Binance 24h ticker.
    """
    if pairs is None:
        pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,DOGEUSDT,XRPUSDT").split(",")
    try:
        sess = get_session()
        r = sess.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8); r.raise_for_status()
        tickers = r.json()
        rows = []
        for p in pairs:
            t = next((x for x in tickers if x.get("symbol")==p), None)
            if t:
                rows.append({
                    "symbol": p,
                    "change": float(t.get("priceChangePercent",0)),
                    "vol": float(t.get("quoteVolume",0))
                })
        if not rows:
            return "No data"
        gainers = sorted(rows, key=lambda x: x["change"], reverse=True)[:limit]
        losers = sorted(rows, key=lambda x: x["change"])[:limit]
        txt = "📈 Top Gainers:\n"
        for g in gainers:
            txt += f"{g['symbol']}: {g['change']:.2f}% vol:{int(g['vol']):,}\n"
        txt += "\n📉 Top Losers:\n"
        for g in losers:
            txt += f"{g['symbol']}: {g['change']:.2f}% vol:{int(g['vol']):,}\n"
        return txt
    except Exception as e:
        traceback.print_exc()
        return f"Error fetching top movers: {e}"

def fear_and_greed_index():
    """Fetch Fear & Greed index from alternative.me"""
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        r.raise_for_status()
        j = r.json()
        if "data" in j and j["data"]:
            val = int(j["data"][0]["value"])
            txt = j["data"][0]["value_classification"]
            return f"😐 Fear & Greed: {val} — {txt}"
        return "F&G data unavailable"
    except Exception as e:
        traceback.print_exc()
        return f"Error fetching F&G: {e}"

def quickchart_price_image(symbol, interval="1h", points=30):
    """
    Fetch recent close prices from market_providers and request quickchart image.
    Returns (image_bytes or None, error_message or None)
    """
    from market_providers import fetch_klines_df, normalize_interval
    try:
        df = fetch_klines_df(symbol=symbol, interval=interval, limit=points)
        closes = df["close"].tolist()[-points:]
        labels = [str(i) for i in range(len(closes))]
        chart_cfg = {
            "type": "line",
            "data": {"labels": labels, "datasets": [{"label": symbol, "data": closes, "fill": False}]},
            "options": {"plugins":{"legend":{"display":False}},"scales":{"x":{"display":False}}}
        }
        params = {"c": json.dumps(chart_cfg), "width": 800, "height": 360, "devicePixelRatio":2}
        r = requests.get(QUICKCHART_URL, params=params, timeout=15)
        r.raise_for_status()
        return r.content, None
    except Exception as e:
        traceback.print_exc()
        return None, f"QuickChart error: {e}"

def futures_leverage_suggestion(symbol, df=None):
    """
    Simple rule-of-thumb leverage suggestion based on volatility (std of returns).
    Returns a dict {suggestion_str, recommended_leverage}
    """
    try:
        if df is None:
            from market_providers import fetch_klines_df
            df = fetch_klines_df(symbol=symbol, interval="1h", limit=120)
        returns = df["close"].pct_change().dropna()
        vol = returns.std() * (24**0.5)  # rough dailyized volatility
        # map vol to leverage: lower vol -> higher leverage, cap at 20x
        if vol < 0.01:
            lev = 20
        elif vol < 0.02:
            lev = 10
        elif vol < 0.04:
            lev = 5
        else:
            lev = 2
        suggestion = f"Estimated daily vol: {vol:.3f}. Recommended leverage: {lev}x (conservative)"
        return {"vol": round(float(vol),5), "leverage": lev, "suggestion": suggestion}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def ai_market_brief_text(pairs=None):
    """
    Build a concise market brief prompt and get AI summary.
    """
    try:
        if pairs is None:
            pairs = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
        summary = "Provide a concise 6-sentence market brief and one-line trading verdict (BUY/SELL/HOLD) for these pairs:\n"
        for p in pairs:
            summary += f"- {p}\n"
        # call ai
        resp = ai_analysis_text(summary)
        return resp
    except Exception as e:
        traceback.print_exc()
        return f"AI brief error: {e}"
