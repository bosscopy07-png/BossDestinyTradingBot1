# pro_features.py
"""
Pro Market Intelligence Module for Destiny Trading Empire Bot.

Features:
- top_gainers_pairs(limit=...) -> short text summary (multi-exchange)
- fear_and_greed_index() -> returns F&G string
- futures_leverage_suggestion(symbol, df=None) -> volatility-based leverage suggestion
- quickchart_price_image(symbol, interval, points, exchange) -> PNG bytes or None
- ai_market_brief_text(symbol, exchanges) -> AI-generated brief
- momentum_and_candle_analysis(symbol, interval, limit, exchange) -> momentum/candle checks
- pro_market_report(symbols, exchanges, interval) -> combined report (text + optional image)
- get_multi_exchange_snapshot(symbol, exchanges) -> latest prices & avg
"""

import os
import time
import json
import traceback
import requests
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional

# Local project imports (if missing, provide graceful fallbacks)
try:
    from market_providers import fetch_trending_pairs_text, fetch_trending_pairs_branded, fetch_klines_multi, get_session
except Exception:
    fetch_trending_pairs_text = None
    fetch_trending_pairs_branded = None
    fetch_klines_multi = None
    get_session = None

try:
    from ai_client import ai_analysis_text
except Exception:
    ai_analysis_text = None

try:
    from signal_engine import generate_signal, detect_candle_pattern
except Exception:
    # We will still offer a wrapper that uses generate_signal if present
    generate_signal = None
    detect_candle_pattern = None

# QuickChart (public)
QUICKCHART_URL = "https://quickchart.io/chart"

# Basic defaults
DEFAULT_EXCHANGES = os.getenv("DEFAULT_EXCHANGES", "binance,bybit,kucoin,okx").split(",")
BRAND = "Destiny Trading Empire Bot 💎"

# ---------------------------
# Helper utilities
# ---------------------------
def _safe_request_get(url, params=None, timeout=10):
    try:
        s = get_session() if get_session else requests
        r = s.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception:
        traceback.print_exc()
        return None

def _brand_text(txt: str) -> str:
    if BRAND not in txt:
        return f"{txt}\n\n— <b>{BRAND}</b>"
    return txt

# ---------------------------
# Top movers & trending
# ---------------------------
def top_gainers_pairs(limit: int = 5, pairs: Optional[List[str]] = None, exchanges: Optional[List[str]] = None) -> str:
    """
    Return a concise text of top gainers/losers across exchanges.
    If branded image generator is available, it will call it but return a text summary so callers can decide.
    """
    try:
        if fetch_trending_pairs_text:
            # prefer textual trending function if available
            txt = fetch_trending_pairs_text(pairs=pairs, exchanges=exchanges)
            return _brand_text(txt)
        # Fallback: attempt to use branded image function (returns image+caption)
        if fetch_trending_pairs_branded:
            img, caption = fetch_trending_pairs_branded(limit=limit)
            if caption:
                return _brand_text(caption)
        return _brand_text("Top movers data not available (market_providers missing).")
    except Exception as e:
        traceback.print_exc()
        return _brand_text(f"Error fetching top movers: {e}")

# ---------------------------
# Fear & Greed Index
# ---------------------------
def fear_and_greed_index() -> str:
    """Return current Fear & Greed Index text"""
    try:
        r = _safe_request_get("https://api.alternative.me/fng/", timeout=8)
        if not r:
            return _brand_text("F&G data unavailable (request failed).")
        j = r.json()
        if "data" in j and j["data"]:
            val = int(j["data"][0]["value"])
            txt = j["data"][0]["value_classification"]
            return _brand_text(f"😐 Fear & Greed Index: {val} — {txt}")
        return _brand_text("F&G data unavailable")
    except Exception as e:
        traceback.print_exc()
        return _brand_text(f"Error fetching F&G: {e}")

# ---------------------------
# Multi-exchange snapshot
# ---------------------------
def get_multi_exchange_snapshot(symbol: str, exchanges: Optional[List[str]] = None) -> Dict:
    """
    Return latest close prices from multiple exchanges and average.
    Uses fetch_klines_multi if available, otherwise tries Binance only.
    """
    exchanges = exchanges or DEFAULT_EXCHANGES
    prices = {}
    for ex in exchanges:
        try:
            if fetch_klines_multi:
                df = fetch_klines_multi(symbol=symbol, interval="1h", limit=1, exchange=ex)
            else:
                # fallback: attempt public Binance klines via requests
                df = fetch_klines_multi(symbol=symbol, interval="1h", limit=1, exchange="binance") if fetch_klines_multi else None
            if df is None or df.empty:
                prices[ex] = None
            else:
                prices[ex] = float(df['close'].iloc[-1])
        except Exception:
            traceback.print_exc()
            prices[ex] = None
    valid = [p for p in prices.values() if p is not None]
    avg = sum(valid)/len(valid) if valid else None
    return {"avg_price": avg, "prices": prices}

# ---------------------------
# Futures leverage suggestion
# ---------------------------
def futures_leverage_suggestion(symbol: str, df=None) -> Dict:
    """
    Suggest futures leverage based on historical volatility.
    Lower volatility -> higher leverage suggestion (cautious defaults).
    """
    try:
        if df is None:
            if fetch_klines_multi:
                df = fetch_klines_multi(symbol=symbol, interval="1h", limit=120, exchange="binance")
            else:
                return {"error": "klines fetcher not available"}
        returns = df['close'].pct_change().dropna()
        daily_vol = returns.std() * (24 ** 0.5)
        if daily_vol < 0.005:
            lev = 50
        elif daily_vol < 0.01:
            lev = 20
        elif daily_vol < 0.02:
            lev = 10
        elif daily_vol < 0.04:
            lev = 5
        else:
            lev = 2
        return {"vol": round(daily_vol, 6), "leverage": lev}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# ---------------------------
# QuickChart image generator
# ---------------------------
def quickchart_price_image(symbol: str, interval: str = "1h", points: int = 30, exchange: str = "binance") -> Optional[bytes]:
    """
    Return PNG bytes showing price line for last `points` closes using QuickChart.
    """
    try:
        if not fetch_klines_multi:
            return None
        df = fetch_klines_multi(symbol=symbol, interval=interval, limit=points, exchange=exchange)
        if df is None or df.empty:
            return None
        closes = df['close'].astype(float).tolist()[-points:]
        labels = list(range(len(closes)))
        chart_cfg = {
            "type": "line",
            "data": {"labels": labels, "datasets": [{"label": symbol, "data": closes, "fill": False}]},
            "options": {"plugins": {"legend": {"display": False}}, "elements": {"point": {"radius": 0}}}
        }
        params = {"c": json.dumps(chart_cfg), "width": 900, "height": 360}
        r = requests.get(QUICKCHART_URL, params=params, timeout=12)
        r.raise_for_status()
        return r.content
    except Exception:
        traceback.print_exc()
        return None

# ---------------------------
# AI Market Brief
# ---------------------------
def ai_market_brief_text(symbol: str, exchanges: Optional[List[str]] = None) -> str:
    """
    Return an AI-generated market brief about the symbol.
    Requires ai_client.ai_analysis_text to be available.
    """
    try:
        snapshot = get_multi_exchange_snapshot(symbol, exchanges or DEFAULT_EXCHANGES)
        prompt = (
            f"Short market brief for {symbol}.\n"
            f"Latest prices across exchanges: {snapshot['prices']}\n"
            f"Use this info to give: 1) current market bias, 2) key support/resistance, 3) immediate risks."
        )
        if ai_analysis_text:
            return _brand_text(ai_analysis_text(prompt))
        else:
            return _brand_text(f"AI service not available. Market snapshot: {snapshot}")
    except Exception as e:
        traceback.print_exc()
        return _brand_text(f"⚠️ Error generating AI brief: {e}")

# ---------------------------
# Momentum & Candle Pattern Analysis
# ---------------------------
def momentum_and_candle_analysis(symbol: str, interval: str = "1h", limit: int = 200, exchange: str = "binance") -> Dict:
    """
    Returns combination of:
    - indicator summary (EMA/SMA cross, RSI, MACD)
    - candle pattern detection (if detect_candle_pattern is available)
    - volatility (std)
    """
    try:
        if not fetch_klines_multi:
            return {"error": "klines fetcher not available"}
        df = fetch_klines_multi(symbol=symbol, interval=interval, limit=limit, exchange=exchange)
        if df is None or df.empty:
            return {"error": "no market data"}

        # ensure float columns
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['open'] = df['open'].astype(float)

        # simple indicators
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        avg_gain = up.rolling(14).mean()
        avg_loss = down.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        analysis = {
            "symbol": symbol,
            "interval": interval,
            "close": float(last['close']),
            "ema20": float(last['ema20']),
            "ema50": float(last['ema50']),
            "ema_cross": "bullish" if last['ema20'] > last['ema50'] else ("bearish" if last['ema20'] < last['ema50'] else "flat"),
            "rsi": float(last['rsi']) if not pd_isna(last['rsi']) else None,
            "macd": float(last['macd']),
            "macd_signal": float(last['macd_sig']),
            "macd_hist": float(last['macd'] - last['macd_sig']),
            "volatility": float(df['close'].pct_change().std() * (24 ** 0.5)),
        }

        # Candle pattern detection (use provided detect_candle_pattern if present)
        candle_pattern = None
        if detect_candle_pattern:
            try:
                cp = detect_candle_pattern(symbol, interval=interval, points=20, exchanges=[exchange])
                candle_pattern = cp
            except Exception:
                traceback.print_exc()
                candle_pattern = None

        analysis["candle_pattern"] = candle_pattern or "N/A"

        # Momentum score (simple heuristic)
        score = 0.5
        if analysis["ema_cross"] == "bullish":
            score += 0.15
        else:
            score -= 0.1
        macd_hist = analysis["macd_hist"]
        if macd_hist > 0:
            score += 0.1
        else:
            score -= 0.05
        rsi_val = analysis.get("rsi") or 50
        if rsi_val < 30:
            score -= 0.1
        elif rsi_val > 70:
            score -= 0.05

        analysis["momentum_score"] = round(min(max(score, 0.01), 0.99), 2)
        return analysis
    except Exception:
        traceback.print_exc()
        return {"error": "analysis_failed"}

# small helper to avoid pandas import in top scope
def pd_isna(x):
    try:
        import pandas as _pd
        return _pd.isna(x)
    except Exception:
        return x is None

# ---------------------------
# Pro Combined Market Report
# ---------------------------
def pro_market_report(symbols: List[str], exchanges: Optional[List[str]] = None, interval: str = "1h", top_n: int = 3) -> Dict:
    """
    Produce a combined report across requested symbols:
    - quick multi-exchange snapshot (avg price)
    - momentum & candle analysis
    - leverage suggestion
    - quickchart PNG (first symbol only)
    - AI brief (if available)
    Returns:
      {
         "timestamp": "...",
         "summary_text": "...",    # short text suitable for Telegram
         "image": <bytes>|None,    # quickchart or branded image bytes
         "details": { <symbol>: {...} }
      }
    """
    exchanges = exchanges or DEFAULT_EXCHANGES
    report = {"timestamp": datetime.utcnow().isoformat(), "summary_text": "", "image": None, "details": {}}
    try:
        details = {}
        winners = []
        for s in symbols:
            # snapshot
            snap = get_multi_exchange_snapshot(s, exchanges)
            # analysis
            analysis = momentum_and_candle_analysis(s, interval=interval, limit=120, exchange=exchanges[0])
            # leverage
            lev = futures_leverage_suggestion(s, df=None)  # internal will fetch klines
            details[s] = {"snapshot": snap, "analysis": analysis, "leverage": lev}
            # build quick ranking by momentum score if available
            mscore = analysis.get("momentum_score") if isinstance(analysis, dict) else 0.0
            winners.append((s, mscore))

        # rank top by score
        winners.sort(key=lambda x: x[1] or 0.0, reverse=True)
        top_list = winners[:top_n]

        # build summary text
        lines = [f"📊 Pro Market Report — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"]
        for s, score in top_list:
            snap = details[s]['snapshot']
            price = snap.get("avg_price") if isinstance(snap, dict) else None
            lines.append(f"{s}: Score {score:.2f} — AvgPrice: {price if price is not None else 'N/A'}")

        report["summary_text"] = _brand_text("\n".join(lines))
        report["details"] = details

        # quickchart for top symbol
        top_sym = symbols[0] if symbols else None
        if top_sym:
            img = quickchart_price_image(top_sym, interval=interval, points=60, exchange=exchanges[0])
            if img:
                report["image"] = img

        # optional AI brief for top symbol
        if ai_analysis_text and top_sym:
            try:
                report["ai_brief"] = ai_market_brief_text(top_sym, exchanges=exchanges[:3])
            except Exception:
                report["ai_brief"] = "AI brief error"

        return report
    except Exception:
        traceback.print_exc()
        return {"error": "pro_report_failed"}

# ---------------------------
# Small utilities exported for bot_runner compatibility
# ---------------------------
# Keep function names used by bot_runner.py:
def ai_market_brief(symbol):
    return ai_market_brief_text(symbol)

# expose module-level list of helpers
__all__ = [
    "top_gainers_pairs",
    "fear_and_greed_index",
    "futures_leverage_suggestion",
    "quickchart_price_image",
    "ai_market_brief_text",
    "momentum_and_candle_analysis",
    "pro_market_report",
    "get_multi_exchange_snapshot",
    "ai_market_brief",
                                ]
