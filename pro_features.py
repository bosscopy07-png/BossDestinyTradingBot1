# pro_features.py
"""
Pro Market Intelligence Module for Destiny Trading Empire Bot.

Advanced market analysis features including multi-exchange data aggregation,
technical analysis, AI-powered insights, and visualization generation.
"""

import os
import json
import logging
import traceback
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from decimal import Decimal

import requests
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger("pro_features")
logger.setLevel(logging.INFO)

# Local project imports with graceful fallbacks
try:
    from market_providers import (
        fetch_klines_multi,
        get_realtime_price,
        get_trending_pairs,
        get_trending_pairs_text,
        generate_sparkline
    )
    HAS_MARKET_PROVIDERS = True
except ImportError as e:
    logger.warning(f"market_providers not available: {e}")
    fetch_klines_multi = None
    get_realtime_price = None
    get_trending_pairs = None
    get_trending_pairs_text = None
    generate_sparkline = None
    HAS_MARKET_PROVIDERS = False

try:
    from ai_client import ai_analysis_text, get_market_brief
    HAS_AI = True
except ImportError as e:
    logger.warning(f"ai_client not available: {e}")
    ai_analysis_text = None
    get_market_brief = None
    HAS_AI = False

try:
    from image_utils import create_brand_image
    HAS_IMAGE_UTILS = True
except ImportError as e:
    logger.warning(f"image_utils not available: {e}")
    create_brand_image = None
    HAS_IMAGE_UTILS = False

try:
    from signal_engine import SignalType
    HAS_SIGNAL_ENGINE = True
except ImportError:
    SignalType = None
    HAS_SIGNAL_ENGINE = False

# Configuration
QUICKCHART_URL = "https://quickchart.io/chart"
FEAR_GREED_API = "https://api.alternative.me/fng/"
COINGECKO_API = "https://api.coingecko.com/api/v3"

DEFAULT_EXCHANGES = os.getenv("DEFAULT_EXCHANGES", "binance,bybit,kucoin,okx").split(",")
BRAND = "Destiny Trading Empire Bot 💎"


@dataclass
class MarketSnapshot:
    """Multi-exchange price snapshot."""
    symbol: str
    prices: Dict[str, Optional[float]]
    average: Optional[float]
    spread: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    timestamp: str


@dataclass
class LeverageSuggestion:
    """Futures leverage recommendation based on volatility."""
    symbol: str
    daily_volatility: float
    suggested_leverage: int
    risk_level: str
    max_position_size: Optional[float] = None


@dataclass
class MomentumAnalysis:
    """Technical momentum analysis results."""
    symbol: str
    interval: str
    current_price: float
    ema_cross: str
    rsi: Optional[float]
    macd_histogram: float
    momentum_score: float
    volatility_24h: float
    candle_pattern: Optional[str]
    trend_strength: str


# ----- HTTP Utilities -----
def _get_session() -> requests.Session:
    """Create configured requests session."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "DestinyTradingEmpireBot/1.0",
        "Accept": "application/json"
    })
    return session


_session: Optional[requests.Session] = None

def _safe_request(
    url: str,
    params: Optional[Dict] = None,
    timeout: int = 10,
    retries: int = 3
) -> Optional[requests.Response]:
    """
    Make safe HTTP request with retry logic.
    """
    global _session
    if _session is None:
        _session = _get_session()
    
    for attempt in range(retries):
        try:
            response = _session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return None


def _brand_text(text: str) -> str:
    """Append brand tag to text."""
    if BRAND not in text:
        return f"{text}\n\n— <b>{BRAND}</b>"
    return text


# ----- Market Data -----
def get_fear_greed_index() -> Dict[str, Any]:
    """
    Fetch current Fear & Greed Index from alternative.me.
    """
    try:
        response = _safe_request(FEAR_GREED_API, timeout=8)
        if not response:
            return {"error": "API request failed", "value": None, "classification": None}
        
        data = response.json()
        if "data" not in data or not data["data"]:
            return {"error": "Invalid response format", "value": None, "classification": None}
        
        fg_data = data["data"][0]
        value = int(fg_data.get("value", 0))
        classification = fg_data.get("value_classification", "Unknown")
        timestamp = datetime.fromtimestamp(int(fg_data.get("timestamp", 0))).isoformat()
        
        # Determine emoji
        emoji = "😱" if value <= 20 else "😨" if value <= 40 else "😐" if value <= 60 else "😏" if value <= 80 else "🤩"
        
        return {
            "value": value,
            "classification": classification,
            "emoji": emoji,
            "timestamp": timestamp,
            "text": f"{emoji} Fear & Greed Index: {value} — {classification}"
        }
        
    except Exception as e:
        logger.error(f"Fear & Greed fetch failed: {e}")
        return {"error": str(e), "value": None, "classification": None}


def get_multi_exchange_snapshot(
    symbol: str,
    exchanges: Optional[List[str]] = None
) -> MarketSnapshot:
    """
    Get price snapshot across multiple exchanges.
    """
    exchanges = exchanges or DEFAULT_EXCHANGES
    symbol = symbol.upper()
    prices: Dict[str, Optional[float]] = {}
    
    for exchange in exchanges:
        try:
            if fetch_klines_multi:
                # Try to get latest close from klines
                df = fetch_klines_multi(symbol, "1h", limit=1, exchange=exchange)
                if df is not None and not df.empty:
                    price = float(df["close"].iloc[-1])
                    prices[exchange] = price
                    continue
            
            # Fallback to realtime price if available
            if get_realtime_price:
                price = get_realtime_price(symbol)
                if price:
                    prices[exchange] = price
                    
        except Exception as e:
            logger.debug(f"Price fetch failed for {exchange}: {e}")
            prices[exchange] = None
    
    # Calculate statistics
    valid_prices = [p for p in prices.values() if p is not None]
    
    if valid_prices:
        avg_price = sum(valid_prices) / len(valid_prices)
        spread = max(valid_prices) - min(valid_prices)
        spread_percent = (spread / avg_price) * 100 if avg_price > 0 else 0
        best_bid = min(valid_prices)
        best_ask = max(valid_prices)
    else:
        avg_price = spread = spread_percent = best_bid = best_ask = None
    
    return MarketSnapshot(
        symbol=symbol,
        prices=prices,
        average=avg_price,
        spread=spread_percent,
        best_bid=best_bid,
        best_ask=best_ask,
        timestamp=datetime.utcnow().isoformat()
    )


def get_top_movers(
    limit: int = 10,
    pairs: Optional[List[str]] = None,
    min_volume: float = 1_000_000
) -> str:
    """
    Get top gaining/losing pairs with formatted output.
    """
    try:
        if get_trending_pairs_text:
            text = get_trending_pairs_text(pairs)
            return _brand_text(text)
        
        if get_trending_pairs:
            trending = get_trending_pairs(pairs, min_volume=min_volume, top_n=limit)
            if not trending:
                return _brand_text("📊 No trending data available.")
            
            lines = [f"🔥 Top {len(trending)} Movers (24h Change)"]
            for item in trending:
                emoji = "🟢" if item.get("change_24h", 0) > 0 else "🔴"
                lines.append(
                    f"{emoji} {item.get('symbol', 'N/A'):<12} | "
                    f"{item.get('change_24h', 0):+.2f}% | "
                    f"Vol: ${item.get('volume_24h', 0):,.0f}"
                )
            return _brand_text("\n".join(lines))
        
        return _brand_text("⚠️ Market data providers unavailable.")
        
    except Exception as e:
        logger.error(f"Top movers failed: {e}")
        return _brand_text(f"❌ Error fetching movers: {str(e)[:100]}")


# ----- Technical Analysis -----
def calculate_volatility(
    df: pd.DataFrame,
    annualize: bool = True
) -> Dict[str, float]:
    """
    Calculate various volatility metrics from price data.
    """
    try:
        closes = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(closes) < 2:
            return {"error": "Insufficient data"}
        
        # Daily returns
        returns = closes.pct_change().dropna()
        
        # Standard deviation
        std = returns.std()
        
        # Annualized volatility (assuming hourly data, 24*365 hours/year)
        if annualize:
            annual_vol = std * (24 * 365) ** 0.5
        else:
            annual_vol = std
        
        # Other metrics
        var_95 = np.percentile(returns, 5)  # Value at Risk (95%)
        max_drawdown = (closes / closes.cummax() - 1).min()
        
        return {
            "daily_volatility": round(std, 6),
            "annualized_volatility": round(annual_vol, 4),
            "var_95": round(var_95, 4),
            "max_drawdown": round(max_drawdown, 4)
        }
        
    except Exception as e:
        logger.error(f"Volatility calculation failed: {e}")
        return {"error": str(e)}


def get_leverage_suggestion(
    symbol: str,
    df: Optional[pd.DataFrame] = None
) -> LeverageSuggestion:
    """
    Suggest futures leverage based on volatility analysis.
    Conservative approach: lower leverage for higher volatility.
    """
    try:
        # Get data if not provided
        if df is None:
            if not fetch_klines_multi:
                return LeverageSuggestion(
                    symbol=symbol,
                    daily_volatility=0,
                    suggested_leverage=1,
                    risk_level="unknown",
                    error="Data fetcher unavailable"
                )
            df = fetch_klines_multi(symbol, "1h", limit=168, exchange="binance")  # 1 week of hourly
        
        if df is None or df.empty:
            return LeverageSuggestion(
                symbol=symbol,
                daily_volatility=0,
                suggested_leverage=1,
                risk_level="no_data"
            )
        
        # Calculate volatility
        vol_metrics = calculate_volatility(df, annualize=True)
        daily_vol = vol_metrics.get("daily_volatility", 0.01)
        
        # Leverage tiers based on daily volatility
        if daily_vol < 0.005:  # < 0.5% daily
            leverage = 50
            risk_level = "ultra_low"
            max_position = 0.5  # 50% of balance
        elif daily_vol < 0.01:  # < 1% daily
            leverage = 25
            risk_level = "low"
            max_position = 0.3
        elif daily_vol < 0.02:  # < 2% daily
            leverage = 10
            risk_level = "moderate"
            max_position = 0.2
        elif daily_vol < 0.04:  # < 4% daily
            leverage = 5
            risk_level = "high"
            max_position = 0.1
        else:  # > 4% daily
            leverage = 2
            risk_level = "extreme"
            max_position = 0.05
        
        return LeverageSuggestion(
            symbol=symbol,
            daily_volatility=round(daily_vol * 100, 4),  # As percentage
            suggested_leverage=leverage,
            risk_level=risk_level,
            max_position_size=max_position
        )
        
    except Exception as e:
        logger.error(f"Leverage suggestion failed: {e}")
        return LeverageSuggestion(
            symbol=symbol,
            daily_volatility=0,
            suggested_leverage=1,
            risk_level="error"
        )


def analyze_momentum(
    symbol: str,
    interval: str = "1h",
    limit: int = 200,
    exchange: str = "binance"
) -> MomentumAnalysis:
    """
    Comprehensive momentum and candle pattern analysis.
    """
    try:
        if not fetch_klines_multi:
            raise RuntimeError("Klines fetcher unavailable")
        
        df = fetch_klines_multi(symbol, interval, limit=limit, exchange=exchange)
        if df is None or len(df) < 50:
            raise ValueError("Insufficient data for analysis")
        
        # Ensure numeric
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Calculate indicators
        closes = df["close"]
        
        # EMAs
        ema20 = closes.ewm(span=20, adjust=False, min_periods=1).mean()
        ema50 = closes.ewm(span=50, adjust=False, min_periods=1).mean()
        
        # RSI
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=1).mean()
        rs = gain / (loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        # Volatility
        returns = closes.pct_change().dropna()
        vol_24h = returns.std() * (24 ** 0.5) * 100  # As percentage
        
        # Current values
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        # EMA cross determination
        cur_ema20, cur_ema50 = ema20.iloc[-1], ema50.iloc[-1]
        prev_ema20, prev_ema50 = ema20.iloc[-2] if len(ema20) > 1 else cur_ema20, ema50.iloc[-2] if len(ema50) > 1 else cur_ema50
        
        if prev_ema20 <= prev_ema50 and cur_ema20 > cur_ema50:
            ema_cross = "golden_cross"
        elif prev_ema20 >= prev_ema50 and cur_ema20 < cur_ema50:
            ema_cross = "death_cross"
        elif cur_ema20 > cur_ema50:
            ema_cross = "bullish"
        else:
            ema_cross = "bearish"
        
        # Candle pattern detection
        pattern = detect_candle_pattern(df)
        
        # Momentum score calculation
        score = 0.5
        
        # EMA contribution
        if ema_cross == "golden_cross":
            score += 0.2
        elif ema_cross == "death_cross":
            score -= 0.2
        elif ema_cross == "bullish":
            score += 0.1
        else:
            score -= 0.1
        
        # MACD contribution
        cur_hist = macd_hist.iloc[-1]
        prev_hist = macd_hist.iloc[-2] if len(macd_hist) > 1 else cur_hist
        if cur_hist > 0:
            score += 0.1
            if cur_hist > prev_hist:
                score += 0.05
        else:
            score -= 0.1
        
        # RSI contribution
        cur_rsi = rsi.iloc[-1]
        if pd.isna(cur_rsi):
            cur_rsi = 50
        if cur_rsi < 30:
            score -= 0.1  # Oversold, potential bounce down
        elif cur_rsi > 70:
            score -= 0.05  # Overbought, potential pullback
        elif 40 <= cur_rsi <= 60:
            score += 0.02  # Neutral zone
        
        # Pattern contribution
        if pattern:
            if "bullish" in pattern.lower():
                score += 0.15
            elif "bearish" in pattern.lower():
                score -= 0.15
        
        # Clamp score
        score = max(0.01, min(0.99, score))
        
        # Trend strength
        if score > 0.7:
            trend = "strong_bullish"
        elif score > 0.55:
            trend = "bullish"
        elif score < 0.3:
            trend = "strong_bearish"
        elif score < 0.45:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return MomentumAnalysis(
            symbol=symbol,
            interval=interval,
            current_price=round(float(last["close"]), 8),
            ema_cross=ema_cross,
            rsi=round(float(cur_rsi), 2) if not pd.isna(cur_rsi) else None,
            macd_histogram=round(float(cur_hist), 6),
            momentum_score=round(score, 2),
            volatility_24h=round(float(vol_24h), 4),
            candle_pattern=pattern,
            trend_strength=trend
        )
        
    except Exception as e:
        logger.error(f"Momentum analysis failed: {e}")
        return MomentumAnalysis(
            symbol=symbol,
            interval=interval,
            current_price=0,
            ema_cross="error",
            rsi=None,
            macd_histogram=0,
            momentum_score=0,
            volatility_24h=0,
            candle_pattern=f"error: {str(e)[:50]}",
            trend_strength="unknown"
        )


def detect_candle_pattern(df: pd.DataFrame, lookback: int = 5) -> Optional[str]:
    """
    Detect common candlestick patterns in recent data.
    """
    try:
        if len(df) < 3:
            return None
        
        patterns = []
        
        for i in range(-lookback, 0):
            if i < -len(df) + 1:
                continue
            
            curr = df.iloc[i]
            prev = df.iloc[i-1] if i > -len(df) + 1 else curr
            
            # Body and range calculations
            curr_body = abs(curr["close"] - curr["open"])
            curr_range = curr["high"] - curr["low"]
            prev_body = abs(prev["close"] - prev["open"])
            
            # Doji (small body)
            if curr_range > 0 and curr_body / curr_range < 0.1:
                patterns.append("doji")
            
            # Hammer (small body at top, long lower wick)
            if curr_range > 0:
                lower_wick = min(curr["open"], curr["close"]) - curr["low"]
                upper_wick = curr["high"] - max(curr["open"], curr["close"])
                if lower_wick > 2 * curr_body and upper_wick < curr_body:
                    patterns.append("hammer" if curr["close"] > curr["open"] else "hanging_man")
            
            # Engulfing patterns
            if prev_body > 0:
                # Bullish engulfing
                if (prev["close"] < prev["open"] and  # Previous red
                    curr["close"] > curr["open"] and    # Current green
                    curr["open"] < prev["close"] and    # Open below prev close
                    curr["close"] > prev["open"]):      # Close above prev open
                    patterns.append("bullish_engulfing")
                
                # Bearish engulfing
                elif (prev["close"] > prev["open"] and  # Previous green
                      curr["close"] < curr["open"] and    # Current red
                      curr["open"] > prev["close"] and    # Open above prev close
                      curr["close"] < prev["open"]):      # Close below prev open
                    patterns.append("bearish_engulfing")
            
            # Morning/Evening star (simplified)
            if i > -len(df) + 2:
                prev2 = df.iloc[i-2]
                if (prev2["close"] < prev2["open"] and  # Long red
                    abs(prev["close"] - prev["open"]) < 0.3 * abs(prev2["close"] - prev2["open"]) and  # Small
                    curr["close"] > curr["open"] and curr["close"] > (prev2["open"] + prev2["close"])/2):
                    patterns.append("morning_star")
                                elif (prev2["close"] > prev2["open"] and
                      abs(prev["close"] - prev["open"]) < 0.3 * abs(prev2["close"] - prev2["open"]) and
                      curr["close"] < curr["open"] and curr["close"] < (prev2["open"] + prev2["close"])/2):
                    patterns.append("evening_star")
        
        # Return most recent significant pattern
        priority = ["bullish_engulfing", "bearish_engulfing", "morning_star", 
                   "evening_star", "hammer", "hanging_man", "doji"]
        
        for p in priority:
            if p in patterns:
                return p
        
        return patterns[-1] if patterns else None
        
    except Exception as e:
        logger.debug(f"Pattern detection error: {e}")
        return None


# ----- Visualization -----
def generate_price_chart(
    symbol: str,
    interval: str = "1h",
    points: int = 60,
    exchange: str = "binance",
    indicators: bool = True
) -> Optional[bytes]:
    """
    Generate price chart using QuickChart or local generation.
    """
    try:
        if not fetch_klines_multi:
            return None
        
        df = fetch_klines_multi(symbol, interval, limit=points, exchange=exchange)
        if df is None or df.empty:
            return None
        
        closes = pd.to_numeric(df["close"], errors="coerce").dropna().tolist()
        if len(closes) < 2:
            return None
        
        # Use market_providers sparkline if available
        if generate_sparkline:
            return generate_sparkline(closes, label=symbol)
        
        # Fallback to QuickChart
        labels = list(range(len(closes)))
        
        chart_config = {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": symbol,
                    "data": closes,
                    "borderColor": "rgb(255, 215, 0)",  # Gold
                    "backgroundColor": "rgba(255, 215, 0, 0.1)",
                    "fill": True,
                    "pointRadius": 0,
                    "borderWidth": 2,
                    "tension": 0.4
                }]
            },
            "options": {
                "plugins": {
                    "legend": {"display": True, "labels": {"color": "white"}},
                    "title": {
                        "display": True,
                        "text": f"{symbol} {interval}",
                        "color": "white",
                        "font": {"size": 16}
                    }
                },
                "scales": {
                    "x": {"display": False},
                    "y": {
                        "display": True,
                        "grid": {"color": "rgba(255,255,255,0.1)"},
                        "ticks": {"color": "white"}
                    }
                }
            }
        }
        
        params = {
            "c": json.dumps(chart_config),
            "width": 900,
            "height": 400,
            "devicePixelRatio": 2,
            "format": "png",
            "backgroundColor": "#0a0c12"
        }
        
        response = _safe_request(QUICKCHART_URL, params=params, timeout=15)
        if response:
            return response.content
        
        return None
        
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        return None


# ----- AI Integration -----
def get_ai_market_insight(
    symbol: str,
    exchanges: Optional[List[str]] = None,
    include_technical: bool = True
) -> str:
    """
    Generate AI-powered market insight with optional technical context.
    """
    try:
        if not HAS_AI or not get_market_brief:
            return _brand_text("🤖 AI service currently unavailable.")
        
        # Gather context
        snapshot = get_multi_exchange_snapshot(symbol, exchanges)
        context = f"Symbol: {symbol}\nAverage Price: {snapshot.average}\nExchange Prices: {snapshot.prices}\n"
        
        if include_technical:
            momentum = analyze_momentum(symbol)
            context += f"""
Technical Context:
- Momentum Score: {momentum.momentum_score}
- Trend: {momentum.trend_strength}
- RSI: {momentum.rsi}
- EMA Cross: {momentum.ema_cross}
- Pattern: {momentum.candle_pattern}
- 24h Volatility: {momentum.volatility_24h}%
"""
        
        # Get AI analysis
        brief = get_market_brief([symbol])
        return _brand_text(f"🤖 AI Market Insight for {symbol}:\n\n{brief}")
        
    except Exception as e:
        logger.error(f"AI insight failed: {e}")
        return _brand_text(f"⚠️ AI analysis error: {str(e)[:100]}")


# ----- Pro Report Generation -----
def generate_pro_report(
    symbols: List[str],
    exchanges: Optional[List[str]] = None,
    interval: str = "1h",
    include_charts: bool = True,
    include_ai: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive pro market report.
    """
    exchanges = exchanges or DEFAULT_EXCHANGES
    timestamp = datetime.utcnow()
    
    report = {
        "timestamp": timestamp.isoformat(),
        "summary": "",
        "symbols": {},
        "fear_greed": None,
        "chart": None,
        "ai_insight": None
    }
    
    try:
        # Fear & Greed
        fg = get_fear_greed_index()
        report["fear_greed"] = fg
        
        # Analyze each symbol
        ranked = []
        for symbol in symbols:
            try:
                # Multi-exchange data
                snapshot = get_multi_exchange_snapshot(symbol, exchanges)
                
                # Technical analysis
                momentum = analyze_momentum(symbol, interval, exchange=exchanges[0])
                
                # Leverage suggestion
                leverage = get_leverage_suggestion(symbol)
                
                # Store
                report["symbols"][symbol] = {
                    "snapshot": snapshot.__dict__,
                    "momentum": momentum.__dict__,
                    "leverage": leverage.__dict__
                }
                
                # For ranking
                ranked.append((symbol, momentum.momentum_score))
                
            except Exception as e:
                logger.warning(f"Report generation failed for {symbol}: {e}")
                report["symbols"][symbol] = {"error": str(e)}
        
        # Sort by momentum score
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s[0] for s in ranked[:3]]
        
        # Build summary text
        lines = [
            f"📊 Pro Market Report — {timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"😐 Market Sentiment: {fg.get('text', 'N/A')}" if fg else "",
            "",
            "🏆 Top Opportunities:"
        ]
        
        for i, (sym, score) in enumerate(ranked[:5], 1):
            data = report["symbols"].get(sym, {})
            momentum = data.get("momentum", {})
            snapshot = data.get("snapshot", {})
            
            emoji = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
            lines.append(
                f"{i}. {emoji} {sym} | Score: {score:.2f} | "
                f"Price: ${snapshot.get('average', 'N/A')} | "
                f"Trend: {momentum.get('trend_strength', 'N/A')}"
            )
        
        report["summary"] = _brand_text("\n".join(lines))
        
        # Generate chart for top symbol
        if include_charts and top_symbols:
            chart_bytes = generate_price_chart(top_symbols[0], interval, points=60)
            if chart_bytes:
                report["chart"] = chart_bytes
        
        # AI insight for top symbol
        if include_ai and top_symbols and HAS_AI:
            report["ai_insight"] = get_ai_market_insight(top_symbols[0], exchanges)
        
    except Exception as e:
        logger.error(f"Pro report generation failed: {e}")
        report["summary"] = _brand_text(f"❌ Report generation error: {str(e)[:200]}")
        report["error"] = str(e)
    
    return report


# ----- Backward Compatibility -----
def fear_and_greed_index() -> str:
    """Legacy compatibility wrapper."""
    result = get_fear_greed_index()
    return result.get("text", "Fear & Greed data unavailable")

def futures_leverage_suggestion(symbol: str, df=None) -> Dict:
    """Legacy compatibility wrapper."""
    result = get_leverage_suggestion(symbol, df)
    return {
        "vol": result.daily_volatility,
        "leverage": result.suggested_leverage,
        "risk_level": result.risk_level
    }

def quickchart_price_image(symbol: str, interval: str = "1h", points: int = 30, exchange: str = "binance") -> Optional[bytes]:
    """Legacy compatibility wrapper."""
    return generate_price_chart(symbol, interval, points, exchange)

def ai_market_brief_text(symbol: str, exchanges: Optional[List[str]] = None) -> str:
    """Legacy compatibility wrapper."""
    return get_ai_market_insight(symbol, exchanges)

def momentum_and_candle_analysis(symbol: str, interval: str = "1h", limit: int = 200, exchange: str = "binance") -> Dict:
    """Legacy compatibility wrapper."""
    result = analyze_momentum(symbol, interval, limit, exchange)
    return {
        "symbol": result.symbol,
        "interval": result.interval,
        "close": result.current_price,
        "ema_cross": result.ema_cross,
        "rsi": result.rsi,
        "macd": result.macd_histogram,
        "macd_hist": result.macd_histogram,
        "volatility": result.volatility_24h,
        "candle_pattern": result.candle_pattern,
        "momentum_score": result.momentum_score
    }

def pro_market_report(symbols: List[str], exchanges: Optional[List[str]] = None, interval: str = "1h", top_n: int = 3) -> Dict:
    """Legacy compatibility wrapper."""
    return generate_pro_report(symbols, exchanges, interval)

def top_gainers_pairs(limit: int = 5, pairs: Optional[List[str]] = None, exchanges: Optional[List[str]] = None) -> str:
    """Legacy compatibility wrapper."""
    return get_top_movers(limit, pairs)


# ----- Exports -----
__all__ = [
    # New style
    "get_fear_greed_index",
    "get_multi_exchange_snapshot",
    "get_top_movers",
    "get_leverage_suggestion",
    "analyze_momentum",
    "detect_candle_pattern",
    "generate_price_chart",
    "get_ai_market_insight",
    "generate_pro_report",
    # Legacy compatibility
    "fear_and_greed_index",
    "futures_leverage_suggestion",
    "quickchart_price_image",
    "ai_market_brief_text",
    "momentum_and_candle_analysis",
    "pro_market_report",
    "top_gainers_pairs",
    # Dataclasses
    "MarketSnapshot",
    "LeverageSuggestion",
    "MomentumAnalysis"
]


# ----- Self-Test -----
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("Pro Features Module Self-Test")
    print("=" * 60)
    
    # Test Fear & Greed
    print("\n1. Fear & Greed Index:")
    fg = get_fear_greed_index()
    print(f"   {fg.get('text', 'N/A')}")
    
    # Test snapshot
    print("\n2. Multi-Exchange Snapshot (BTCUSDT):")
    snapshot = get_multi_exchange_snapshot("BTCUSDT", ["binance"])
    print(f"   Prices: {snapshot.prices}")
    print(f"   Average: {snapshot.average}")
    
    # Test momentum
    print("\n3. Momentum Analysis (BTCUSDT):")
    if HAS_MARKET_PROVIDERS:
        momentum = analyze_momentum("BTCUSDT", "1h", limit=100)
        print(f"   Score: {momentum.momentum_score}")
        print(f"   Trend: {momentum.trend_strength}")
        print(f"   Pattern: {momentum.candle_pattern}")
    else:
        print("   Skipped (market_providers unavailable)")
    
    # Test leverage
    print("\n4. Leverage Suggestion (BTCUSDT):")
    if HAS_MARKET_PROVIDERS:
        lev = get_leverage_suggestion("BTCUSDT")
        print(f"   Daily Vol: {lev.daily_volatility}%")
        print(f"   Suggested Leverage: {lev.suggested_leverage}x")
        print(f"   Risk Level: {lev.risk_level}")
    else:
        print("   Skipped (market_providers unavailable)")
    
    print("\n" + "=" * 60)
    print("Self-test complete")
