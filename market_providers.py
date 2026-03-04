# market_providers.py
"""
Unified multi-exchange market data provider for trading bots.

Features:
- Multi-exchange klines via ccxt (preferred) or REST fallbacks
- Real-time ticker data with failover across exchanges
- Trending pairs aggregation
- Multi-timeframe analysis with signal scoring
- Chart generation via QuickChart
"""

import os
import time
import json
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from decimal import Decimal

import requests
import pandas as pd
import numpy as np
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress noisy logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
pd.options.mode.chained_assignment = None

logger = logging.getLogger("market_providers")
logger.setLevel(logging.INFO)

# Optional imports with graceful degradation
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    ccxt = None
    HAS_CCXT = False
    logger.info("ccxt not available, using REST fallbacks only")

try:
    from image_utils import create_brand_image
except ImportError:
    create_brand_image = None
    logger.debug("image_utils not available for branded outputs")

# ----- Configuration -----
QUICKCHART_URL = "https://quickchart.io/chart"

EXCHANGE_ENDPOINTS = {
    "binance": {
        "klines": "https://api.binance.com/api/v3/klines",
        "ticker_24h": "https://api.binance.com/api/v3/ticker/24hr",
        "ticker_price": "https://api.binance.com/api/v3/ticker/price"
    },
    "bybit": {
        "klines": "https://api.bybit.com/v5/market/kline",
        "tickers": "https://api.bybit.com/v5/market/tickers"
    },
    "kucoin": {
        "klines": "https://api.kucoin.com/api/v1/market/candles",
        "tickers": "https://api.kucoin.com/api/v1/market/allTickers"
    },
    "okx": {
        "klines": "https://www.okx.com/api/v5/market/candles",
        "tickers": "https://www.okx.com/api/v5/market/tickers"
    }
}

DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
DEFAULT_PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

# ----- Data Classes -----
@dataclass
class TickerData:
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    exchange: str
    raw_data: Dict[str, Any]


@dataclass  
class KlineData:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def ohlcv(self) -> Tuple[float, float, float, float, float]:
        return (self.open, self.high, self.low, self.close, self.volume)


# ----- HTTP Session Management -----
def get_session() -> requests.Session:
    """Create configured session with retries and optional proxy."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "DestinyTradingEmpireBot/1.0",
        "Accept": "application/json"
    })
    
    # Proxy support
    proxy = os.getenv("PROXY_URL")
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
        logger.debug(f"Using proxy: {proxy}")
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session


# Singleton session for reuse
_session: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = get_session()
    return _session


# ----- Symbol Normalization -----
def normalize_symbol(symbol: str, exchange: str) -> str:
    """
    Normalize symbol format for specific exchange.
    e.g., BTCUSDT -> BTC-USDT for OKX/KuCoin
    """
    clean = symbol.upper().replace("/", "").replace("-", "")
    exchange = exchange.lower()
    
    if exchange in ("okx", "kucoin"):
        # Insert dash before USDT, USD, USDC
        for suffix in ["USDT", "USDC", "USD", "BUSD"]:
            if clean.endswith(suffix):
                return f"{clean[:-len(suffix)]}-{suffix}"
    
    return clean


def get_ccxt_symbol(symbol: str) -> str:
    """Convert to CCXT format (BTC/USDT)."""
    clean = symbol.upper().replace("/", "").replace("-", "")
    if "USDT" in clean and not clean.endswith("/USDT"):
        return clean.replace("USDT", "/USDT")
    if "USD" in clean and "USDT" not in clean:
        return clean.replace("USD", "/USD")
    return f"{clean}/USDT"


# ----- Timeframe Mapping -----
KUCOIN_INTERVALS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1hour", "2h": "2hour", "4h": "4hour",
    "6h": "6hour", "8h": "8hour", "12h": "12hour", "1d": "1day"
}

def map_interval(interval: str, exchange: str) -> str:
    """Map standard interval to exchange-specific format."""
    if exchange.lower() == "kucoin":
        return KUCOIN_INTERVALS.get(interval, interval)
    return interval


# ----- OHLCV Parsing -----
def parse_ohlcv_array(data: List[List], exchange: str = "unknown") -> Optional[pd.DataFrame]:
    """
    Parse various OHLCV array formats into standardized DataFrame.
    Handles: [ts, o, h, l, c, v], [ts, o, h, l, c, v, ...], etc.
    """
    if not data or not isinstance(data, list):
        return None
    
    try:
        # Detect timestamp unit
        sample_ts = float(data[0][0])
        unit = "ms" if sample_ts > 1e12 else "s"
        
        # Standardize to 6 columns
        df_data = []
        for row in data:
            if len(row) >= 6:
                df_data.append([
                    float(row[0]),  # timestamp
                    float(row[1]),  # open
                    float(row[2]),  # high
                    float(row[3]),  # low
                    float(row[4]),  # close
                    float(row[5])   # volume
                ])
        
        if not df_data:
            return None
        
        df = pd.DataFrame(df_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit)
        
        return df
        
    except Exception as e:
        logger.error(f"OHLCV parsing failed for {exchange}: {e}")
        return None


def parse_binance_klines(raw_data: List[Dict]) -> Optional[pd.DataFrame]:
    """Parse Binance-specific kline format."""
    try:
        df = pd.DataFrame(raw_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base", 
            "taker_buy_quote", "ignore"
        ])
        
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        
        return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
        
    except Exception as e:
        logger.error(f"Binance kline parse error: {e}")
        return None


# ----- Kline Fetchers -----
def fetch_klines_ccxt(symbol: str, interval: str, limit: int = 300, exchange: str = "binance") -> Optional[pd.DataFrame]:
    """Fetch klines using CCXT library."""
    if not HAS_CCXT:
        return None
    
    try:
        exchange_class = getattr(ccxt, exchange.lower())
        ex = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        ex.load_markets()
        
        ccxt_symbol = get_ccxt_symbol(symbol)
        ohlcv = ex.fetch_ohlcv(ccxt_symbol, timeframe=interval, limit=limit)
        
        return parse_ohlcv_array(ohlcv, exchange)
        
    except Exception as e:
        logger.debug(f"CCXT fetch failed for {symbol} on {exchange}: {e}")
        return None


def fetch_klines_binance(symbol: str, interval: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch klines from Binance REST API."""
    try:
        session = _get_session()
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        response = session.get(
            EXCHANGE_ENDPOINTS["binance"]["klines"],
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        return parse_binance_klines(response.json())
        
    except Exception as e:
        logger.debug(f"Binance klines failed: {e}")
        return None


def fetch_klines_bybit(symbol: str, interval: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch klines from Bybit REST API."""
    try:
        session = _get_session()
        params = {
            "category": "spot",
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 200)
        }
        
        response = session.get(
            EXCHANGE_ENDPOINTS["bybit"]["klines"],
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        if data.get("retCode") != 0:
            logger.debug(f"Bybit error: {data.get('retMsg')}")
            return None
            
        ohlcv = data.get("result", {}).get("list", [])
        # Bybit returns [timestamp, open, high, low, close, volume, turnover]
        return parse_ohlcv_array(ohlcv, "bybit")
        
    except Exception as e:
        logger.debug(f"Bybit klines failed: {e}")
        return None


def fetch_klines_kucoin(symbol: str, interval: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch klines from KuCoin REST API."""
    try:
        session = _get_session()
        ku_symbol = normalize_symbol(symbol, "kucoin")
        ku_interval = map_interval(interval, "kucoin")
        
        params = {
            "symbol": ku_symbol,
            "type": ku_interval
        }
        
        response = session.get(
            EXCHANGE_ENDPOINTS["kucoin"]["klines"],
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        if not data.get("data"):
            return None
            
        # KuCoin returns [[timestamp, open, close, high, low, volume, turnover], ...]
        # Need to reorder to standard OHLCV
        raw = data["data"]
        standardized = [[r[0], r[1], r[3], r[4], r[2], r[5]] for r in raw]  # Reorder H/L
        return parse_ohlcv_array(standardized, "kucoin")
        
    except Exception as e:
        logger.debug(f"KuCoin klines failed: {e}")
        return None


def fetch_klines_okx(symbol: str, interval: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch klines from OKX REST API."""
    try:
        session = _get_session()
        okx_symbol = normalize_symbol(symbol, "okx")
        
        params = {
            "instId": okx_symbol,
            "bar": interval,
            "limit": min(limit, 300)
        }
        
        response = session.get(
            EXCHANGE_ENDPOINTS["okx"]["klines"],
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        if data.get("code") != "0":
            logger.debug(f"OKX error: {data.get('msg')}")
            return None
            
        ohlcv = data.get("data", [])
        # OKX returns [timestamp, open, high, low, close, volume, turnover]
        return parse_ohlcv_array(ohlcv, "okx")
        
    except Exception as e:
        logger.debug(f"OKX klines failed: {e}")
        return None


def fetch_klines_multi(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 300,
    exchange: Optional[str] = None,
    prefer_ccxt: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch klines with cascading failover across exchanges.
    
    Priority: CCXT (if enabled and prefer_ccxt) -> Specified exchange -> All exchanges
    """
    symbol = symbol.upper()
    interval = interval.lower()
    
    # Try CCXT first if preferred
    if HAS_CCXT and prefer_ccxt and exchange:
        result = fetch_klines_ccxt(symbol, interval, limit, exchange)
        if result is not None and not result.empty:
            logger.debug(f"CCXT success: {symbol} from {exchange}")
            return result
    
    # Exchange-specific fetchers map
    fetchers = {
        "binance": fetch_klines_binance,
        "bybit": fetch_klines_bybit,
        "kucoin": fetch_klines_kucoin,
        "okx": fetch_klines_okx
    }
    
    # Try specific exchange if requested
    if exchange and exchange.lower() in fetchers:
        result = fetchers[exchange.lower()](symbol, interval, limit)
        if result is not None and not result.empty:
            logger.debug(f"REST success: {symbol} from {exchange}")
            return result
        logger.warning(f"Exchange {exchange} failed for {symbol}, trying fallbacks...")
    
    # Cascade through all exchanges
    for exch_name, fetcher in fetchers.items():
        if exch_name == exchange:
            continue  # Already tried
        try:
            result = fetcher(symbol, interval, limit)
            if result is not None and not result.empty:
                logger.info(f"Fallback success: {symbol} from {exch_name}")
                return result
        except Exception as e:
            logger.debug(f"Fallback {exch_name} failed: {e}")
    
    logger.error(f"All exchanges failed for {symbol} {interval}")
    return None


# ----- Real-time Price Fetching -----
def fetch_price_binance(symbol: str) -> Optional[float]:
    """Fetch current price from Binance."""
    try:
        session = _get_session()
        response = session.get(
            EXCHANGE_ENDPOINTS["binance"]["ticker_price"],
            params={"symbol": symbol.upper()},
            timeout=5
        )
        response.raise_for_status()
        return float(response.json()["price"])
    except Exception as e:
        logger.debug(f"Binance price failed: {e}")
        return None


def fetch_price_bybit(symbol: str) -> Optional[float]:
    """Fetch current price from Bybit."""
    try:
        session = _get_session()
        response = session.get(
            EXCHANGE_ENDPOINTS["bybit"]["tickers"],
            params={"category": "spot", "symbol": symbol.upper()},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            return float(data["result"]["list"][0]["lastPrice"])
        return None
    except Exception as e:
        logger.debug(f"Bybit price failed: {e}")
        return None


def fetch_price_okx(symbol: str) -> Optional[float]:
    """Fetch current price from OKX."""
    try:
        session = _get_session()
        okx_symbol = normalize_symbol(symbol, "okx")
        response = session.get(
            f"{EXCHANGE_ENDPOINTS['okx']['tickers']}?instType=SPOT&instId={okx_symbol}",
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "0" and data.get("data"):
            return float(data["data"][0]["last"])
        return None
    except Exception as e:
        logger.debug(f"OKX price failed: {e}")
        return None


def get_realtime_price(symbol: str) -> Optional[float]:
    """
    Get real-time price with failover across exchanges.
    Returns first successful price or None if all fail.
    """
    symbol = symbol.upper()
    
    fetchers = [
        ("binance", fetch_price_binance),
        ("bybit", fetch_price_bybit),
        ("okx", fetch_price_okx)
    ]
    
    for name, fetcher in fetchers:
        try:
            price = fetcher(symbol)
            if price is not None and price > 0:
                logger.debug(f"Price for {symbol} from {name}: {price}")
                return price
        except Exception as e:
            logger.debug(f"{name} price fetch failed: {e}")
    
    logger.warning(f"Could not get price for {symbol} from any exchange")
    return None


# ----- Ticker & Trending -----
def fetch_binance_tickers() -> List[TickerData]:
    """Fetch 24h tickers from Binance."""
    try:
        session = _get_session()
        response = session.get(EXCHANGE_ENDPOINTS["binance"]["ticker_24h"], timeout=8)
        response.raise_for_status()
        
        raw = response.json()
        result = []
        for t in raw:
            try:
                result.append(TickerData(
                    symbol=t["symbol"],
                    price=float(t["lastPrice"]),
                    change_24h=float(t["priceChangePercent"]),
                    volume_24h=float(t["quoteVolume"]),
                    exchange="binance",
                    raw_data=t
                ))
            except (KeyError, ValueError):
                continue
        return result
    except Exception as e:
        logger.error(f"Binance tickers failed: {e}")
        return []


def fetch_bybit_tickers() -> List[TickerData]:
    """Fetch tickers from Bybit."""
    try:
        session = _get_session()
        response = session.get(
            EXCHANGE_ENDPOINTS["bybit"]["tickers"],
            params={"category": "spot"},
            timeout=8
        )
        response.raise_for_status()
        
        data = response.json()
        if data.get("retCode") != 0:
            return []
        
        result = []
        for t in data.get("result", {}).get("list", []):
            try:
                result.append(TickerData(
                    symbol=t["symbol"],
                    price=float(t["lastPrice"]),
                    change_24h=float(t["price24hPcnt"]) * 100,
                    volume_24h=float(t["turnover24h"]),
                    exchange="bybit",
                    raw_data=t
                ))
            except (KeyError, ValueError):
                continue
        return result
    except Exception as e:
        logger.error(f"Bybit tickers failed: {e}")
        return []


def fetch_kucoin_tickers() -> List[TickerData]:
    """Fetch tickers from KuCoin."""
    try:
        session = _get_session()
        response = session.get(EXCHANGE_ENDPOINTS["kucoin"]["tickers"], timeout=8)
        response.raise_for_status()
        
        data = response.json()
        result = []
        for t in data.get("data", {}).get("ticker", []):
            try:
                # Normalize symbol
                sym = t["symbol"].replace("-", "")
                change = float(t.get("changeRate", 0)) * 100
                result.append(TickerData(
                    symbol=sym,
                    price=float(t["last"]),
                    change_24h=change,
                    volume_24h=float(t.get("volValue", 0)),
                    exchange="kucoin",
                    raw_data=t
                ))
            except (KeyError, ValueError):
                continue
        return result
    except Exception as e:
        logger.error(f"KuCoin tickers failed: {e}")
        return []


def fetch_okx_tickers() -> List[TickerData]:
    """Fetch tickers from OKX."""
    try:
        session = _get_session()
        response = session.get(
            f"{EXCHANGE_ENDPOINTS['okx']['tickers']}?instType=SPOT",
            timeout=8
        )
        response.raise_for_status()
        
        data = response.json()
        result = []
        for t in data.get("data", []):
            try:
                sym = t["instId"].replace("-", "")
                result.append(TickerData(
                    symbol=sym,
                    price=float(t["last"]),
                    change_24h=float(t["chg24h"]) * 100,
                    volume_24h=float(t["volCcy24h"]),
                    exchange="okx",
                    raw_data=t
                ))
            except (KeyError, ValueError):
                continue
        return result
    except Exception as e:
        logger.error(f"OKX tickers failed: {e}")
        return []


def get_trending_pairs(
    pairs: Optional[List[str]] = None,
    min_volume: float = 1_000_000,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Aggregate trending pairs across all exchanges.
    Returns sorted list by 24h change percentage.
    """
    if pairs is None:
        pairs = os.getenv("PAIRS", ",".join(DEFAULT_PAIRS)).split(",")
    
    pairs_upper = [p.upper() for p in pairs]
    all_tickers: List[TickerData] = []
    
    # Fetch from all exchanges
    fetchers = [fetch_binance_tickers, fetch_bybit_tickers, 
                fetch_kucoin_tickers, fetch_okx_tickers]
    
    for fetcher in fetchers:
        try:
            tickers = fetcher()
            # Filter to requested pairs only
            filtered = [t for t in tickers if t.symbol in pairs_upper]
            all_tickers.extend(filtered)
        except Exception as e:
            logger.error(f"Ticker fetch failed: {e}")
    
    if not all_tickers:
        return []
    
    # Aggregate by symbol (average across exchanges)
    df = pd.DataFrame([
        {
            "symbol": t.symbol,
            "price": t.price,
            "change_24h": t.change_24h,
            "volume_24h": t.volume_24h,
            "exchange": t.exchange
        }
        for t in all_tickers
    ])
    
    # Group and aggregate
    grouped = df.groupby("symbol").agg({
        "price": "mean",
        "change_24h": "mean",
        "volume_24h": "sum",
        "exchange": lambda x: ", ".join(set(x))
    }).reset_index()
    
    # Filter and sort
    grouped = grouped[grouped["volume_24h"] >= min_volume]
    grouped = grouped.sort_values("change_24h", ascending=False).head(top_n)
    
    return grouped.to_dict("records")


def get_trending_pairs_text(pairs: Optional[List[str]] = None) -> str:
    """Get trending pairs as formatted text."""
    try:
        trending = get_trending_pairs(pairs)
        if not trending:
            return "📊 No trending data available."
        
        lines = ["📈 Trending Pairs (24h Change):"]
        for t in trending:
            emoji = "🟢" if t["change_24h"] > 0 else "🔴" if t["change_24h"] < 0 else "⚪"
            lines.append(
                f"{emoji} {t['symbol']:<10} | {t['change_24h']:+.2f}% | "
                f"Vol: ${t['volume_24h']:,.0f} | {t['exchange']}"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Trending text generation failed: {e}")
        return "⚠️ Failed to generate trending pairs."


def get_trending_branded(limit: int = 10) -> Tuple[Optional[BytesIO], str]:
    """
    Generate branded image of trending pairs with sparkline.
    """
    try:
        trending = get_trending_pairs(top_n=limit)
        if not trending:
            return None, "No trending data available."
        
        # Build text lines
        lines = [f"🔥 Top {len(trending)} Trending Pairs (Multi-Exchange)"]
        for t in trending:
            lines.append(
                f"{t['symbol']:<12} | {t['change_24h']:+6.2f}% | "
                f"Vol: ${t['volume_24h']:,.0f}"
            )
        
        # Try to get sparkline for top performer
        chart_bytes = None
        try:
            top_symbol = trending[0]["symbol"]
            klines = fetch_klines_multi(top_symbol, "1h", limit=48)
            if klines is not None and not klines.empty:
                closes = klines["close"].tolist()
                chart_bytes = generate_sparkline(closes, label=top_symbol)
        except Exception as e:
            logger.debug(f"Sparkline generation failed: {e}")
        
        if create_brand_image:
            buf = create_brand_image(lines, chart_img=chart_bytes)
            return buf, f"Top {limit} Trending Pairs"
        
        return None, "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Branded trending failed: {e}")
        return None, "Error generating trending analysis."


# ----- Chart Generation -----
def generate_sparkline(
    values: List[float],
    width: int = 800,
    height: int = 240,
    label: str = ""
) -> Optional[bytes]:
    """
    Generate sparkline chart using QuickChart.io.
    """
    if not values or len(values) < 2:
        return None
    
    try:
        config = {
            "type": "line",
            "data": {
                "labels": list(range(len(values))),
                "datasets": [{
                    "label": label,
                    "data": values,
                    "borderColor": "rgb(255, 215, 0)",
                    "backgroundColor": "rgba(255, 215, 0, 0.1)",
                    "fill": True,
                    "pointRadius": 0,
                    "borderWidth": 2,
                    "tension": 0.4
                }]
            },
            "options": {
                "plugins": {"legend": {"display": False}},
                "scales": {
                    "x": {"display": False},
                    "y": {"display": False}
                },
                "animation": {"duration": 0}
            }
        }
        
        params = {
            "c": json.dumps(config),
            "width": width,
            "height": height,
            "devicePixelRatio": 2,
            "format": "png"
        }
        
        response = _get_session().get(
            QUICKCHART_URL,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        return response.content
        
    except Exception as e:
        logger.error(f"Sparkline generation failed: {e}")
        return None


# ----- Technical Analysis -----
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI for a price series."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return float(100 - (100 / (1 + rs)).iloc[-1])


def calculate_ema(prices: pd.Series, period: int = 20) -> float:
    """Calculate EMA."""
    return float(prices.ewm(span=period, adjust=False).mean().iloc[-1])


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return float(true_range.rolling(period).mean().iloc[-1])


def analyze_pair_multi_timeframes(
    symbol: str,
    timeframes: List[str] = None,
    exchange: str = "binance"
) -> Dict[str, Any]:
    """
    Analyze a trading pair across multiple timeframes.
    Returns signal consensus and technical indicators.
    """
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]
    
    results = {}
    signals = []
    
    for tf in timeframes:
        try:
            df = fetch_klines_multi(symbol, tf, limit=100, exchange=exchange)
            if df is None or df.empty or len(df) < 30:
                continue
            
            # Calculate indicators
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            rsi = calculate_rsi(df["close"])
            ema20 = calculate_ema(df["close"], 20)
            ema50 = calculate_ema(df["close"], 50)
            atr = calculate_atr(df)
            
            # Determine signal
            signal = "HOLD"
            reasons = []
            
            # Trend analysis
            if current["close"] > ema20 > ema50:
                signal = "LONG"
                reasons.append("price_above_ema20_50")
            elif current["close"] < ema20 < ema50:
                signal = "SHORT"
                reasons.append("price_below_ema20_50")
            
            # RSI confirmation
            if rsi < 30 and signal != "SHORT":
                signal = "LONG"
                reasons.append("oversold_rsi")
            elif rsi > 70 and signal != "LONG":
                signal = "SHORT"
                reasons.append("overbought_rsi")
            
            # Momentum
            if current["close"] > current["open"] and prev["close"] > prev["open"]:
                if signal == "LONG":
                    reasons.append("bullish_momentum")
            elif current["close"] < current["open"] and prev["close"] < prev["open"]:
                if signal == "SHORT":
                    reasons.append("bearish_momentum")
            
            results[tf] = {
                "close": float(current["close"]),
                "signal": signal,
                "rsi": round(rsi, 2),
                "ema20": round(ema20, 2),
                "ema50": round(ema50, 2),
                "atr": round(atr, 6),
                "reasons": reasons
            }
            signals.append(signal)
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol} {tf}: {e}")
            continue
    
    # Consensus logic
    if not signals:
        return {"error": "No data available for analysis", "symbol": symbol}
    
    long_count = signals.count("LONG")
    short_count = signals.count("SHORT")
    total = len(signals)
    
    consensus = "HOLD"
    confidence = 0.0
    
    if long_count > short_count and long_count >= total * 0.6:
        consensus = "LONG"
        confidence = long_count / total
    elif short_count > long_count and short_count >= total * 0.6:
        consensus = "SHORT"
        confidence = short_count / total
    
    return {
        "symbol": symbol,
        "timeframes_analyzed": list(results.keys()),
        "analysis": results,
        "consensus_signal": consensus,
        "confidence": round(confidence, 2),
        "recommendation": f"{consensus} (confidence: {confidence:.0%})" if consensus != "HOLD" else "No clear signal"
    }


def detect_strong_signals(
    pairs: List[str] = None,
    min_confidence: float = 0.7,
    exchange: str = "binance"
) -> List[Dict[str, Any]]:
    """
    Scan multiple pairs for strong trading signals.
    """
    if pairs is None:
        pairs = DEFAULT_PAIRS
    
    strong_signals = []
    
    for symbol in pairs:
        try:
            analysis = analyze_pair_multi_timeframes(symbol, exchange=exchange)
            if analysis.get("confidence", 0) >= min_confidence:
                strong_signals.append(analysis)
                logger.info(f"Strong signal detected: {symbol} -> {analysis['consensus_signal']}")
        except Exception as e:
            logger.error(f"Signal detection failed for {symbol}: {e}")
    
    # Sort by confidence
    strong_signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return strong_signals
        
