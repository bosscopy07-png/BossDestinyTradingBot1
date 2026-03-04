# signal_engine.py
"""
Signal generation engine for Destiny Trading Empire.
Provides multi-exchange analysis, technical indicators, and automated scanning.
"""

import os
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger("signal_engine")
logger.setLevel(logging.INFO)

# Optional imports with graceful fallback
try:
    from market_providers import fetch_klines_multi, get_realtime_price
    HAS_MARKET_PROVIDERS = True
except ImportError:
    fetch_klines_multi = None
    get_realtime_price = None
    HAS_MARKET_PROVIDERS = False
    logger.debug("market_providers not available")

try:
    from ai_client import ai_analysis_text, get_ai_signal_context
    HAS_AI = True
except ImportError:
    ai_analysis_text = None
    get_ai_signal_context = None
    HAS_AI = False
    logger.debug("ai_client not available")

try:
    from image_utils import build_signal_image
    HAS_IMAGE_UTILS = True
except ImportError:
    build_signal_image = None
    HAS_IMAGE_UTILS = False


# ----- Constants -----
class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class SignalResult:
    symbol: str
    interval: str
    signal: SignalType
    entry: Optional[float]
    sl: Optional[float]
    tp1: Optional[float]
    confidence: float
    reasons: List[str]
    exchange: str
    timestamp: str
    consensus: bool = False
    per_exchange: Optional[List[Dict]] = None
    ai_explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "signal": self.signal.value,
            "entry": self.entry,
            "sl": self.sl,
            "tp1": self.tp1,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
            "consensus": self.consensus,
            "per_exchange": self.per_exchange,
            "ai_explanation": self.ai_explanation
        }


# Configuration
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_EXCHANGES = ["binance", "bybit", "kucoin", "okx"]
DEFAULT_INTERVAL = "1h"
MIN_DATA_POINTS = 30
ATR_MULTIPLIER = 1.5

# Thread-safe session for HTTP requests
_session: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    """Get or create session with retry logic."""
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        _session.mount("https://", HTTPAdapter(max_retries=retries))
    return _session


# ----- Technical Indicators -----
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    Uses Wilder's smoothing method.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    
    # Use exponential moving average for Wilder's RSI
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD, Signal line, and Histogram.
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """
    Calculate Bollinger Bands.
    """
    sma = calculate_sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=1).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return upper_band, sma, lower_band


# ----- Data Fetching -----
def fetch_klines_binance(
    symbol: str,
    interval: str,
    limit: int = 300
) -> Optional[pd.DataFrame]:
    """
    Fetch klines from Binance REST API as fallback.
    """
    try:
        session = _get_session()
        response = session.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": limit
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, list) or len(data) == 0:
            logger.debug(f"Empty response from Binance for {symbol}")
            return None
        
        # Parse Binance kline format
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Convert types
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        
        return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]]
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Binance request failed for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Binance klines parse error for {symbol}: {e}")
        return None


def fetch_klines_with_fallback(
    symbol: str,
    interval: str,
    exchange: str,
    limit: int = 300
) -> Optional[pd.DataFrame]:
    """
    Fetch klines with cascading fallback.
    """
    # Try market_providers first
    if HAS_MARKET_PROVIDERS and fetch_klines_multi:
        try:
            df = fetch_klines_multi(symbol, interval, limit=limit, exchange=exchange)
            if df is not None and not df.empty:
                logger.debug(f"Got data from market_providers for {symbol} on {exchange}")
                return df
        except Exception as e:
            logger.debug(f"market_providers failed: {e}")
    
    # Fallback to direct Binance
    if exchange == "binance":
        return fetch_klines_binance(symbol, interval, limit)
    
    # For other exchanges, try Binance as universal fallback
    logger.warning(f"Falling back to Binance for {symbol} (original: {exchange})")
    return fetch_klines_binance(symbol, interval, limit)


# ----- Signal Scoring -----
def score_dataframe(df: pd.DataFrame) -> Tuple[float, List[str]]:
    """
    Calculate signal score (0.0 to 1.0) and reasons from OHLCV data.
    """
    reasons: List[str] = []
    
    try:
        if df is None or len(df) < MIN_DATA_POINTS:
            return 0.0, ["insufficient_data"]
        
        closes = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(closes) < MIN_DATA_POINTS:
            return 0.0, ["insufficient_valid_closes"]
        
        # Calculate indicators
        ema_fast = calculate_ema(closes, 9)
        ema_slow = calculate_ema(closes, 21)
        rsi_val = calculate_rsi(closes, 14)
        macd_line, signal_line, macd_hist = calculate_macd(closes)
        
        # Get current and previous values
        cur_ef = float(ema_fast.iloc[-1])
        cur_es = float(ema_slow.iloc[-1])
        prev_ef = float(ema_fast.iloc[-2]) if len(ema_fast) > 1 else cur_ef
        prev_es = float(ema_slow.iloc[-2]) if len(ema_slow) > 1 else cur_es
        
        cur_rsi = float(rsi_val.iloc[-1])
        cur_macd_hist = float(macd_hist.iloc[-1])
        prev_macd_hist = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else cur_macd_hist
        
        # Base score
        score = 0.5
        
        # EMA Crossover (strong signal)
        if prev_ef <= prev_es and cur_ef > cur_es:
            score += 0.20
            reasons.append("ema_golden_cross")
        elif prev_ef >= prev_es and cur_ef < cur_es:
            score -= 0.20
            reasons.append("ema_death_cross")
        elif cur_ef > cur_es:
            score += 0.08
            reasons.append("bullish_trend")
        else:
            score -= 0.08
            reasons.append("bearish_trend")
        
        # MACD Histogram momentum
        if cur_macd_hist > 0:
            score += 0.06
            reasons.append("macd_positive")
            if cur_macd_hist > prev_macd_hist:
                score += 0.04
                reasons.append("macd_increasing")
        else:
            score -= 0.06
            reasons.append("macd_negative")
        
        # RSI conditions
        if cur_rsi < 30:
            score += 0.10
            reasons.append("rsi_oversold")
        elif cur_rsi > 70:
            score -= 0.10
            reasons.append("rsi_overbought")
        elif 40 <= cur_rsi <= 60:
            reasons.append("rsi_neutral")
        
        # Price momentum (short term)
        lookback = min(5, len(closes) - 1)
        if lookback > 0:
            momentum = (closes.iloc[-1] - closes.iloc[-lookback]) / closes.iloc[-lookback]
            if momentum > 0.02:  # 2% up
                score += 0.05
                reasons.append("strong_momentum_up")
            elif momentum < -0.02:  # 2% down
                score -= 0.05
                reasons.append("strong_momentum_down")
            elif momentum > 0:
                score += 0.02
                reasons.append("weak_momentum_up")
            else:
                score -= 0.02
                reasons.append("weak_momentum_down")
        
        # Candlestick patterns (if OHLC available)
        if all(col in df.columns for col in ["open", "high", "low"]):
            try:
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                
                last_body = abs(last["close"] - last["open"])
                prev_body = abs(prev["close"] - prev["open"])
                last_range = last["high"] - last["low"]
                
                # Bullish engulfing
                if (prev["close"] < prev["open"] and  # Previous red
                    last["close"] > last["open"] and   # Current green
                    last["open"] < prev["close"] and   # Open below prev close
                    last["close"] > prev["open"]):     # Close above prev open
                    score += 0.12
                    reasons.append("bullish_engulfing")
                
                # Bearish engulfing
                elif (prev["close"] > prev["open"] and  # Previous green
                      last["close"] < last["open"] and   # Current red
                      last["open"] > prev["close"] and   # Open above prev close
                      last["close"] < prev["open"]):     # Close below prev open
                    score -= 0.12
                    reasons.append("bearish_engulfing")
                
                # Doji (indecision)
                if last_body < (last_range * 0.1):
                    reasons.append("doji_indecision")
                    score *= 0.9  # Reduce confidence
                
            except Exception as e:
                logger.debug(f"Candlestick analysis failed: {e}")
        
        # Bollinger Band position
        if len(closes) >= 20:
            try:
                upper, middle, lower = calculate_bollinger_bands(closes)
                last_price = closes.iloc[-1]
                bb_position = (last_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-12)
                
                if bb_position < 0.1:  # Near lower band
                    score += 0.06
                    reasons.append("bb_near_lower")
                elif bb_position > 0.9:  # Near upper band
                    score -= 0.06
                    reasons.append("bb_near_upper")
            except Exception as e:
                logger.debug(f"Bollinger analysis failed: {e}")
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        
        return round(score, 4), reasons if reasons else ["no_clear_pattern"]
        
    except Exception as e:
        logger.exception("Error in score_dataframe")
        return 0.0, [f"scoring_error: {str(e)[:50]}"]


# ----- Exchange Analysis -----
def analyze_single_exchange(
    symbol: str,
    interval: str,
    exchange: str
) -> Dict[str, Any]:
    """
    Analyze a single exchange for trading signals.
    """
    try:
        # Fetch data
        df = fetch_klines_with_fallback(symbol, interval, exchange, limit=300)
        
        if df is None or df.empty or len(df) < MIN_DATA_POINTS:
            return {
                "exchange": exchange,
                "error": "insufficient_data",
                "symbol": symbol.upper(),
                "interval": interval
            }
        
        # Ensure numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Drop rows with NaN in critical columns
        df = df.dropna(subset=["close"])
        if len(df) < MIN_DATA_POINTS:
            return {
                "exchange": exchange,
                "error": "too_many_invalid_values",
                "symbol": symbol.upper(),
                "interval": interval
            }
        
        # Calculate score
        score, reasons = score_dataframe(df)
        
        # Determine signal type
        last_price = float(df["close"].iloc[-1])
        prev_price = float(df["close"].iloc[-2]) if len(df) > 1 else last_price
        
        if score >= 0.68:
            signal = SignalType.LONG if last_price > prev_price else SignalType.SHORT
        elif score <= 0.32:
            signal = SignalType.SHORT if last_price < prev_price else SignalType.LONG
        else:
            signal = SignalType.HOLD
        
        # Calculate ATR for stop loss / take profit
        try:
            atr_series = calculate_atr(df)
            atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else last_price * 0.01
        except Exception:
            atr_val = last_price * 0.01  # Fallback: 1% of price
        
        # Set levels based on signal
        if signal == SignalType.LONG:
            sl = round(max(0.0, last_price - (ATR_MULTIPLIER * atr_val)), 8)
            tp1 = round(last_price + (ATR_MULTIPLIER * atr_val), 8)
        elif signal == SignalType.SHORT:
            sl = round(last_price + (ATR_MULTIPLIER * atr_val), 8)
            tp1 = round(max(0.0, last_price - (ATR_MULTIPLIER * atr_val)), 8)
        else:
            sl = None
            tp1 = None
        
        return {
            "exchange": exchange,
            "symbol": symbol.upper(),
            "interval": interval,
            "signal": signal.value,
            "entry": round(last_price, 8),
            "sl": sl,
            "tp1": tp1,
            "confidence": score,
            "reasons": reasons,
            "data_points": len(df),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"Analysis failed for {symbol} on {exchange}")
        return {
            "exchange": exchange,
            "symbol": symbol.upper(),
            "interval": interval,
            "error": f"analysis_failed: {str(e)[:100]}"
        }


# ----- Multi-Exchange Signal Generation -----
def generate_signal_multi(
    symbol: str = "BTCUSDT",
    interval: str = DEFAULT_INTERVAL,
    exchanges: Optional[List[str]] = None,
    min_confidence: float = 0.65,
    use_ai: bool = False,
    require_consensus: bool = False
) -> SignalResult:
    """
    Generate trading signal by analyzing multiple exchanges.
    Uses consensus and confidence scoring for final signal.
    """
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES
    
    per_exchange_results: List[Dict[str, Any]] = []
    valid_results: List[Dict[str, Any]] = []
    
    # Analyze all exchanges
    for exchange in exchanges:
        try:
            result = analyze_single_exchange(symbol, interval, exchange)
            per_exchange_results.append(result)
            
            if "error" not in result:
                valid_results.append(result)
                
        except Exception as e:
            logger.error(f"Exchange analysis exception for {exchange}: {e}")
            per_exchange_results.append({
                "exchange": exchange,
                "error": f"exception: {str(e)[:50]}",
                "symbol": symbol.upper(),
                "interval": interval
            })
    
    # Handle no valid data
    if not valid_results:
        logger.warning(f"No valid data for {symbol} on any exchange")
        return SignalResult(
            symbol=symbol.upper(),
            interval=interval,
            signal=SignalType.HOLD,
            entry=None,
            sl=None,
            tp1=None,
            confidence=0.0,
            reasons=["no_data_on_any_exchange"],
            exchange="none",
            timestamp=datetime.utcnow().isoformat(),
            per_exchange=per_exchange_results
        )
    
    # Count votes
    votes = {SignalType.LONG.value: 0, SignalType.SHORT.value: 0, SignalType.HOLD.value: 0}
    confidences = []
    
    for r in valid_results:
        sig = r.get("signal", "HOLD")
        votes[sig] = votes.get(sig, 0) + 1
        confidences.append(r.get("confidence", 0.0))
    
    # Find best result (highest confidence)
    best_result = max(valid_results, key=lambda x: x.get("confidence", 0.0))
    
    # Calculate consensus
    total_votes = sum(votes.values())
    consensus_threshold = 0.6  # 60% agreement
    
    has_consensus = False
    consensus_signal = SignalType.HOLD
    
    for sig_type, count in votes.items():
        if count / total_votes >= consensus_threshold and sig_type in (SignalType.LONG.value, SignalType.SHORT.value):
            has_consensus = True
            consensus_signal = SignalType(sig_type)
            break
    
    # Determine final signal
    if require_consensus and not has_consensus:
        final_signal = SignalType.HOLD
        final_confidence = best_result.get("confidence", 0.0) * 0.5  # Penalize for no consensus
    else:
        final_signal = SignalType(best_result.get("signal", "HOLD"))
        final_confidence = best_result.get("confidence", 0.0)

    # Boost confidence if consensus matches best signal
    if has_consensus and consensus_signal == final_signal:
        final_confidence = min(0.98, final_confidence + 0.1)
    
    # Get AI explanation if requested
    ai_explanation = None
    if use_ai and HAS_AI and final_signal != SignalType.HOLD:
        try:
            market_data = {
                "symbol": symbol,
                "price": best_result.get("entry"),
                "signal": final_signal.value,
                "confidence": final_confidence,
                "interval": interval,
                "reasons": best_result.get("reasons", [])
            }
            ai_explanation = get_ai_signal_context(symbol, market_data)
        except Exception as e:
            logger.warning(f"AI explanation failed: {e}")
    
    return SignalResult(
        symbol=best_result.get("symbol", symbol.upper()),
        interval=best_result.get("interval", interval),
        signal=final_signal,
        entry=best_result.get("entry"),
        sl=best_result.get("sl"),
        tp1=best_result.get("tp1"),
        confidence=round(final_confidence, 4),
        reasons=best_result.get("reasons", []),
        exchange=best_result.get("exchange", "unknown"),
        timestamp=datetime.utcnow().isoformat(),
        consensus=has_consensus,
        per_exchange=per_exchange_results,
        ai_explanation=ai_explanation
    )


def generate_signal(
    symbol: str = "BTCUSDT",
    interval: str = DEFAULT_INTERVAL
) -> SignalResult:
    """
    Simple interface for single-exchange signal generation.
    """
    return generate_signal_multi(symbol, interval, exchanges=["binance"])


# ----- Auto Scanner -----
class SignalScanner:
    """Thread-safe automatic signal scanner."""
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._callback: Optional[Callable[[SignalResult], None]] = None
        self._running = False
    
    def register_callback(self, callback: Callable[[SignalResult], None]) -> None:
        """Register callback for strong signals."""
        self._callback = callback
        logger.info(f"Signal callback registered: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def _scan_loop(
        self,
        pairs: List[str],
        interval: str,
        exchanges: List[str],
        min_confidence: float,
        poll_seconds: int,
        use_ai: bool
    ):
        """Main scanner loop."""
        logger.info(f"Scanner started: {len(pairs)} pairs, interval={interval}, poll={poll_seconds}s")
        
        while not self._stop_event.is_set():
            cycle_start = time.time()
            
            try:
                for pair in pairs:
                    if self._stop_event.is_set():
                        break
                    
                    # Generate signal
                    result = generate_signal_multi(
                        pair,
                        interval,
                        exchanges=exchanges,
                        min_confidence=min_confidence,
                        use_ai=use_ai
                    )
                    
                    # Check if strong signal
                    is_strong = (
                        result.signal in (SignalType.LONG, SignalType.SHORT) and
                        result.confidence >= min_confidence
                    )
                    
                    if is_strong:
                        logger.info(
                            f"Strong signal: {pair} {result.signal.value} "
                            f"conf={result.confidence:.2f} via {result.exchange}"
                        )
                        
                        if self._callback:
                            try:
                                self._callback(result)
                            except Exception as e:
                                logger.exception(f"Signal callback failed: {e}")
                        else:
                            logger.warning("No callback registered for strong signal")
                    
                    # Small delay between pairs to avoid rate limits
                    time.sleep(0.5)
                
            except Exception as e:
                logger.exception(f"Scanner cycle error: {e}")
            
            # Wait for next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, poll_seconds - elapsed)
            
            # Check stop event during sleep
            if sleep_time > 0 and not self._stop_event.is_set():
                self._stop_event.wait(timeout=sleep_time)
        
        self._running = False
        logger.info("Scanner stopped")
    
    def start(
        self,
        pairs: List[str],
        interval: str = "1h",
        exchanges: Optional[List[str]] = None,
        min_confidence: float = 0.75,
        poll_seconds: int = 300,
        use_ai: bool = False
    ) -> bool:
        """Start the scanner thread."""
        with self._lock:
            if self._running:
                logger.warning("Scanner already running")
                return False
            
            self._stop_event.clear()
            self._running = True
            
            exs = exchanges or DEFAULT_EXCHANGES
            
            self._thread = threading.Thread(
                target=self._scan_loop,
                args=(pairs, interval, exs, min_confidence, poll_seconds, use_ai),
                daemon=True,
                name="SignalScanner"
            )
            self._thread.start()
            
            logger.info(f"Scanner thread started: {self._thread.ident}")
            return True
    
    def stop(self, timeout: float = 10.0) -> bool:
        """Stop the scanner gracefully."""
        with self._lock:
            if not self._running or not self._thread:
                return False
            
            logger.info("Stopping scanner...")
            self._stop_event.set()
            self._thread.join(timeout=timeout)
            
            if self._thread.is_alive():
                logger.warning(f"Scanner did not stop within {timeout}s")
                return False
            
            self._running = False
            self._thread = None
            logger.info("Scanner stopped successfully")
            return True
    
    @property
    def is_running(self) -> bool:
        """Check if scanner is active."""
        return self._running and self._thread is not None and self._thread.is_alive()


# Global scanner instance
_scanner = SignalScanner()

# Convenience functions for backward compatibility
def register_send_callback(fn: Callable[[SignalResult], None]) -> None:
    _scanner.register_callback(fn)

def start_auto_scanner(
    pairs: List[str],
    interval: str = "1h",
    exchanges: Optional[List[str]] = None,
    min_confidence: float = 0.75,
    poll_seconds: int = 300,
    use_ai: bool = False
) -> bool:
    return _scanner.start(pairs, interval, exchanges, min_confidence, poll_seconds, use_ai)

def stop_auto_scanner(timeout: float = 10.0) -> bool:
    return _scanner.stop(timeout)

def is_scanner_running() -> bool:
    return _scanner.is_running


# ----- Bot Integration -----
def generate_and_send_signal(
    bot,
    pair: str,
    interval: str,
    chat_id: int,
    auto: bool = False
) -> bool:
    """
    Generate signal and send via bot (for bot_runner integration).
    This is the main interface expected by bot_runner.py.
    """
    try:
        # Generate signal
        result = generate_signal_multi(pair, interval)
        
        # Build message
        signal_emoji = {
            SignalType.LONG: "🟢",
            SignalType.SHORT: "🔴",
            SignalType.HOLD: "⚪"
        }.get(result.signal, "⚪")
        
        lines = [
            f"{signal_emoji} Destiny Trading Empire — Signal Alert",
            f"",
            f"ID: S{int(time.time())}",
            f"Pair: {result.symbol} | Timeframe: {result.interval}",
            f"Signal: {result.signal.value}",
            f"Exchange: {result.exchange}",
            f"Consensus: {'Yes' if result.consensus else 'No'}",
            f"",
            f"Entry: {result.entry:.8f}" if result.entry else "Entry: N/A",
            f"Stop Loss: {result.sl:.8f}" if result.sl else "SL: N/A",
            f"Take Profit: {result.tp1:.8f}" if result.tp1 else "TP: N/A",
            f"",
            f"Confidence: {int(result.confidence * 100)}%",
            f"Reasons: {', '.join(result.reasons[:5])}",
        ]
        
        if result.ai_explanation:
            lines.extend(["", f"🤖 AI Insight: {result.ai_explanation[:200]}..."])
        
        caption = "\n".join(lines)
        
        # Try to build image
        image_buf = None
        if HAS_IMAGE_UTILS and build_signal_image:
            try:
                signal_dict = result.to_dict()
                signal_dict["suggested_risk_usd"] = "Calculated by risk manager"
                image_buf = build_signal_image(signal_dict)
            except Exception as e:
                logger.warning(f"Image build failed: {e}")
        
        # Send via bot
        from telebot import types
        
        kb = types.InlineKeyboardMarkup(row_width=2)
        kb.add(
            types.InlineKeyboardButton("📷 Link PnL", callback_data=f"link_S{int(time.time())}"),
            types.InlineKeyboardButton("🤖 AI Details", callback_data=f"ai_{result.symbol}")
        )
        
        if image_buf:
            bot.send_photo(
                chat_id,
                image_buf,
                caption=f"<i>{caption}</i>\n\n— <b>Destiny Trading Empire Bot 💎</b>",
                reply_markup=kb,
                parse_mode="HTML"
            )
        else:
            bot.send_message(
                chat_id,
                f"{caption}\n\n— <b>Destiny Trading Empire Bot 💎</b>",
                reply_markup=kb,
                parse_mode="HTML"
            )
        
        return True
        
    except Exception as e:
        logger.exception(f"generate_and_send_signal failed for {pair}")
        try:
            bot.send_message(
                chat_id,
                f"⚠️ Failed to generate signal for {pair}\n\n— <b>Destiny Trading Empire Bot 💎</b>",
                parse_mode="HTML"
            )
        except Exception:
            pass
        return False


# ----- Exports -----
__all__ = [
    "generate_signal_multi",
    "generate_signal",
    "SignalResult",
    "SignalType",
    "start_auto_scanner",
    "stop_auto_scanner",
    "is_scanner_running",
    "register_send_callback",
    "generate_and_send_signal",
    "SignalScanner"
]


# ----- Self-Test -----
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 50)
    print("Signal Engine Self-Test")
    print("=" * 50)
    
    # Test single signal
    print("\n1. Testing single signal generation (BTCUSDT):")
    result = generate_signal("BTCUSDT", "1h")
    print(f"   Signal: {result.signal.value}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Entry: {result.entry}")
    print(f"   Reasons: {result.reasons[:3]}")
    
    # Test multi-exchange
    print("\n2. Testing multi-exchange signal:")
    result = generate_signal_multi("ETHUSDT", "1h", exchanges=["binance", "bybit"])
    print(f"   Best: {result.exchange} | {result.signal.value} | {result.confidence}")
    print(f"   Per-exchange results: {len(result.per_exchange or [])}")
    
    # Test scanner (don't start, just check)
    print("\n3. Scanner status:")
    print(f"   Running: {is_scanner_running()}")
    
    print("\n" + "=" * 50)
    print("Self-test complete")
            
