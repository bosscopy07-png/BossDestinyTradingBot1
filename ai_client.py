# ai_client.py
"""
AI-powered analysis client for Destiny Trading Empire.
Provides OpenAI integration, exchange streaming, image analysis, and signal generation.
"""

import os
import time
import logging
import base64
import json
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger("ai_client")
logger.setLevel(logging.INFO)

# Optional dependencies with graceful degradation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import ccxt.pro as ccxt_pro
    HAS_CCXT_PRO = True
except ImportError:
    ccxt_pro = None
    HAS_CCXT_PRO = False

try:
    from binance import ThreadedWebsocketManager
    from binance.client import Client as BinanceRestClient
    HAS_BINANCE = True
except ImportError:
    ThreadedWebsocketManager = None
    BinanceRestClient = None
    HAS_BINANCE = False


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class AIAnalysisResult:
    success: bool
    analysis: Optional[str] = None
    raw_response: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class TradingSignal:
    signal: SignalType
    confidence: float
    reasons: List[str]
    indicators: Dict[str, float]
    price: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


# ----- OpenAI Client Initialization -----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

openai_client = None

if OPENAI_API_KEY:
    try:
        # Try modern client first
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI modern client initialized")
    except ImportError:
        # Fall back to legacy
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            openai_client = openai
            logger.info("OpenAI legacy client initialized")
        except Exception as e:
            logger.error(f"OpenAI legacy init failed: {e}")
    except Exception as e:
        logger.error(f"OpenAI modern init failed: {e}")
else:
    logger.warning("OPENAI_API_KEY not set - AI features disabled")

HAS_OPENAI = openai_client is not None


# ----- AI Text Analysis -----
def get_market_brief(pairs: List[str], timeframe: str = "1h") -> str:
    """
    Generate AI market brief for given pairs.
    Simplified interface for bot_runner integration.
    """
    if not HAS_OPENAI:
        return "AI analysis unavailable - no API key configured."
    
    prompt = f"""Analyze the current market conditions for: {', '.join(pairs)}.
Timeframe: {timeframe}

Provide:
1. Overall market sentiment (bullish/bearish/neutral)
2. Key support/resistance levels for major pairs
3. Any notable patterns or setups
4. Risk considerations

Keep it concise (3-4 sentences per pair)."""

    result = ai_analysis_text(prompt, max_tokens=600)
    if result.success:
        return result.analysis
    return f"Analysis failed: {result.error}"


def ai_analysis_text(
    prompt: str,
    retries: int = 3,
    temperature: float = 0.6,
    max_tokens: int = 400,
    system_prompt: Optional[str] = None
) -> AIAnalysisResult:
    """
    Generate AI-based trading analysis with retry logic.
    """
    if not HAS_OPENAI:
        return AIAnalysisResult(success=False, error="OpenAI client not configured")
    
    system_msg = system_prompt or "You are an expert crypto trading analyst. Provide concise, actionable insights."
    
    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"AI request attempt {attempt}/{retries}")
            
            # Modern client
            if hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
                resp = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                text = resp.choices[0].message.content.strip()
                return AIAnalysisResult(success=True, analysis=text, raw_response=resp)
            
            # Legacy client
            elif hasattr(openai_client, "ChatCompletion"):
                resp = openai_client.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                text = resp.choices[0].message.content.strip()
                return AIAnalysisResult(success=True, analysis=text, raw_response=resp)
            
            else:
                return AIAnalysisResult(success=False, error="Unsupported OpenAI interface")
                
        except Exception as e:
            logger.warning(f"AI attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"AI failed after {retries} attempts: {e}")
                return AIAnalysisResult(success=False, error=str(e))
    
    return AIAnalysisResult(success=False, error="Max retries exceeded")


# ----- Exchange Streaming -----
class ExchangeStreamer:
    """Unified interface for real-time exchange data streaming."""
    
    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None
    ):
        self.exchange_id = exchange_id.lower()
        self.api_key = api_key or os.getenv(f"{self.exchange_id.upper()}_API_KEY")
        self.secret = secret or os.getenv(f"{self.exchange_id.upper()}_SECRET")
        
        self._ccxt_client = None
        self._ws_manager = None
        self._rest_client = None
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize available clients based on dependencies."""
        # Try ccxt.pro first
        if HAS_CCXT_PRO:
            try:
                exchange_class = getattr(ccxt_pro, self.exchange_id, None)
                if exchange_class:
                    self._ccxt_client = exchange_class({
                        "apiKey": self.api_key,
                        "secret": self.secret,
                        "enableRateLimit": True,
                        "options": {"defaultType": "spot"}
                    })
                    logger.info(f"Initialized ccxt.pro client for {self.exchange_id}")
                    return
            except Exception as e:
                logger.warning(f"ccxt.pro init failed: {e}")
        
        # Fallback to python-binance for Binance
        if self.exchange_id == "binance" and HAS_BINANCE:
            try:
                self._rest_client = BinanceRestClient(
                    api_key=self.api_key,
                    api_secret=self.secret
                )
                logger.info("Initialized python-binance REST client")
            except Exception as e:
                logger.error(f"Binance client init failed: {e}")
    
    async def stream_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100):
        """
        Async generator for real-time OHLCV data using ccxt.pro.
        """
        if not HAS_CCXT_PRO or not self._ccxt_client:
            raise RuntimeError("ccxt.pro not available for streaming")
        
        symbol_formatted = symbol.replace("/", "").upper()
        if "/" not in symbol:
            symbol_formatted = f"{symbol[:-4]}/{symbol[-4:]}" if symbol.endswith("USDT") else f"{symbol}/USDT"
        
        try:
            while True:
                ohlcv = await self._ccxt_client.watch_ohlcv(
                    symbol_formatted,
                    timeframe=timeframe,
                    limit=limit
                )
                
                if HAS_PANDAS:
                    df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    yield df
                else:
                    yield ohlcv
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
    
    def start_websocket_klines(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict], None]
    ):
        """
        Start threaded WebSocket for Binance kline data.
        """
        if not HAS_BINANCE:
            raise RuntimeError("python-binance not available")
        
        if self._ws_manager is not None:
            logger.warning("WebSocket already running, stopping previous")
            self.stop_websocket()
        
        self._ws_manager = ThreadedWebsocketManager()
        self._ws_manager.start()
        
        def _handle_message(msg):
            try:
                k = msg.get("k", {})
                if not k:
                    return
                
                candle = {
                    "symbol": symbol,
                    "timestamp": k.get("t"),
                    "open": float(k.get("o", 0)),
                    "high": float(k.get("h", 0)),
                    "low": float(k.get("l", 0)),
                    "close": float(k.get("c", 0)),
                    "volume": float(k.get("v", 0)),
                    "is_final": k.get("x", False),
                    "interval": interval
                }
                callback(candle)
                
            except Exception as e:
                logger.error(f"WebSocket message handler error: {e}")
        
        self._ws_manager.start_kline_socket(
            callback=_handle_message,
            symbol=symbol.upper(),
            interval=interval
        )
        logger.info(f"Started kline WebSocket for {symbol} ({interval})")
    
    def stop_websocket(self):
        """Stop WebSocket manager."""
        if self._ws_manager:
            try:
                self._ws_manager.stop()
                self._ws_manager = None
                logger.info("WebSocket stopped")
            except Exception as e:
                logger.error(f"Error stopping WebSocket: {e}")
    
    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV via REST (fallback method).
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas required for DataFrame output")
        
        # Try ccxt first
        if self._ccxt_client and hasattr(self._ccxt_client, "fetch_ohlcv"):
            try:
                # Use sync ccxt for REST
                import ccxt
                sync_exchange = getattr(ccxt, self.exchange_id)({
                    "apiKey": self.api_key,
                    "secret": self.secret,
                    "enableRateLimit": True
                })
                
                ohlcv = sync_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                return df
                
            except Exception as e:
                logger.warning(f"ccxt REST fetch failed: {e}")
        
        # Try python-binance for Binance
        if self.exchange_id == "binance" and self._rest_client:
            try:
                klines = self._rest_client.get_klines(
                    symbol=symbol.upper(),
                    interval=timeframe,
                    limit=limit
                )
                
                df = pd.DataFrame(klines, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore"
                ])
                
                df = df[["open_time", "open", "high", "low", "close", "volume"]]
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df[["open", "high", "low", "close", "volume"]] = df[
                    ["open", "high", "low", "close", "volume"]
                ].astype(float)
                
                return df
                
            except Exception as e:
                logger.error(f"Binance REST fetch failed: {e}")
        
        return None


# ----- Image Analysis -----
class ImageAnalyzer:
    """Analyze trading chart images using AI or computer vision."""
    
    def __init__(self, model: str = OPENAI_MODEL):
        self.model = model
        self.has_openai = HAS_OPENAI
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")
    
    def analyze_with_ai(self, image_path: str, context: str = "") -> AIAnalysisResult:
        """
        Analyze chart image using OpenAI vision capabilities.
        """
        if not self.has_openai:
            return AIAnalysisResult(success=False, error="OpenAI not configured")
        
        try:
            b64_image = self._encode_image(image_path)
            
            prompt = (
                "Analyze this cryptocurrency price chart. Identify: "
                "1) Trend direction, 2) Key support/resistance levels, "
                "3) Visible patterns (if any), 4) Suggested action (BUY/SELL/HOLD) with reasoning. "
                f"Additional context: {context}" if context else ""
            )
            
            # Check for vision capabilities
            if hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Vision-capable model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                analysis = resp.choices[0].message.content.strip()
                return AIAnalysisResult(success=True, analysis=analysis, raw_response=resp)
            
            return AIAnalysisResult(success=False, error="Vision capabilities not available")
            
        except Exception as e:
            logger.error(f"AI image analysis failed: {e}")
            return AIAnalysisResult(success=False, error=str(e))
    
    def analyze_with_cv(self, image_path: str, expected_bars: int = 30) -> Dict[str, Any]:
        """
        Analyze chart using OpenCV (fallback method).
        Detects candlestick bars and extracts approximate OHLC values.
        """
        if not (HAS_CV2 and HAS_NUMPY):
            return {"success": False, "error": "OpenCV/NumPy not available"}
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": "Failed to load image"}
            
            height, width = img.shape[:2]
            
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive threshold for candlestick detection
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations to group candle components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(
                morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter and sort by x position
            bars = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter by size (candlestick dimensions)
                if w > 2 and h > 15 and y > height * 0.1 and y < height * 0.9:
                    bars.append({
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "center_y": y + h / 2
                    })
            
            # Sort by x position and take last N
            bars = sorted(bars, key=lambda b: b["x"])[-expected_bars:]
            
            if len(bars) < 5:
                return {"success": False, "error": f"Only detected {len(bars)} bars, need more data"}
            
            # Convert y-positions to normalized prices (inverted: top=high, bottom=low)
            def y_to_price(y):
                return 1.0 - (y / height)
            
            ohlcv_data = []
            for bar in bars:
                top = bar["y"]
                bottom = bar["y"] + bar["h"]
                
                # Approximate OHLC from bar geometry
                # This is a rough estimation - real implementation would need more sophisticated logic
                high = y_to_price(top)
                low = y_to_price(bottom)
                mid = (high + low) / 2
                
                ohlcv_data.append({
                    "open": mid,
                    "high": high,
                    "low": low,
                    "close": mid,  # Can't determine from single bar without color analysis
                    "x": bar["x"]
                })
            
            return {
                "success": True,
                "bars_detected": len(ohlcv_data),
                "data": ohlcv_data,
                "image_dimensions": {"width": width, "height": height}
            }
            
        except Exception as e:
            logger.error(f"CV analysis failed: {e}")
            return {"success": False, "error": str(e)}


# ----- Signal Generation -----
class SignalGenerator:
    """Generate trading signals from price data using technical analysis."""
    
    def __init__(self):
        self.available = HAS_PANDAS and HAS_NUMPY
        if not self.available:
            logger.warning("SignalGenerator requires pandas and numpy")
    
        @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.001)  # Avoid div by zero
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """MACD indicator."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal from OHLCV DataFrame.
        """
        if not self.available:
            logger.error("Cannot generate signal - pandas/numpy unavailable")
            return None
        
        if len(df) < 50:
            logger.warning(f"Insufficient data: {len(df)} rows, need 50+")
            return None
        
        try:
            # Ensure numeric
            close = pd.to_numeric(df["close"], errors="coerce")
            if close.isna().any():
                close = close.fillna(method="ffill")
            
            # Calculate indicators
            ema20 = self.ema(close, 20)
            ema50 = self.ema(close, 50)
            sma50 = self.sma(close, 50)
            rsi14 = self.rsi(close, 14)
            macd_line, signal_line, hist = self.macd(close)
            
            last = {
                "close": close.iloc[-1],
                "ema20": ema20.iloc[-1],
                "ema50": ema50.iloc[-1],
                "sma50": sma50.iloc[-1],
                "rsi14": rsi14.iloc[-1],
                "macd": macd_line.iloc[-1],
                "macd_signal": signal_line.iloc[-1],
                "macd_hist": hist.iloc[-1],
                "prev_close": close.iloc[-2] if len(close) > 1 else close.iloc[-1],
                "prev_ema20": ema20.iloc[-2] if len(ema20) > 1 else ema20.iloc[-1],
                "prev_ema50": ema50.iloc[-2] if len(ema50) > 1 else ema50.iloc[-1],
            }
            
            # Determine signal
            signal_type = SignalType.HOLD
            confidence = 0.5
            reasons = []
            
            # Trend analysis (EMA cross)
            bullish_trend = last["ema20"] > last["ema50"]
            bearish_trend = last["ema20"] < last["ema50"]
            ema_cross_up = last["ema20"] > last["ema50"] and last["prev_ema20"] <= last["prev_ema50"]
            ema_cross_down = last["ema20"] < last["ema50"] and last["prev_ema20"] >= last["prev_ema50"]
            
            if ema_cross_up:
                signal_type = SignalType.LONG
                confidence += 0.2
                reasons.append("EMA20 crossed above EMA50 (golden cross)")
            elif ema_cross_down:
                signal_type = SignalType.SHORT
                confidence += 0.2
                reasons.append("EMA20 crossed below EMA50 (death cross)")
            elif bullish_trend:
                signal_type = SignalType.LONG
                confidence += 0.1
                reasons.append("Bullish trend (EMA20 > EMA50)")
            elif bearish_trend:
                signal_type = SignalType.SHORT
                confidence += 0.1
                reasons.append("Bearish trend (EMA20 < EMA50)")
            
            # RSI conditions
            if last["rsi14"] < 30:
                if signal_type in [SignalType.HOLD, SignalType.LONG]:
                    signal_type = SignalType.LONG
                    confidence += 0.15
                reasons.append(f"RSI oversold ({last['rsi14']:.1f})")
            elif last["rsi14"] > 70:
                if signal_type in [SignalType.HOLD, SignalType.SHORT]:
                    signal_type = SignalType.SHORT
                    confidence += 0.15
                reasons.append(f"RSI overbought ({last['rsi14']:.1f})")
            
            # MACD confirmation
            if last["macd_hist"] > 0 and hist.iloc[-2] <= 0:
                if signal_type == SignalType.LONG:
                    confidence += 0.1
                reasons.append("MACD histogram turned positive")
            elif last["macd_hist"] < 0 and hist.iloc[-2] >= 0:
                if signal_type == SignalType.SHORT:
                    confidence += 0.1
                reasons.append("MACD histogram turned negative")
            
            # Price vs SMA
            if last["close"] > last["sma50"]:
                if signal_type == SignalType.LONG:
                    confidence += 0.05
                reasons.append("Price above SMA50")
            else:
                if signal_type == SignalType.SHORT:
                    confidence += 0.05
                reasons.append("Price below SMA50")
            
            # Cap confidence
            confidence = min(0.95, max(0.1, confidence))
            
            return TradingSignal(
                signal=signal_type,
                confidence=round(confidence, 2),
                reasons=reasons,
                indicators={
                    "ema20": round(last["ema20"], 2),
                    "ema50": round(last["ema50"], 2),
                    "sma50": round(last["sma50"], 2),
                    "rsi14": round(last["rsi14"], 2),
                    "macd": round(last["macd"], 4),
                    "macd_signal": round(last["macd_signal"], 4)
                },
                price=round(last["close"], 8)
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return None


# ----- High-Level Interface -----
def analyze_image_and_generate_signal(
    image_path: str,
    use_ai: bool = True,
    price_scale: float = 100.0
) -> Dict[str, Any]:
    """
    Complete pipeline: image -> analysis -> signal.
    """
    result = {
        "success": False,
        "image_path": image_path,
        "ai_analysis": None,
        "cv_analysis": None,
        "signal": None,
        "error": None
    }
    
    # Step 1: AI Analysis (if available and requested)
    if use_ai and HAS_OPENAI:
        analyzer = ImageAnalyzer()
        ai_result = analyzer.analyze_with_ai(image_path)
        result["ai_analysis"] = ai_result.__dict__ if hasattr(ai_result, "__dict__") else ai_result
        
        if ai_result.success:
            logger.info("AI image analysis successful")
            # Could parse AI output for signal, but we'll use CV for structured data
    
    # Step 2: Computer Vision for structured data
    analyzer = ImageAnalyzer()
    cv_result = analyzer.analyze_with_cv(image_path)
    result["cv_analysis"] = cv_result
    
    if not cv_result.get("success"):
        result["error"] = cv_result.get("error", "CV analysis failed")
        return result
    
    # Step 3: Generate signal from CV data
    if HAS_PANDAS and cv_result.get("data"):
        try:
            # Convert to DataFrame
            data = cv_result["data"]
            df = pd.DataFrame(data)
            
            # Scale prices
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col] * price_scale
            
            # Generate signal
            generator = SignalGenerator()
            signal = generator.generate_signal(df)
            
            if signal:
                result["signal"] = {
                    "type": signal.signal.value,
                    "confidence": signal.confidence,
                    "reasons": signal.reasons,
                    "indicators": signal.indicators,
                    "price": signal.price
                }
                result["success"] = True
            else:
                result["error"] = "Signal generation returned None"
                
        except Exception as e:
            result["error"] = f"Signal generation failed: {e}"
            logger.error(result["error"])
    
    return result


# Convenience function for bot_runner
def get_ai_signal_context(symbol: str, market_data: Dict) -> str:
    """Generate AI context for a specific signal."""
    prompt = f"""Trading Setup Analysis for {symbol}:
Current Price: {market_data.get('price', 'N/A')}
Signal: {market_data.get('signal', 'N/A')}
Confidence: {market_data.get('confidence', 'N/A')}
Timeframe: {market_data.get('interval', '1h')}

Provide a brief 2-sentence market context and risk warning."""
    
    result = ai_analysis_text(prompt, max_tokens=200)
    return result.analysis if result.success else "AI analysis unavailable"
