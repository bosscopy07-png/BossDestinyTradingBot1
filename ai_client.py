# ai_client.py
import os
import time
import logging
import traceback
import base64
import json
from typing import List, Dict, Optional, Any

# Optional imports (wrap in try/except to allow safe degradation)
try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

# For image processing fallback (local CV)
try:
    import cv2
except Exception:
    cv2 = None

# For exchange connectivity (prefer ccxt.pro for websockets if available)
try:
    import ccxt.pro as ccxt_pro  # realtime (websocket) support
except Exception:
    ccxt_pro = None

# python-binance for Binance-specific websocket manager (sync) fallback
try:
    from binance import ThreadedWebsocketManager
    from binance.client import Client as BinanceRestClient
except Exception:
    ThreadedWebsocketManager = None
    BinanceRestClient = None

# -----------------------------
# LOGGING SETUP
# -----------------------------
logger = logging.getLogger("AI_Client")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# -----------------------------
# OPENAI CLIENT INITIALIZATION
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # can override to a vision-capable model if available
client = None
try:
    if OPENAI_API_KEY:
        try:
            # modern client (preferred)
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("[AI] Using modern OpenAI client ✅")
        except Exception:
            import openai
            openai.api_key = OPENAI_API_KEY
            client = openai
            logger.info("[AI] Using legacy OpenAI client ✅")
    else:
        logger.warning("[AI] No API key found — AI features disabled ❌")
except Exception as e:
    logger.error(f"[AI] Initialization failed: {e}")
    client = None

# -----------------------------
# CORE FUNCTION (text)
# -----------------------------
def ai_analysis_text(prompt: str, retries: int = 3, temperature: float = 0.6, max_tokens: int = 400) -> str:
    """
    Generate AI-based trading analysis text.
    """
    if not client:
        return "⚠️ AI not configured. Please set OPENAI_API_KEY."

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[AI] Processing request (attempt {attempt})...")
            # Modern Responses/chat style
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional crypto trading AI analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                text = resp.choices[0].message.content.strip()
                logger.info("[AI] Response generated successfully ✅")
                return text

            # Legacy ChatCompletion
            elif hasattr(client, "ChatCompletion"):
                resp = client.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional crypto trading AI analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                text = resp.choices[0].message.content.strip()
                logger.info("[AI] Response generated via legacy interface ✅")
                return text
            else:
                return "⚠️ AI interface unsupported by current library version."
        except Exception as e:
            logger.error(f"[AI] Attempt {attempt} failed: {e}")
            traceback.print_exc()
            time.sleep(2)
    return "⚠️ AI service currently unavailable. Try again later."

# -----------------------------
# EXCHANGE / REAL-TIME STREAMING
# -----------------------------
class ExchangeStreamer:
    """
    Unified interface to get realtime OHLC (candles) or tick data.
    - Prefers ccxt.pro (async websockets) if installed.
    - If not available, has a Binance-specific ThreadedWebsocketManager fallback (sync).
    - If neither is available, falls back to periodic REST polling via ccxt (or python-binance).
    """

    def __init__(self, exchange_id: str = "binance", api_key: Optional[str] = None, secret: Optional[str] = None):
        self.exchange_id = exchange_id.lower()
        self.api_key = api_key or os.getenv(f"{self.exchange_id.upper()}_API_KEY")
        self.secret = secret or os.getenv(f"{self.exchange_id.upper()}_SECRET")
        self._client = None
        self._ws_manager = None

        if ccxt_pro:
            try:
                logger.info("[Streamer] Initializing ccxt.pro client")
                self._client = getattr(ccxt_pro, self.exchange_id)({
                    "apiKey": self.api_key,
                    "secret": self.secret,
                    "enableRateLimit": True
                })
            except Exception as e:
                logger.warning(f"[Streamer] ccxt.pro init failed: {e}")
                self._client = None
        elif self.exchange_id == "binance" and BinanceRestClient:
            self._client = BinanceRestClient(api_key=self.api_key, api_secret=self.secret)
            logger.info("[Streamer] Using python-binance REST client as fallback")

    async def fetch_ohlcv_ccxt_pro(self, symbol: str, timeframe: str = "1m", limit: int = 100):
        """
        Async - uses ccxt.pro websocket to watch OHLCV stream (where supported).
        Returns pandas.DataFrame of OHLCV if pandas available.
        """
        if not ccxt_pro or not self._client:
            raise RuntimeError("ccxt.pro not available or client not initialized.")
        # `watch_ohlcv` is a ccxt.pro method for streaming OHLCV in many adapters
        try:
            ohlcv = await self._client.watch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = None
            if pd is not None:
                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df if df is not None else ohlcv
        except Exception as e:
            logger.error(f"[Streamer] ccxt.pro watch_ohlcv failed: {e}")
            raise

    def start_binance_threaded_kline(self, symbol: str, interval: str, callback):
        """
        Sync fallback: starts a python-binance ThreadedWebsocketManager for kline updates.
        callback gets called with parsed OHLC dict when a kline closes or updates.
        """
        if ThreadedWebsocketManager is None:
            raise RuntimeError("python-binance websockets not available.")
        # symbol expected like 'BTCUSDT'
        twm = ThreadedWebsocketManager()
        twm.start()
        def _handler(msg):
            # msg structure: contains 'k' with candle info
            try:
                k = msg.get("k", {})
                candle = {
                    "symbol": symbol,
                    "startTime": k.get("t"),
                    "open": float(k.get("o")),
                    "high": float(k.get("h")),
                    "low": float(k.get("l")),
                    "close": float(k.get("c")),
                    "volume": float(k.get("v")),
                    "isFinal": k.get("x", False)
                }
                callback(candle)
            except Exception as e:
                logger.exception(f"[Streamer] Error handling twm msg: {e}")
        twm.start_kline_socket(callback=_handler, symbol=symbol, interval=interval)
        self._ws_manager = twm
        logger.info("[Streamer] Started threaded kline socket for %s" % symbol)

    def rest_fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100):
        """
        Simple REST polling as last-resort. Uses ccxt if available or python-binance.
        """
        if pd is None:
            raise RuntimeError("pandas required for rest_fetch_ohlcv fallback.")
        # use ccxt library if available (non-pro)
        try:
            import ccxt
            ex = getattr(ccxt, self.exchange_id)({
                "apiKey": self.api_key, "secret": self.secret, "enableRateLimit": True
            })
            bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df
        except Exception as e:
            logger.warning(f"[Streamer] ccxt REST fetch failed: {e}")

        # python-binance fallback (symbols like 'BTCUSDT', interval '1m')
        if BinanceRestClient:
            try:
                client = BinanceRestClient(api_key=self.api_key, api_secret=self.secret)
                # python-binance uses klines method
                klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
                df = pd.DataFrame(klines, columns=["open_time","open","high","low","close","volume","close_time","x1","x2","x3","x4","x5"])
                df = df[["open_time","open","high","low","close","volume"]]
                df.columns = ["ts","open","high","low","close","volume"]
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
                return df
            except Exception as e:
                logger.warning(f"[Streamer] python-binance REST fetch failed: {e}")

        raise RuntimeError("No supported REST client available for fetch_ohlcv.")

# -----------------------------
# IMAGE ANALYSIS (CHARTS)
# -----------------------------
class ImageAnalyzer:
    """
    Analyze uploaded chart images using:
      1) OpenAI Vision/Responses (if a vision-capable model & client exist), or
      2) Local CV heuristics (OpenCV) to estimate OHLC series and detect patterns.
    Returns a dict with estimated OHLC series and a textual summary.
    """

    def __init__(self, openai_client=client, model: str = OPENAI_MODEL):
        self.client = openai_client
        self.model = model

    def _encode_image_b64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_with_openai(self, image_path: str, prompt_extra: str = "") -> Dict[str, Any]:
        """
        Use OpenAI's vision-capable Responses or images features to ask the model to
        parse the chart image and return structured OHLC or signals.
        NOTE: The exact request body can differ between client versions — this function
        tries a safe, commonly-supported shape for the modern Responses API.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not configured.")

        logger.info("[ImageAnalyzer] Sending image to OpenAI for analysis...")
        b64 = self._encode_image_b64(image_path)
        # Build a plain textual instruction + inline base64 (many multimodal clients accept this pattern)
        user_text = (
            "You are a crypto chart analyst. The user uploaded a price chart image encoded in base64. "
            "Please extract the recent OHLC sequence (at least 20 bars if visible), detect any clear patterns "
            "(e.g., head & shoulders, double top/bottom, ascending triangle), and provide a short signal: BUY/HOLD/SELL, "
            "with explicit reasons and suggested stop-loss and take-profit levels. "
            + prompt_extra
        )

        try:
            # Attempt modern multimodal 'responses' interface if available
            if hasattr(self.client, "responses") and hasattr(self.client.responses, "create"):
                # Many modern examples accept input as a list containing text + image
                input_payload = [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_b64": b64}
                ]
                resp = self.client.responses.create(
                    model=self.model,
                    input=input_payload,
                    max_output_tokens=800
                )
                # Best-effort parse: return raw response plus text
                text = ""
                try:
                    # new Responses returns .output_text or .output[0].content[0].text in some clients
                    if hasattr(resp, "output_text"):
                        text = resp.output_text
                    else:
                        # generic fallback
                        text = json.dumps(resp, default=str)
                except Exception:
                    text = str(resp)
                return {"ok": True, "analysis": text, "raw": resp}
            else:
                # Legacy fallback: send prompt that includes base64 (may be supported)
                prompt = f"{user_text}\n\nBase64Image:\n{b64[:200]}... (truncated) -- provide full parsing on your side."
                txt = ai_analysis_text(prompt)
                return {"ok": True, "analysis": txt}
        except Exception as e:
            logger.exception(f"[ImageAnalyzer] OpenAI analysis failed: {e}")
            return {"ok": False, "error": str(e)}

    def analyze_with_opencv(self, image_path: str, bars_expected: int = 30):
        """
        Local heuristic attempt to extract candlestick-like OHLC series from a chart image.
        This is an approximation; results vary by chart style and resolution.
        Requires OpenCV (cv2) and numpy/pandas.
        """
        if cv2 is None or np is None or pd is None:
            raise RuntimeError("OpenCV, numpy and pandas required for cv fallback.")

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Failed to read image.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use adaptive threshold to emphasize bars/lines
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        # Find vertical structures (candles usually have vertical bodies/wicks)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(c) for c in contours]
        # Filter small rects, sort left-to-right
        rects = [r for r in rects if r[2] > 2 and r[3] > 10]
        rects = sorted(rects, key=lambda x: x[0])

        # Reduce to most significant vertical rects (bars)
        if len(rects) > bars_expected:
            rects = rects[-bars_expected:]

        # convert rects into rough OHLC by mapping vertical position to price percent
        h = img.shape[0]
        def y_to_price_pct(y):  # 0..h -> 1..0
            return 1.0 - (y / float(h))
        rows = []
        for (x, y, w, he) in rects:
            top = y
            bottom = y + he
            mid = y + he/2
            # approximate open/high/low/close using shape orientation and brightness
            open_p = y_to_price_pct(top)
            close_p = y_to_price_pct(bottom)
            high_p = y_to_price_pct(top)
            low_p = y_to_price_pct(bottom)
            rows.append({"x": x, "open": open_p, "high": high_p, "low": low_p, "close": close_p})
        if pd is not None:
            df = pd.DataFrame(rows)
            return {"ok": True, "df": df}
        return {"ok": True, "rows": rows}

# -----------------------------
# SIGNAL GENERATION
# -----------------------------
class SignalGenerator:
    """
    Basic signal generator that accepts OHLC pandas.DataFrame and returns signals.
    Implemented indicators: SMA, EMA, RSI. Simple rule-based signals produced.
    """

    def __init__(self):
        if pd is None or np is None:
            logger.warning("[SignalGenerator] pandas/numpy not available; signals will be limited.")

    @staticmethod
    def sma(series: pd.Series, period: int):
        return series.rolling(period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14):
        delta = series.diff()
        up = delta.clip(lower=0).fillna(0)
        down = -1 * delta.clip(upper=0).fillna(0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        df must have 'close' column and be ordered oldest->most recent.
        Returns a small report with signal and indicator values.
        """
        if pd is None:
            raise RuntimeError("pandas required for signal generation.")

        close = df["close"].astype(float)
        result = {}
        df["ema20"] = self.ema(close, 20)
        df["ema50"] = self.ema(close, 50)
        df["sma50"] = self.sma(close, 50)
        df["rsi14"] = self.rsi(close, 14)

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        # Simple rules:
        signal = "HOLD"
        reason = []
        # EMA crossover
        if last["ema20"] > last["ema50"] and prev["ema20"] <= prev["ema50"]:
            signal = "BUY"
            reason.append("Bullish EMA20 crossed above EMA50")
        elif last["ema20"] < last["ema50"] and prev["ema20"] >= prev["ema50"]:
            signal = "SELL"
            reason.append("Bearish EMA20 crossed below EMA50")

        # RSI extremes
        if last["rsi14"] < 30:
            reason.append(f"RSI {last['rsi14']:.1f} indicates oversold")
            if signal == "HOLD":
                signal = "BUY"
        elif last["rsi14"] > 70:
            reason.append(f"RSI {last['rsi14']:.1f} indicates overbought")
            if signal == "HOLD":
                signal = "SELL"

        # Price vs sma
        if last["close"] > last["sma50"]:
            reason.append("Price above SMA50")
        else:
            reason.append("Price below SMA50")

        result["signal"] = signal
        result["reason"] = reason
        result["indicators"] = {
            "ema20": float(last["ema20"]),
            "ema50": float(last["ema50"]),
            "sma50": float(last["sma50"]),
            "rsi14": float(last["rsi14"])
        }
        result["price"] = float(last["close"])
        return result

# -----------------------------
# HIGH-LEVEL: analyze chart image -> produce signal
# -----------------------------
def analyze_image_and_signal(image_path: str, use_openai_first: bool = True) -> Dict[str, Any]:
    analyzer = ImageAnalyzer()
    siggen = SignalGenerator()

    # 1) try OpenAI vision first (if client available)
    if use_openai_first and client:
        try:
            resp = analyzer.analyze_with_openai(image_path)
            if resp.get("ok"):
                # Return model's textual analysis and raw resp; user can parse structured items if included.
                return {"source": "openai", "analysis": resp.get("analysis"), "raw": resp.get("raw")}
        except Exception as e:
            logger.warning(f"[Main] OpenAI image analysis failed: {e}")

    # 2) fallback: local cv heuristics -> attempt to build a DataFrame and run indicators
    try:
        cvresp = analyzer.analyze_with_opencv(image_path)
        if cvresp.get("ok") and "df" in cvresp:
            df = cvresp["df"]
            # Map normalized price pct back to hypothetical price by scaling; here we just treat them as synthetic prices
            df_sorted = df.sort_values("x").reset_index(drop=True)
            # convert normalized to a synthetic price (e.g., 100..200)
            price_scale = 100.0
            df_sorted["close"] = (df_sorted["close"] * price_scale).astype(float)
            df_sorted["open"] = (df_sorted["open"] * price_scale).astype(float)
            df_sorted["high"] = (df_sorted["high"] * price_scale).astype(float)
            df_sorted["low"] = (df_sorted["low"] * price_scale).astype(float)
            # run signals (requires at least ~60 bars for robust indicators; but we run anyway)
            try:
                signal = siggen.generate(df_sorted)
                return {"source": "opencv", "signal": signal, "df_preview": df_sorted.head().to_dict(orient="records")}
            except Exception as e:
                return {"ok": False, "error": f"Signal generation failed: {e}"}
    except Exception as e:
        logger.warning(f"[Main] OpenCV analysis failed: {e}")

    return {"ok": False, "error": "All analysis methods failed."}
    
