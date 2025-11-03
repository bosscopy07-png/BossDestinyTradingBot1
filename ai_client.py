# ai_client.py
import os
import time
import logging
import traceback
import base64
import json
from typing import List, Dict, Optional, Any

# Optional libraries
try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import ccxt.pro as ccxt_pro
except Exception:
    ccxt_pro = None

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
# FEATURE FLAGS
# -----------------------------
HAS_PANDAS = pd is not None
HAS_CV2 = cv2 is not None
HAS_CCXT_PRO = ccxt_pro is not None
HAS_BINANCE = BinanceRestClient is not None

# -----------------------------
# OPENAI CLIENT
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = None

try:
    if OPENAI_API_KEY:
        try:
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

HAS_AI_CLIENT = client is not None

# -----------------------------
# AI TEXT ANALYSIS
# -----------------------------
def ai_analysis_text(prompt: str, retries: int = 3, temperature: float = 0.6, max_tokens: int = 400) -> Dict[str, Any]:
    """
    Generate AI-based trading analysis text with structured output.
    """
    if not HAS_AI_CLIENT:
        return {"ok": False, "error": "AI client not configured."}

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[AI] Processing request (attempt {attempt})...")

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
                return {"ok": True, "analysis": text, "raw": resp}

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
                return {"ok": True, "analysis": text, "raw": resp}

            else:
                return {"ok": False, "error": "Unsupported AI interface."}

        except Exception as e:
            logger.error(f"[AI] Attempt {attempt} failed: {e}")
            traceback.print_exc()
            time.sleep(2)

    return {"ok": False, "error": "AI service currently unavailable."}

# -----------------------------
# EXCHANGE STREAMER
# -----------------------------
class ExchangeStreamer:
    def __init__(self, exchange_id: str = "binance", api_key: Optional[str] = None, secret: Optional[str] = None):
        self.exchange_id = exchange_id.lower()
        self.api_key = api_key or os.getenv(f"{self.exchange_id.upper()}_API_KEY")
        self.secret = secret or os.getenv(f"{self.exchange_id.upper()}_SECRET")
        self._client = None
        self._ws_manager = None

        if HAS_CCXT_PRO:
            try:
                self._client = getattr(ccxt_pro, self.exchange_id)({
                    "apiKey": self.api_key,
                    "secret": self.secret,
                    "enableRateLimit": True
                })
                logger.info("[Streamer] ccxt.pro client initialized")
            except Exception as e:
                logger.warning(f"[Streamer] ccxt.pro init failed: {e}")
        elif self.exchange_id == "binance" and HAS_BINANCE:
            self._client = BinanceRestClient(api_key=self.api_key, api_secret=self.secret)
            logger.info("[Streamer] Using python-binance REST client as fallback")

    async def fetch_ohlcv_ccxt_pro(self, symbol: str, timeframe: str = "1m", limit: int = 100):
        if not HAS_CCXT_PRO or not self._client:
            raise RuntimeError("ccxt.pro not available.")
        ohlcv = await self._client.watch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if HAS_PANDAS:
            df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df
        return ohlcv

    def start_binance_threaded_kline(self, symbol: str, interval: str, callback):
        if not HAS_BINANCE:
            raise RuntimeError("Binance websockets not available.")
        twm = ThreadedWebsocketManager()
        twm.start()

        def _handler(msg):
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
                logger.exception(f"[Streamer] Error handling msg: {e}")

        twm.start_kline_socket(callback=_handler, symbol=symbol, interval=interval)
        self._ws_manager = twm
        logger.info(f"[Streamer] Started threaded kline socket for {symbol}")

    def rest_fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100):
        if not HAS_PANDAS:
            raise RuntimeError("pandas required for REST fetch.")
        try:
            import ccxt
            ex = getattr(ccxt, self.exchange_id)({"apiKey": self.api_key, "secret": self.secret, "enableRateLimit": True})
            bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df
        except Exception as e:
            logger.warning(f"[Streamer] ccxt REST fetch failed: {e}")
        if HAS_BINANCE:
            try:
                client = BinanceRestClient(api_key=self.api_key, api_secret=self.secret)
                klines = client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
                df = pd.DataFrame(klines, columns=["open_time","open","high","low","close","volume","close_time","x1","x2","x3","x4","x5"])
                df = df[["open_time","open","high","low","close","volume"]]
                df.columns = ["ts","open","high","low","close","volume"]
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
                return df
            except Exception as e:
                logger.warning(f"[Streamer] python-binance REST fetch failed: {e}")
        raise RuntimeError("No supported REST client available.")

# -----------------------------
# IMAGE ANALYZER
# -----------------------------
class ImageAnalyzer:
    def __init__(self, openai_client=client, model: str = OPENAI_MODEL):
        self.client = openai_client
        self.model = model

    def _encode_image_b64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_with_openai(self, image_path: str, prompt_extra: str = "") -> Dict[str, Any]:
        if not HAS_AI_CLIENT:
            return {"ok": False, "error": "OpenAI client not configured."}
        b64 = self._encode_image_b64(image_path)
        user_text = (
            "You are a crypto chart analyst. Analyze this price chart image and extract OHLC, detect patterns, "
            "and provide a BUY/HOLD/SELL signal with reasons. " + prompt_extra
        )
        try:
            if hasattr(self.client, "responses") and hasattr(self.client.responses, "create"):
                input_payload = [{"type":"input_text","text":user_text}, {"type":"input_image","image_b64":b64}]
                resp = self.client.responses.create(model=self.model, input=input_payload, max_output_tokens=800)
                text = getattr(resp, "output_text", str(resp))
                return {"ok": True, "analysis": text, "raw": resp}
            else:
                prompt = f"{user_text}\n\nBase64Image (truncated): {b64[:200]}"
                txt = ai_analysis_text(prompt)
                return {"ok": True, "analysis": txt}
        except Exception as e:
            logger.exception(f"[ImageAnalyzer] OpenAI analysis failed: {e}")
            return {"ok": False, "error": str(e)}

    def analyze_with_opencv(self, image_path: str, bars_expected: int = 30):
        if not (HAS_CV2 and HAS_PANDAS and np is not None):
            return {"ok": False, "error": "OpenCV, pandas, numpy required."}
        img = cv2.imread(image_path)
        if img is None:
            return {"ok": False, "error": "Failed to read image."}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,7))
        morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(c) for c in contours]
        rects = sorted([r for r in rects if r[2]>2 and r[3]>10], key=lambda x: x[0])[-bars_expected:]

        h = img.shape[0]
        def y_to_price_pct(y): return 1.0 - (y / float(h))
        rows = []
        for (x,y,w,he) in rects:
            top = y; bottom=y+he
            rows.append({"x":x,"open":y_to_price_pct(top),"high":y_to_price_pct(top),
                         "low":y_to_price_pct(bottom),"close":y_to_price_pct(bottom)})
        df = pd.DataFrame(rows)
        return {"ok": True, "df": df}

# -----------------------------
# SIGNAL GENERATOR
# -----------------------------
class SignalGenerator:
    def __init__(self):
        if not HAS_PANDAS or np is None:
            logger.warning("[SignalGenerator] pandas/numpy not available; signals limited.")

    @staticmethod
    def sma(series: pd.Series, period: int): return series.rolling(period).mean()
    @staticmethod
    def ema(series: pd.Series, period: int): return series.ewm(span=period, adjust=False).mean()
    @staticmethod
    def rsi(series: pd.Series, period: int = 14):
        delta = series.diff()
        up = delta.clip(lower=0).fillna(0)
        down = -1*delta.clip(upper=0).fillna(0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / ma_down
        return 100 - (100/(1+rs))

    def generate(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not HAS_PANDAS:
            return {"ok": False, "error":"pandas required for signal generation."}
        close = df["close"].astype(float)
        df["ema20"] = self.ema(close, 20)
        df["ema50"] = self.ema(close, 50)
        df["sma50"] = self.sma(close, 50)
        df["rsi14"] = self.rsi(close, 14)

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df)>=2 else last

        signal = "HOLD"
        reason = []

        if last["ema20"]>last["ema50"] and prev["ema20"]<=prev["ema50"]:
            signal="BUY"; reason.append("EMA20 crossed above EMA50")
        elif last["ema20"]<last["ema50"] and prev["ema20"]>=prev["ema50"]:
            signal="SELL"; reason.append("EMA20 crossed below EMA50")

        if last["rsi14"]<30: 
            reason.append(f"RSI {last['rsi14']:.1f} oversold"); 
            if signal=="HOLD": signal="BUY"
        elif last["rsi14"]>70: 
            reason.append(f"RSI {last['rsi14']:.1f} overbought"); 
            if signal=="HOLD": signal="SELL"

        reason.append("Price above SMA50" if last["close"]>last["sma50"] else "Price below SMA50")

        return {
            "ok": True,
            "signal": signal,
            "reason": reason,
            "indicators": {
                "ema20": float(last["ema20"]),
                "ema50": float(last["ema50"]),
                "sma50": float(last["sma50"]),
                "rsi14": float(last["rsi14"])
            },
            "price": float(last["close"])
        }

# -----------------------------
# HIGH-LEVEL IMAGE -> SIGNAL
# -----------------------------
def analyze_image_and_signal(image_path: str, use_openai_first: bool = True) -> Dict[str, Any]:
    analyzer = ImageAnalyzer()
    siggen = SignalGenerator()

    if use_openai_first and HAS_AI_CLIENT:
        try:
            resp = analyzer.analyze_with_openai(image_path)
            if resp.get("ok"): return {"source":"openai","analysis":resp.get("analysis"),"raw":resp.get("raw")}
        except Exception as e:
            logger.warning(f"[Main] OpenAI image analysis failed: {e}")

    try:
        cvresp = analyzer.analyze_with_opencv(image_path)
        if cvresp.get("ok") and "df" in cvresp:
            df = cvresp["df"].sort_values("x").reset_index(drop=True)
            price_scale = 100.0
            df[["open","high","low","close"]] = df[["open","high","low","close"]]*price_scale
            signal = siggen.generate(df)
            return {"source":"opencv","signal":signal,"df_preview":df.head().to_dict(orient="records")}
    except Exception as e:
        logger.warning(f"[Main] OpenCV analysis failed: {e}")

    return {"ok": False, "error": "All analysis methods failed."}
