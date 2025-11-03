# market_providers.py
"""
Multi-exchange market data provider.

- Uses public REST endpoints for Binance/Bybit/KuCoin/OKX
- If ccxt is installed it prefers ccxt for OHLCV
- Exposes:
    - fetch_klines_multi(symbol, interval, limit, exchange)
    - analyze_pair_multi_timeframes(symbol, timeframes...)
    - detect_strong_signals(...)
    - fetch_trending_pairs_branded(...)
"""

import os
import json
import traceback
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional ccxt (preferred for multi-exchange OHLCV)
try:
    import ccxt
    HAS_CCXT = True
except Exception:
    HAS_CCXT = False

# Image utility (create branded images for trending)
try:
    from image_utils import create_brand_image
except Exception:
    create_brand_image = None

QUICKCHART_URL = "https://quickchart.io/chart"

# Endpoints
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"
BYBIT_TICKER = "https://api.bybit.com/v2/public/tickers"
BYBIT_KLINES_V5 = "https://api.bybit.com/v5/market/kline"
KUCOIN_KLINES = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_TICKER = "https://api.kucoin.com/api/v1/market/allTickers"
OKX_KLINES = "https://www.okx.com/api/v5/market/candles"
OKX_TICKER = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"

DEFAULT_TFS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "DestinyTradingEmpireBot/1.0"})
    proxy = os.getenv("PROXY_URL")
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504],
                    allowed_methods=frozenset(["GET","POST"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _quickchart_sparkline(values: List[float], width:int=800, height:int=240, label:str="") -> Optional[bytes]:
    try:
        cfg = {
            "type":"line",
            "data":{"labels":list(range(len(values))), "datasets":[{"label":label,"data":values,"fill":False,"pointRadius":0}]},
            "options":{"plugins":{"legend":{"display":False}},"elements":{"line":{"tension":0}}}
        }
        params = {"c": json.dumps(cfg), "width": width, "height": height, "devicePixelRatio":2}
        r = get_session().get(QUICKCHART_URL, params=params, timeout=12)
        r.raise_for_status()
        return r.content
    except Exception:
        traceback.print_exc()
        return None

# --- ticker helpers
def fetch_binance_tickers() -> List[Dict[str,Any]]:
    r = get_session().get(BINANCE_TICKER, timeout=8); r.raise_for_status(); return r.json()

def fetch_bybit_tickers() -> List[Dict[str,Any]]:
    r = get_session().get(BYBIT_TICKER, timeout=8); r.raise_for_status(); return r.json().get("result", [])

def fetch_kucoin_tickers() -> List[Dict[str,Any]]:
    r = get_session().get(KUCOIN_TICKER, timeout=8); r.raise_for_status(); return r.json().get("data", {}).get("ticker", [])

def fetch_okx_tickers() -> List[Dict[str,Any]]:
    r = get_session().get(OKX_TICKER, timeout=8); r.raise_for_status(); return r.json().get("data", [])

# --- convert common raw data -> DataFrame
def _parse_binance_klines(raw) -> pd.DataFrame:
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["open_time","open","high","low","close","volume"]]

def _try_parse_list_of_lists(arr) -> Optional[pd.DataFrame]:
    try:
        if isinstance(arr, list) and arr and isinstance(arr[0], list):
            # try binance style; if length >=6 take first six columns
            cols = ["open_time","open","high","low","close","volume"]
            df = pd.DataFrame(arr)
            if df.shape[1] >= 6:
                df = df.iloc[:,-df.shape[1]:]  # keep shape consistent
                # ensure convert
                if df.shape[1] >= 6:
                    df = df.iloc[:, :6]
                    df.columns = cols
                    # open_time may be ms
                    df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
                    for c in ["open","high","low","close","volume"]:
                        df[c] = df[c].astype(float)
                    return df[["open_time","open","high","low","close","volume"]]
    except Exception:
        traceback.print_exc()
    return None

# --- main klines fetcher (robust)
def fetch_klines_multi(symbol: str="BTCUSDT", interval: str="1h", limit: int=200, exchange: str="binance") -> Optional[pd.DataFrame]:
    """
    symbol: 'BTCUSDT' style
    interval: '1m','5m','15m','1h',...
    exchange: 'binance','bybit','kucoin','okx' or 'auto'
    """
    exchange = (exchange or "binance").lower()
    sess = get_session()

    # prefer ccxt if available
    if HAS_CCXT:
        try:
            ccxt_map = {"okx":"okx", "kucoin":"kucoin", "bybit":"bybit", "binance":"binance"}
            ccid = ccxt_map.get(exchange, exchange)
            ex = getattr(ccxt, ccid)()
            ex.load_markets()
            cc_symbol = symbol.replace("USDT", "/USDT")
            ohlcv = ex.fetch_ohlcv(cc_symbol, timeframe=interval, limit=limit)
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=["open_time","open","high","low","close","volume"])
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                for c in ["open","high","low","close","volume"]:
                    df[c] = df[c].astype(float)
                return df
        except Exception:
            traceback.print_exc()

    # exchange-specific REST fallbacks

    # Binance
    try:
        if exchange == "binance" or exchange == "auto":
            r = sess.get(BINANCE_KLINES, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
            r.raise_for_status()
            df = _parse_binance_klines(r.json())
            return df
    except Exception:
        traceback.print_exc()

    # Bybit (v5)
    try:
        if exchange == "bybit":
            r = sess.get("https://api.bybit.com/v5/market/kline", params={"category":"spot","symbol":symbol.upper(),"interval":interval,"limit":int(limit)}, timeout=10)
            r.raise_for_status()
            j = r.json()
            if j.get("result") and j["result"].get("list"):
                arr = j["result"]["list"]
                df = pd.DataFrame(arr, columns=["open_time","open","high","low","close","volume"])
                # detect unit
                first = df["open_time"].iloc[0]
                if isinstance(first, (int,float)) and first < 1e12:
                    df["open_time"] = pd.to_datetime(df["open_time"], unit="s")
                else:
                    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
                return df
    except Exception:
        traceback.print_exc()

    # KuCoin
    try:
        if exchange == "kucoin":
            ku_sym = symbol.upper()
            # KuCoin expects symbol like "BTC-USDT" often
            if ku_sym.endswith("USDT"):
                ku_sym = ku_sym[:-4] + "-USDT"
            r = sess.get(KUCOIN_KLINES, params={"symbol":ku_sym, "type":interval}, timeout=10)
            r.raise_for_status()
            j = r.json()
            if isinstance(j.get("data"), list):
                arr = j["data"]
                df = _try_parse_list_of_lists(arr)
                if df is not None:
                    return df
    except Exception:
        traceback.print_exc()

    # OKX
    try:
        if exchange == "okx":
            ok_sym = symbol.upper()
            if ok_sym.endswith("USDT"):
                ok_sym = ok_sym[:-4] + "-USDT"
            r = sess.get(OKX_KLINES, params={"instId":ok_sym, "bar":interval, "limit":int(limit)}, timeout=10)
            r.raise_for_status()
            j = r.json()
            if j.get("data"):
                arr = j["data"]
                df = _try_parse_list_of_lists(arr)
                if df is not None:
                    return df
    except Exception:
        traceback.print_exc()

    # final fallback: try Binance again
    try:
        r = sess.get(BINANCE_KLINES, params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}, timeout=10)
        r.raise_for_status()
        return _parse_binance_klines(r.json())
    except Exception:
        traceback.print_exc()

    return None

# ---------- indicators and scoring (lightweight) ----------
def sma(series: pd.Series, period:int): return series.rolling(period).mean()
def ema(series: pd.Series, period:int): return series.ewm(span=period, adjust=False).mean()
def rsi(series: pd.Series, period:int=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    mc = ef - es
    msig = mc.ewm(span=signal, adjust=False).mean()
    return mc, msig, mc-msig
def atr(df: pd.DataFrame, period:int=14):
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# candle patterns
def detect_engulfing(df: pd.DataFrame):
    try:
        if len(df) < 2: return None
        prev, last = df.iloc[-2], df.iloc[-1]
        prev_body = abs(prev.close - prev.open)
        last_body = abs(last.close - last.open)
        if prev.close < prev.open and last.close > last.open and last_body > prev_body and last.close > prev.open:
            return "BULL_ENGULFING"
        if prev.close > prev.open and last.close < last.open and last_body > prev_body and last.close < prev.open:
            return "BEAR_ENGULFING"
    except Exception:
        pass
    return None

def detect_single_candle(row) -> Optional[str]:
    try:
        o,h,l,c = row.open, row.high, row.low, row.close
        body = abs(c-o); rng = (h-l) + 1e-12
        upper = h - max(c,o); lower = min(c,o) - l
        if body / rng < 0.35 and lower > 2*body: return "HAMMER"
        if body / rng < 0.35 and upper > 2*body: return "SHOOTING_STAR"
    except Exception:
        pass
    return None

def score_momentum_and_candle(df: pd.DataFrame) -> (float, List[str]):
    try:
        if df is None or len(df) < 30:
            return 0.0, ["insufficient_data"]
        closes = df["close"]
        reasons = []
        ema_fast = ema(closes, 9).iloc[-1]; ema_slow = ema(closes,21).iloc[-1]
        prev_fast = ema(closes,9).iloc[-2]; prev_slow = ema(closes,21).iloc[-2]
        ema_score = 0.0
        if ema_fast > ema_slow and prev_fast <= prev_slow:
            ema_score += 0.25; reasons.append("ema_cross_up")
        elif ema_fast < ema_slow and prev_fast >= prev_slow:
            ema_score -= 0.25; reasons.append("ema_cross_down")
        r = float(rsi(closes).iloc[-1])
        if r < 30: r_score = 0.15; reasons.append("rsi_oversold")
        elif r > 70: r_score = -0.12; reasons.append("rsi_overbought")
        else: r_score = 0.05
        macd_val, macd_sig, macd_hist = macd(closes)
        mh = macd_hist.iloc[-1]
        macd_score = 0.12 if mh > 0 else -0.03
        atr_val = float(atr(df).iloc[-1])
        atr_score = 0.05 if atr_val < (closes.std()) else -0.01
        pat = detect_engulfing(df)
        single = detect_single_candle(df.iloc[-1])
        if pat:
            pat_score = 0.25 if "BULL" in pat else -0.2
            reasons.append(pat.lower())
        elif single:
            pat_score = 0.18 if single=="HAMMER" else -0.18
            reasons.append(single.lower())
        else:
            pat_score = 0.0
        ret_3 = (closes.iloc[-1] - closes.iloc[-4]) / (closes.iloc[-4] + 1e-12)
        momentum = 0.08 if ret_3 > 0.01 else (-0.08 if ret_3 < -0.01 else 0.0)
        if momentum>0: reasons.append("short_momentum_pos")
        elif momentum<0: reasons.append("short_momentum_neg")
        score = 0.5 + ema_score + r_score + macd_score + atr_score + pat_score + momentum
        score = max(0.0, min(1.0, score))
        return round(score,4), reasons
    except Exception:
        traceback.print_exc()
        return 0.0, ["error_scoring"]

# analyze pair across TFs
def analyze_pair_multi_timeframes(symbol: str, timeframes: Optional[List[str]] = None, exchange: str="binance") -> Dict[str,Any]:
    if timeframes is None: timeframes = DEFAULT_TFS
    results = {}
    scores = []
    closes={}
    try:
        for tf in timeframes:
            df = fetch_klines_multi(symbol, interval=tf, limit=200, exchange=exchange)
            if df is None or df.empty or "close" not in df:
                results[tf] = {"error":"no_data"}
                continue
            df = df.copy()
            df["close"] = df["close"].astype(float)
            score, reasons = score_momentum_and_candle(df)
            last = float(df["close"].iloc[-1]); prev = float(df["close"].iloc[-2])
            sdir = "LONG" if score>0.55 and last>prev else ("SHORT" if score>0.55 and last<prev else "HOLD")
            atr_val = float(atr(df).iloc[-1]) if len(df)>15 else (df["high"].max()-df["low"].min())*0.005
            if sdir=="LONG":
                sl = last - 1.5*atr_val; tp1 = last + 1.5*atr_val
            elif sdir=="SHORT":
                sl = last + 1.5*atr_val; tp1 = last - 1.5*atr_val
            else:
                sl = last*0.995; tp1 = last*1.005
            results[tf] = {"score":round(score,3),"signal":sdir,"reasons":reasons,"close":last,"sl":round(sl,8),"tp1":round(tp1,8)}
            scores.append(score); closes[tf]=last
        # combine scores, weight higher TF more
        tf_weights_map = {"1m":0.5,"5m":0.7,"15m":0.9,"30m":1.0,"1h":1.4,"4h":1.6,"1d":2.0}
        total_weight = sum(tf_weights_map.get(tf,1.0) for tf in timeframes)
        sum_prod = sum(results.get(tf,{}).get("score",0.0)*tf_weights_map.get(tf,1.0) for tf in timeframes)
        combined_score = (sum_prod / (total_weight+1e-12)) if total_weight>0 else 0.0
        long_count = sum(1 for tf in timeframes if results.get(tf,{}).get("signal")=="LONG")
        short_count = sum(1 for tf in timeframes if results.get(tf,{}).get("signal")=="SHORT")
        if combined_score >= 0.9 and long_count>short_count: combined_signal="STRONG_LONG"
        elif combined_score >= 0.9 and short_count>long_count: combined_signal="STRONG_SHORT"
        elif combined_score >= 0.65 and long_count>short_count: combined_signal="LONG"
        elif combined_score >= 0.65 and short_count>long_count: combined_signal="SHORT"
        else: combined_signal="HOLD"
        return {"symbol":symbol.upper(),"exchange":exchange,"analysis":results,"combined_score":round(combined_score,4),"combined_signal":combined_signal,"prices":closes}
    except Exception:
        traceback.print_exc()
        return {"error":"analysis_failed"}

def detect_strong_signals(pairs: Optional[List[str]]=None, timeframes: Optional[List[str]]=None, exchange:str="binance", min_confidence:float=0.90) -> List[Dict[str,Any]]:
    if pairs is None: pairs = os.getenv("PAIRS","BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT").split(",")
    if timeframes is None: timeframes = DEFAULT_TFS
    out=[]
    for p in pairs:
        try:
            res = analyze_pair_multi_timeframes(p, timeframes=timeframes, exchange=exchange)
            if not isinstance(res, dict) or res.get("error"): continue
            score = float(res.get("combined_score",0.0))
            if score >= min_confidence and res.get("combined_signal") in ("STRONG_LONG","STRONG_SHORT"):
                # build image summary
                chart_bytes=None
                try:
                    kdf = fetch_klines_multi(p, "1h", limit=80, exchange=exchange)
                    if isinstance(kdf, pd.DataFrame) and "close" in kdf:
                        chart_bytes = _quickchart_sparkline(kdf["close"].tolist()[-60:], label=p)
                except Exception:
                    pass
                # get sl/tp from 1h if present
                oneh = res.get("analysis",{}).get("1h") or next(iter(res.get("analysis",{}).values()), {})
                sl = oneh.get("sl"); tp1 = oneh.get("tp1")
                lines = [f"🔥 {res['combined_signal']} detected", f"Pair: {p} ({exchange})", f"Score: {score:.3f}"]
                for tf,tinfo in res.get("analysis",{}).items():
                    if isinstance(tinfo, dict) and "score" in tinfo:
                        lines.append(f"{tf}: {tinfo['signal']} ({int(tinfo['score']*100)}%)")
                img_buf=None
                try:
                    if create_brand_image:
                        img_buf = create_brand_image(lines, chart_img=chart_bytes)
                except Exception:
                    traceback.print_exc()
                out.append({"symbol":p.upper(),"combined_score":round(score,4),"combined_signal":res.get("combined_signal"),"sl":sl,"tp1":tp1,"analysis":res,"image":img_buf,"caption_lines":lines})
        except Exception:
            traceback.print_exc()
            continue
    out = sorted(out, key=lambda x: x.get("combined_score",0.0), reverse=True)
    return out

def fetch_trending_pairs_branded(limit:int=10) -> Tuple[Optional[BytesIO], str]:
    rows=[]
    try:
        for item in fetch_binance_tickers():
            try:
                rows.append({"symbol":item.get("symbol"), "change":float(item.get("priceChangePercent",0)), "vol":float(item.get("quoteVolume",0)), "exchange":"Binance"})
            except Exception:
                pass
        for item in fetch_bybit_tickers():
            try:
                pct = float(item.get("price_24h_pcnt",0)) * 100 if item.get("price_24h_pcnt") is not None else 0.0
                rows.append({"symbol":item.get("symbol"), "change":pct, "vol":float(item.get("quote_volume",0) or 0), "exchange":"Bybit"})
            except Exception:
                pass
        if not rows:
            return None, "No exchange data available."
        df = pd.DataFrame(rows).sort_values("change", ascending=False).head(limit)
        lines = [f"🔥 Multi-Exchange Trending Pairs ({len(df)} picks)"]
        for _,r in df.iterrows():
            lines.append(f"{r['symbol']:<12} | {r['change']:+6.2f}% | vol: {int(r['vol']):,} | {r['exchange']}")
        chart_bytes=None
        try:
            top_sym = df.iloc[0]["symbol"]
            kdf = fetch_klines_multi(top_sym, "1h", limit=48)
            if isinstance(kdf, pd.DataFrame) and "close" in kdf:
                chart_bytes = _quickchart_sparkline(kdf["close"].tolist()[-48:], label=top_sym)
        except Exception:
            pass
        if create_brand_image:
            buf = create_brand_image(lines, chart_img=chart_bytes)
            return buf, "Top Multi-Exchange Trending Pairs"
        return None, "\n".join(lines)
    except Exception:
        traceback.print_exc()
        return None, "Error fetching trending pairs."
