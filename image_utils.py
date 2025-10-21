from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# ===============================
# Branding Configuration
# ===============================
BRAND = "Boss Destiny Trading Empire"
BRAND_COLOR = (255, 200, 0)
BACKGROUND_COLOR = (12, 14, 20)
TEXT_COLOR = (230, 230, 230)
WIDTH, HEIGHT = 900, 360
PADDING = 20
BRAND_FOOTER = "\n\n— <b>Boss Destiny Trading Empire</b>"

# ===============================
# Font Loader
# ===============================
def _get_font(size=18, bold=False):
    """Load DejaVuSans font, fallback to default if unavailable."""
    try:
        if bold:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

# ===============================
# Brand Image Generator
# ===============================
def create_brand_image(lines, chart_img=None):
    """
    Create a professional branded image with optional chart.
    - lines: list of strings
    - chart_img: optional PIL image or bytes to paste
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Brand title
    title_font = _get_font(26, bold=True)
    draw.text((PADDING, PADDING), BRAND, fill=BRAND_COLOR, font=title_font)

    # Text lines
    y = PADDING + 40
    line_spacing = 28
    text_font = _get_font(16)
    for line in lines:
        try:
            draw.text((PADDING, y), str(line), fill=TEXT_COLOR, font=text_font)
        except Exception:
            draw.text((PADDING, y), str(line), fill=TEXT_COLOR)
        y += line_spacing

    # Paste chart if provided
    if chart_img:
        try:
            if isinstance(chart_img, bytes):
                chart_img = Image.open(BytesIO(chart_img))
            chart_w = WIDTH - 2 * PADDING
            chart_h = int(chart_w * 0.4)
            chart_img = chart_img.resize((chart_w, chart_h))
            img.paste(chart_img, (PADDING, y))
        except Exception:
            pass

    # Save to BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ===============================
# Signal Image Builder
# ===============================
def build_signal_image(sig, chart_img=None):
    """
    Build a professional signal image from a signal dictionary.
    Includes entry, SL, TP, leverage, confidence, and reasons.
    """
    lines = [
        f"{sig.get('symbol', '')}  |  {sig.get('interval', '')}  |  {sig.get('signal', '')}",
        f"Entry: {sig.get('entry', 'N/A')}   SL: {sig.get('SL', 'N/A')}   TP1: {sig.get('TP1', 'N/A')}",
        f"Leverage: {sig.get('leverage', 'N/A')}x",
        f"Confidence: {int(sig.get('confidence', 0)*100)}%   Risk USD: {sig.get('risk_usd', 'N/A')}",
        "Reasons: " + (", ".join(sig.get('reasons', [])) if sig.get('reasons') else "None")
    ]
    return create_brand_image(lines, chart_img=chart_img)

# ===============================
# Safe Send Helper (Telegram)
# ===============================
def safe_send_with_image(bot, chat_id, text, image_buf=None, reply_markup=None):
    """Safely send photo if available, otherwise text — always with brand footer."""
    try:
        # Append brand footer once
        if BRAND_FOOTER.strip() not in text:
            text += BRAND_FOOTER

        if image_buf:
            bot.send_photo(chat_id, image_buf, caption=text, reply_markup=reply_markup)
        else:
            bot.send_message(chat_id, text, reply_markup=reply_markup)

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            bot.send_message(chat_id, f"⚠️ Failed to send image/text: {e}\n\n{text}", reply_markup=reply_markup)
        except Exception:
            pass

# ===============================
# Multi-Exchange Klines Fetcher
# ===============================
def fetch_klines_multi(symbols, interval="1h", limit=100):
    """
    Fetch OHLCV (candlestick) data for multiple symbols across Binance, Bybit, KuCoin, and OKX.
    Returns a dict: {symbol: DataFrame}
    """
    results = {}
    s = get_session()

    for sym in symbols:
        klines_data = []

        # Binance
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {"symbol": sym, "interval": interval, "limit": limit}
            r = s.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "_", "_", "_", "_", "_", "_"
            ])
            df["exchange"] = "Binance"
            klines_data.append(df[["open_time", "open", "high", "low", "close", "volume", "exchange"]])
        except Exception as e:
            print(f"[WARN] Binance klines failed for {sym}: {e}")

        # Bybit
        try:
            url = f"https://api.bybit.com/v5/market/kline"
            params = {"category": "spot", "symbol": sym, "interval": interval, "limit": limit}
            r = s.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json().get("result", {}).get("list", [])
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume", "_"
            ])
            df["exchange"] = "Bybit"
            klines_data.append(df[["open_time", "open", "high", "low", "close", "volume", "exchange"]])
        except Exception as e:
            print(f"[WARN] Bybit klines failed for {sym}: {e}")

        # KuCoin
        try:
            url = f"https://api.kucoin.com/api/v1/market/candles"
            params = {"symbol": sym, "type": interval}
            r = s.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json().get("data", [])
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "close", "high", "low", "volume", "_"
            ])
            df["exchange"] = "KuCoin"
            klines_data.append(df[["open_time", "open", "high", "low", "close", "volume", "exchange"]])
        except Exception as e:
            print(f"[WARN] KuCoin klines failed for {sym}: {e}")

        # OKX
        try:
            url = f"https://www.okx.com/api/v5/market/candles"
            params = {"instId": sym, "bar": interval, "limit": limit}
            r = s.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json().get("data", [])
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume", "_"
            ])
            df["exchange"] = "OKX"
            klines_data.append(df[["open_time", "open", "high", "low", "close", "volume", "exchange"]])
        except Exception as e:
            print(f"[WARN] OKX klines failed for {sym}: {e}")

        # Merge data if available
        if klines_data:
            results[sym] = pd.concat(klines_data, ignore_index=True)
        else:
            results[sym] = pd.DataFrame()

    return results
