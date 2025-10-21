from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

BRAND = "Boss Destiny Trading Empire"
BRAND_COLOR = (255, 200, 0)
BACKGROUND_COLOR = (12, 14, 20)
TEXT_COLOR = (230, 230, 230)
WIDTH, HEIGHT = 900, 360
PADDING = 20

def _get_font(size=18, bold=False):
    """
    Load DejaVuSans font, fallback to default if unavailable.
    """
    try:
        if bold:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

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

def safe_send_with_image(bot, chat_id, text, image_buf=None, reply_markup=None):
    """
    Safely send a message or photo on Telegram with fallback.
    Ensures branding is included if image_buf is provided.
    """
    try:
        BRAND_FOOTER = "\n\n— <b>Boss Destiny Trading Empire</b>"
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
