# image_utils.py
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

BRAND = "Destiny Trading Empire Bot 💎"
BRAND_COLOR = (255, 200, 0)
BACKGROUND_COLOR = (12, 14, 20)
TEXT_COLOR = (230, 230, 230)
WIDTH, HEIGHT = 900, 360
PADDING = 20

def _get_font(size=18, bold=False):
    try:
        if bold:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def create_brand_image(lines, chart_img=None, title=None, subtitle=None):
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    title_font = _get_font(26, bold=True)
    draw.text((PADDING, PADDING), title or BRAND, fill=BRAND_COLOR, font=title_font)

    if subtitle:
        sub_font = _get_font(14)
        draw.text((WIDTH - 300, PADDING + 6), subtitle, fill=TEXT_COLOR, font=sub_font)

    y = PADDING + 40
    line_spacing = 26
    text_font = _get_font(16)
    for line in lines:
        draw.text((PADDING, y), str(line), fill=TEXT_COLOR, font=text_font)
        y += line_spacing

    if chart_img:
        try:
            if isinstance(chart_img, BytesIO):
                chart_img.seek(0)
                chart = Image.open(chart_img)
            elif isinstance(chart_img, (bytes, bytearray)):
                chart = Image.open(BytesIO(chart_img))
            else:
                chart = chart_img
            chart_w = WIDTH - 2 * PADDING
            chart_h = int(chart_w * 0.32)
            chart = chart.resize((chart_w, chart_h))
            img.paste(chart, (PADDING, y))
        except Exception:
            pass

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def build_signal_image(sig, chart_img=None):
    lines = [
        f"{sig.get('symbol','')}  |  {sig.get('interval','')}  |  {sig.get('signal','')}",
        f"Entry: {sig.get('entry','N/A')}   SL: {sig.get('sl','N/A')}   TP1: {sig.get('tp1','N/A')}",
        f"Confidence: {int(sig.get('confidence',0)*100)}%   Risk USD: {sig.get('suggested_risk_usd','N/A')}",
        "Reasons: " + (", ".join(sig.get('reasons',[])) if sig.get('reasons') else "None")
    ]
    return create_brand_image(lines, chart_img=chart_img, title="Destiny Trading Empire Bot 💎")

def safe_send_with_image(bot, chat_id, text, image_buf=None, reply_markup=None):
    """
    Safely send photo if available, otherwise text — always with brand footer.
    """
    try:
        BRAND_FOOTER = "\n\n— <b>Destiny Trading Empire Bot 💎</b>"
        if BRAND_FOOTER.strip() not in text:
            text = text + BRAND_FOOTER
        if image_buf:
            # telegram expects file-like or bytes; BytesIO works fine
            bot.send_photo(chat_id, image_buf, caption=text, reply_markup=reply_markup)
        else:
            bot.send_message(chat_id, text, reply_markup=reply_markup)
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            bot.send_message(chat_id, f"⚠️ Failed to send image/text.\n\n{text}")
        except Exception:
            pass
