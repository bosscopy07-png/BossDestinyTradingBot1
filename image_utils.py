# image_utils.py
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

BRAND = "Destiny Trading Empire 💎"
BRAND_COLOR = (255, 215, 0)
BACKGROUND_COLOR = (10, 12, 18)
TEXT_COLOR = (235, 235, 235)
WIDTH, HEIGHT = 900, 400
PADDING = 25


def _get_font(size=18, bold=False):
    try:
        if bold:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def create_brand_image(lines, chart_img=None, title=None, subtitle=None):
    """
    Create a branded image with text + optional chart.
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw title
    title_font = _get_font(28, bold=True)
    draw.text((PADDING, PADDING), title or BRAND, fill=BRAND_COLOR, font=title_font)

    if subtitle:
        sub_font = _get_font(14)
        draw.text((WIDTH - 280, PADDING + 8), subtitle, fill=TEXT_COLOR, font=sub_font)

    # Draw text lines
    y = PADDING + 50
    text_font = _get_font(17)
    line_spacing = 28
    for line in lines:
        draw.text((PADDING, y), str(line), fill=TEXT_COLOR, font=text_font)
        y += line_spacing

    # Add chart (if any)
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
            chart_h = int(chart_w * 0.33)
            chart = chart.resize((chart_w, chart_h))
            img.paste(chart, (PADDING, y))
        except Exception:
            pass

    # Save to buffer
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def build_signal_image(sig, chart_img=None):
    """
    Create branded signal image.
    """
    lines = [
        f"{sig.get('symbol','')}  |  {sig.get('interval','')}  |  {sig.get('signal','')}",
        f"Entry: {sig.get('entry','N/A')}   SL: {sig.get('sl','N/A')}   TP1: {sig.get('tp1','N/A')}",
        f"Confidence: {int(sig.get('confidence',0)*100)}%   Risk USD: {sig.get('suggested_risk_usd','N/A')}",
        "Reasons: " + (", ".join(sig.get('reasons',[])) if sig.get('reasons') else "None")
    ]
    return create_brand_image(lines, chart_img=chart_img, title=BRAND)


def build_text_image(message_text):
    """
    Convert any plain text message into a branded image automatically.
    """
    lines = message_text.splitlines()
    return create_brand_image(lines, title=BRAND)


def safe_send_with_image(bot, chat_id, text, image_buf=None, reply_markup=None):
    """
    Always send a branded image — even for normal replies.
    If chart or image buffer is given, use it; else create a text image automatically.
    """
    try:
        if not image_buf:
            image_buf = build_text_image(text)

        bot.send_photo(chat_id, image_buf, caption=f"— <b>{BRAND}</b>", reply_markup=reply_markup, parse_mode="HTML")

    except Exception:
        import traceback
        traceback.print_exc()
        try:
            bot.send_message(chat_id, f"⚠️ Failed to send branded image.\n\n{text}")
        except Exception:
            pass
