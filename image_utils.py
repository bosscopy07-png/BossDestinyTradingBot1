from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap

BRAND = "Destiny Trading Empire 💎"
BRAND_COLOR = (255, 215, 0)
BACKGROUND_COLOR = (10, 12, 18)
TEXT_COLOR = (235, 235, 235)
WIDTH = 900
PADDING = 25
LINE_SPACING = 32
MIN_HEIGHT = 400  # minimum image height


def _get_font(size=18, bold=True):
    """Return a bold or regular font; fallback to default if missing."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _wrap_text(draw, text, font, max_width):
    """Wrap text into multiple lines to fit image width."""
    lines = []
    for paragraph in text.splitlines():
        wrapped = textwrap.wrap(paragraph, width=100)
        for line in wrapped:
            # dynamically split line if it still exceeds width
            while draw.textlength(line, font=font) > max_width:
                mid = len(line) // 2
                lines.append(line[:mid])
                line = line[mid:]
            lines.append(line)
    return lines


def _fit_font(draw, text_lines, max_width, max_font_size=22, min_font_size=12):
    """Auto-reduce font size to make text fit width."""
    font_size = max_font_size
    font = _get_font(font_size)
    while font_size >= min_font_size:
        fits = True
        for line in text_lines:
            if draw.textlength(line, font=font) > max_width:
                fits = False
                break
        if fits:
            return font
        font_size -= 1
        font = _get_font(font_size)
    return font


def create_brand_image(lines, chart_img=None, title=None, subtitle=None):
    """Create branded image with all text embedded + optional chart."""
    # Temporary draw to measure
    temp_img = Image.new("RGB", (WIDTH, MIN_HEIGHT))
    temp_draw = ImageDraw.Draw(temp_img)

    # Wrap all text
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(_wrap_text(temp_draw, str(line), _get_font(), WIDTH - 2 * PADDING))

    # Choose font size automatically
    text_font = _fit_font(temp_draw, wrapped_lines, WIDTH - 2 * PADDING)

    # Compute image height
    chart_height = int((WIDTH - 2 * PADDING) * 0.35) + 10 if chart_img else 0
    height = max(MIN_HEIGHT, PADDING + 60 + len(wrapped_lines) * LINE_SPACING + chart_height + PADDING)

    img = Image.new("RGB", (WIDTH, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw title
    title_font = _get_font(32, bold=True)
    draw.text((PADDING, PADDING), title or BRAND, fill=BRAND_COLOR, font=title_font)

    # Draw subtitle if exists
    if subtitle:
        sub_font = _get_font(18)
        draw.text((WIDTH - 300, PADDING + 5), subtitle, fill=TEXT_COLOR, font=sub_font)

    # Draw text lines
    y = PADDING + 60
    for line in wrapped_lines:
        draw.text((PADDING, y), line, fill=TEXT_COLOR, font=text_font)
        y += LINE_SPACING

    # Add chart if any
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
            chart_h = int(chart_w * 0.35)
            chart = chart.resize((chart_w, chart_h))
            img.paste(chart, (PADDING, y + 10))
        except Exception:
            pass

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def build_signal_image(sig, chart_img=None):
    """Build branded signal image with auto-expanded text."""
    lines = [
        f"{sig.get('symbol','')}  |  {sig.get('interval','')}  |  {sig.get('signal','')}",
        f"Entry: {sig.get('entry','N/A')}   SL: {sig.get('sl','N/A')}   TP1: {sig.get('tp1','N/A')}",
        f"Confidence: {int(sig.get('confidence',0)*100)}%   Risk USD: {sig.get('suggested_risk_usd','N/A')}",
        "Reasons: " + (", ".join(sig.get('reasons',[])) if sig.get('reasons') else "None")
    ]
    return create_brand_image(lines, chart_img=chart_img, title=BRAND)


def build_text_image(message_text):
    """Convert any plain text message into a branded image automatically."""
    lines = message_text.splitlines()
    return create_brand_image(lines, title=BRAND)


def safe_send_with_image(bot, chat_id, text, image_buf=None, reply_markup=None):
    """Always send text as an image, even if not a signal."""
    try:
        if not image_buf:
            image_buf = build_text_image(text)
        bot.send_photo(
            chat_id,
            image_buf,
            caption=f"— <b>{BRAND}</b>",
            reply_markup=reply_markup,
            parse_mode="HTML"
        )
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            bot.send_message(chat_id, f"⚠️ Failed to send branded image.\n\n{text}")
        except Exception:
            pass
