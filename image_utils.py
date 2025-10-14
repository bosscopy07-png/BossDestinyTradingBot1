# image_utils.py
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

BRAND = "Boss Destiny Trading Empire"

def _get_font(size=18, bold=False):
    try:
        if bold:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def create_brand_image(lines):
    w, h = 900, 360
    img = Image.new("RGB", (w, h), (12, 14, 20))
    draw = ImageDraw.Draw(img)
    title_font = _get_font(22, bold=True)
    text_font = _get_font(16)
    draw.text((20, 12), BRAND, fill=(255, 200, 0), font=title_font)
    y = 56
    line_h = 28
    for line in lines:
        # Use textbbox to avoid .textsize errors
        try:
            draw.text((20, y), str(line), fill=(230, 230, 230), font=text_font)
        except Exception:
            draw.text((20, y), str(line), fill=(230, 230, 230))
        y += line_h
    bio = BytesIO()
    img.save(bio, "PNG")
    bio.seek(0)
    return bio

def build_signal_image(sig):
    lines = [
        f"{sig.get('symbol')}  |  {sig.get('interval')}  |  {sig.get('signal')}",
        f"Entry: {sig.get('entry')}   SL: {sig.get('sl')}   TP1: {sig.get('tp1')}",
        f"Confidence: {int(sig.get('confidence', 0)*100)}%   Risk USD: {sig.get('suggested_risk_usd')}",
        "Reasons: " + (", ".join(sig.get('reasons', [])) if sig.get('reasons') else "None")
    ]
    return create_brand_image(lines)

def safe_send_with_image(bot, chat_id, text, image_buf=None, reply_markup=None):
    try:
        if image_buf:
            bot.send_photo(chat_id, image_buf, caption=text, reply_markup=reply_markup)
        else:
            bot.send_message(chat_id, text, reply_markup=reply_markup)
    except Exception:
        try:
            bot.send_message(chat_id, text)
        except:
            pass
