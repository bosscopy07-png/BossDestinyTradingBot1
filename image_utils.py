
# image_utils.py - Branded image generation for Destiny Trading Empire
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import logging
from typing import List, Optional, Union, Tuple

logger = logging.getLogger("image_utils")

# ----- Brand Constants -----
BRAND_NAME = "Destiny Trading Empire 💎"
BRAND_PRIMARY = (255, 215, 0)      # Gold
BRAND_BG = (10, 12, 18)          # Dark navy
BRAND_TEXT = (235, 235, 235)     # Off-white
BRAND_ACCENT = (0, 168, 255)    # Cyan for highlights

# ----- Layout Constants -----
CANVAS_WIDTH = 900
PADDING_X = 30
PADDING_Y = 25
TITLE_HEIGHT = 50
LINE_HEIGHT = 34
CHART_RATIO = 0.4                 # Chart height relative to width
MIN_CANVAS_HEIGHT = 450
MAX_FONT_SIZE = 24
MIN_FONT_SIZE = 13
DEFAULT_FONT_SIZE = 18

# ----- Font Cache -----
_font_cache: dict = {}

def _get_font(size: int = DEFAULT_FONT_SIZE, bold: bool = True) -> ImageFont.FreeTypeFont:
    """
    Get cached font with fallback chain.
    Tries: DejaVu Sans → Arial → Helvetica → Default
    """
    cache_key = f"{size}_{bold}"
    if cache_key in _font_cache:
        return _font_cache[cache_key]
    
    font_names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "Helvetica Bold.ttf" if bold else "Helvetica.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans.ttf",
    ]
    
    font = None
    for name in font_names:
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    
    if font is None:
        font = ImageFont.load_default()
        logger.warning(f"Using default font for size {size}, bold={bold}")
    
    _font_cache[cache_key] = font
    return font


def _calculate_text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    """Calculate text width, handling both old and new Pillow APIs."""
    try:
        # Pillow >= 8.0.0
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except AttributeError:
        # Fallback for older Pillow
        return draw.textlength(text, font=font)


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw, 
    text: str, 
    font: ImageFont.FreeTypeFont, 
    max_width: int
) -> List[str]:
    """
    Intelligently wrap text to fit within max_width pixels.
    Uses binary search for efficient line breaking.
    """
    if not text or not text.strip():
        return []
    
    paragraphs = text.splitlines()
    result = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            result.append("")
            continue
            
        # Estimate chars per line based on average char width
        avg_char_width = _calculate_text_width(draw, "x", font) or 8
        est_chars_per_line = int(max_width / avg_char_width)
        
        # Initial wrap
        wrapped = textwrap.wrap(paragraph, width=max(20, est_chars_per_line))
        
        # Refine: ensure no line exceeds pixel width
        for line in wrapped:
            if _calculate_text_width(draw, line, font) <= max_width:
                result.append(line)
                continue
            
            # Binary search for break point if still too long
            left, right = 1, len(line)
            best_break = len(line) // 2
            
            while left <= right:
                mid = (left + right) // 2
                test_line = line[:mid]
                width = _calculate_text_width(draw, test_line, font)
                
                if width <= max_width:
                    best_break = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            result.append(line[:best_break])
            remainder = line[best_break:].lstrip()
            if remainder:
                result.append(remainder)
    
    return result


def _fit_font_to_width(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    max_width: int,
    max_size: int = MAX_FONT_SIZE,
    min_size: int = MIN_FONT_SIZE
) -> ImageFont.FreeTypeFont:
    """Binary search for optimal font size that fits all lines."""
    if not lines:
        return _get_font(max_size)
    
    # Quick check if max size works
    test_font = _get_font(max_size)
    all_fit = all(
        _calculate_text_width(draw, line, test_font) <= max_width 
        for line in lines if line
    )
    if all_fit:
        return test_font
    
    # Binary search for best size
    low, high = min_size, max_size
    best_font = test_font
    
    while low <= high:
        mid = (low + high) // 2
        font = _get_font(mid)
        
        fits = all(
            _calculate_text_width(draw, line, font) <= max_width 
            for line in lines if line
        )
        
        if fits:
            best_font = font
            low = mid + 1  # Try larger
        else:
            high = mid - 1  # Need smaller
    
    return best_font


def _process_chart_image(
    chart_input: Optional[Union[BytesIO, bytes, Image.Image]]
) -> Optional[Image.Image]:
    """Normalize chart input to PIL Image."""
    if chart_input is None:
        return None
    
    try:
        if isinstance(chart_input, BytesIO):
            chart_input.seek(0)
            return Image.open(chart_input).convert("RGB")
        
        if isinstance(chart_input, (bytes, bytearray)):
            return Image.open(BytesIO(chart_input)).convert("RGB")
        
        if isinstance(chart_input, Image.Image):
            return chart_input.convert("RGB")
        
        return None
        
    except Exception as e:
        logger.warning(f"Failed to process chart image: {e}")
        return None


def create_brand_image(
    content_lines: List[str],
    chart_image: Optional[Union[BytesIO, bytes, Image.Image]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    accent_color: Optional[Tuple[int, int, int]] = None
) -> BytesIO:
    """
    Create a branded image with text content and optional chart.
    
    Args:
        content_lines: List of text lines to display
        chart_image: Optional chart image (BytesIO, bytes, or PIL Image)
        title: Optional custom title (defaults to brand name)
        subtitle: Optional subtitle (displayed top-right)
        accent_color: Optional override for brand color
    
    Returns:
        BytesIO containing PNG image data
    """
    # Setup
    accent = accent_color or BRAND_PRIMARY
    display_title = title or BRAND_NAME
    
    # Create temporary canvas for measurements
    temp_img = Image.new("RGB", (CANVAS_WIDTH, MIN_CANVAS_HEIGHT))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Process all text lines
    wrapped_lines: List[str] = []
    for line in content_lines:
        # Use current default font for initial wrap estimate
        wrapped = _wrap_text_to_width(temp_draw, str(line), _get_font(DEFAULT_FONT_SIZE), 
                                    CANVAS_WIDTH - 2 * PADDING_X)
        wrapped_lines.extend(wrapped)
    
    # Determine optimal font size
    content_font = _fit_font_to_width(temp_draw, wrapped_lines, CANVAS_WIDTH - 2 * PADDING_X)
    
    # Calculate final dimensions
    chart = _process_chart_image(chart_image)
    chart_height = 0
    if chart:
        chart_width = CANVAS_WIDTH - 2 * PADDING_X
        chart_height = int(chart_width * CHART_RATIO) + 20  # 20px spacing
    
    content_height = len(wrapped_lines) * LINE_HEIGHT
    total_height = max(
        MIN_CANVAS_HEIGHT,
        PADDING_Y + TITLE_HEIGHT + content_height + chart_height + PADDING_Y
    )
    
    # Create final canvas
    img = Image.new("RGB", (CANVAS_WIDTH, total_height), BRAND_BG)
    draw = ImageDraw.Draw(img)
    
    # Draw decorative top bar
    draw.rectangle([0, 0, CANVAS_WIDTH, 4], fill=accent)
    
    # Draw title
    title_font = _get_font(28, bold=True)
    draw.text((PADDING_X, PADDING_Y), display_title, fill=accent, font=title_font)
    
    # Draw subtitle (right-aligned)
    if subtitle:
        sub_font = _get_font(16, bold=False)
        sub_width = _calculate_text_width(draw, subtitle, sub_font)
        sub_x = CANVAS_WIDTH - PADDING_X - int(sub_width)
        draw.text((sub_x, PADDING_Y + 8), subtitle, fill=BRAND_TEXT, font=sub_font)
    
    # Draw content lines
    y_pos = PADDING_Y + TITLE_HEIGHT
    for line in wrapped_lines:
        draw.text((PADDING_X, y_pos), line, fill=BRAND_TEXT, font=content_font)
        y_pos += LINE_HEIGHT
    
    # Draw chart if provided
    if chart:
        try:
            chart_width = CANVAS_WIDTH - 2 * PADDING_X
            chart_height = int(chart_width * CHART_RATIO)
            chart_resized = chart.resize((chart_width, chart_height), Image.Resampling.LANCZOS)
            
            # Add subtle border
            border_pad = 2
            draw.rectangle(
                [PADDING_X - border_pad, y_pos - border_pad,
                 PADDING_X + chart_width + border_pad, y_pos + chart_height + border_pad],
                outline=(40, 45, 55),
                width=1
            )
            
            img.paste(chart_resized, (PADDING_X, y_pos))
        except Exception as e:
            logger.warning(f"Failed to paste chart: {e}")
    
    # Export
    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    
    # Cleanup
    temp_img.close()
    if chart:
        chart.close()
    
    return buffer


def build_signal_image(
    signal_data: dict,
    chart_image: Optional[Union[BytesIO, bytes, Image.Image]] = None
) -> BytesIO:
    """
    Create branded image for trading signal.
    
    Args:
        signal_data: Dict with keys: symbol, interval, signal, entry, sl, tp1, 
                     confidence, suggested_risk_usd, reasons
        chart_image: Optional chart visualization
    """
    sig_type = signal_data.get('signal', 'HOLD')
    
    # Color code by signal type
    accent = BRAND_PRIMARY
    if sig_type == 'LONG':
        accent = (0, 255, 128)  # Green
    elif sig_type == 'SHORT':
        accent = (255, 64, 64)   # Red
    
    lines = [
        f"📊 {signal_data.get('symbol', 'Unknown')}  |  ⏱ {signal_data.get('interval', '1h')}  |  🎯 {sig_type}",
        "",
        f"Entry: {signal_data.get('entry', 'N/A'):.8f}".rstrip('0').rstrip('.') if isinstance(signal_data.get('entry'), (int, float)) else f"Entry: {signal_data.get('entry', 'N/A')}",
        f"Stop Loss: {signal_data.get('sl', 'N/A'):.8f}".rstrip('0').rstrip('.') if isinstance(signal_data.get('sl'), (int, float)) else f"SL: {signal_data.get('sl', 'N/A')}",
        f"Take Profit: {signal_data.get('tp1', 'N/A'):.8f}".rstrip('0').rstrip('.') if isinstance(signal_data.get('tp1'), (int, float)) else f"TP: {signal_data.get('tp1', 'N/A')}",
        "",
        f"Confidence: {int(signal_data.get('confidence', 0) * 100)}%  |  Risk: ${signal_data.get('suggested_risk_usd', 'N/A')}",
    ]
    
    # Add reasons if present
    reasons = signal_data.get('reasons', [])
    if reasons:
        lines.extend(["", "📋 Analysis:"])
        for reason in reasons[:5]:  # Limit to 5 reasons
            lines.append(f"  • {reason}")
    
    return create_brand_image(
        lines,
        chart_image=chart_image,
        title=BRAND_NAME,
        accent_color=accent
    )


def build_text_image(
    message_text: str,
    title: Optional[str] = None
) -> BytesIO:
    """
    Convert plain text to branded image.
    
    Args:
        message_text: Text content (can contain newlines)
        title: Optional custom title
    """
    lines = message_text.splitlines() if message_text else ["No content"]
    return create_brand_image(lines, title=title or BRAND_NAME)


def safe_send_with_image(
    bot,
    chat_id: Union[int, str],
    text: str,
    image_buffer: Optional[BytesIO] = None,
    reply_markup=None
) -> bool:
    """
    Send message as branded image with text fallback.
    
    Args:
        bot: TeleBot instance
        chat_id: Target chat ID
        text: Text content (used for image generation or fallback)
        image_buffer: Pre-generated image (optional)
        reply_markup: Keyboard markup
    
    Returns:
        True if sent successfully, False otherwise
    """
    buffer_to_send = image_buffer
    
    try:
        # Generate image if not provided
        if buffer_to_send is None:
            buffer_to_send = build_text_image(text)
        
        # Send as photo
        bot.send_photo(
            chat_id,
            buffer_to_send,
            caption=f"<i>— {BRAND_NAME}</i>",
            reply_markup=reply_markup,
            parse_mode="HTML"
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to send image: {e}")
        
        # Cleanup failed buffer
        if buffer_to_send and buffer_to_send is not image_buffer:
            buffer_to_send.close()
        
        # Fallback to plain text
        try:
            fallback_text = f"{text}\n\n— {BRAND_NAME}"
            bot.send_message(
                chat_id,
                fallback_text[:4096],  # Telegram limit
                reply_markup=reply_markup,
                parse_mode="HTML"
            )
            return True
        except Exception as e2:
            logger.critical(f"Complete send failure: {e2}")
            return False
    
    finally:
        # Cleanup generated buffer
        if buffer_to_send and buffer_to_send is not image_buffer:
            try:
                buffer_to_send.close()
            except Exception:
                pass


def create_welcome_image(
    welcome_lines: List[str],
    title: str = BRAND_NAME
) -> BytesIO:
    """
    Specialized welcome image with enhanced styling.
    """
    # Add decorative elements
    decorated_lines = [
        "╔" + "═" * 40 + "╗",
        "",
    ] + welcome_lines + [
        "",
        "╚" + "═" * 40 + "╝"
    ]
    
    return create_brand_image(decorated_lines, title=title, accent_color=BRAND_PRIMARY)
               
