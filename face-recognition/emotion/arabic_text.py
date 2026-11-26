"""Arabic text rendering on OpenCV frames.

``cv2.putText`` cannot render Arabic: it has no glyph shaping or right-to-left
support, so Arabic comes out as disconnected letters in the wrong order (or as
boxes). This helper renders Arabic properly with Pillow + arabic_reshaper and
blends the result onto a BGR frame.
"""

import os
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np


FONT_CANDIDATES = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
)


def _find_font() -> Optional[str]:
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def shape_arabic(text: str) -> str:
    """Join Arabic letters and reorder them for left-to-right drawing."""

    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text))


@lru_cache(maxsize=128)
def render_text(
    text: str,
    font_size: int = 30,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Render `text` into an RGBA array (cached per text/size/color)."""

    from PIL import Image, ImageDraw, ImageFont

    font_path = _find_font()

    if font_path is None:
        raise RuntimeError("No Arabic-capable system font found")

    font = ImageFont.truetype(font_path, font_size)
    shaped = shape_arabic(text)

    left, top, right, bottom = font.getbbox(shaped)
    padding = 6

    width = max(right - left, 1) + padding * 2
    height = max(bottom - top, 1) + padding * 2

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # `color` is BGR (OpenCV order); Pillow wants RGB.
    draw.text(
        (padding - left, padding - top),
        shaped,
        font=font,
        fill=(color[2], color[1], color[0], 255),
    )

    return np.array(image)


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: Optional[Tuple[int, int]] = None,
    font_size: int = 30,
    color: Sequence[int] = (0, 255, 0),
) -> np.ndarray:
    """Draw Arabic `text` onto `frame` (BGR, in place) and return the frame.

    `origin` is the top-left corner; by default the text is placed at the
    bottom-left corner of the frame.
    """

    rgba = render_text(text, font_size, tuple(int(c) for c in color))

    height, width = rgba.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    if origin is None:
        origin = (10, frame_height - height - 10)

    x = max(0, min(int(origin[0]), frame_width - width))
    y = max(0, min(int(origin[1]), frame_height - height))

    roi = frame[y:y + height, x:x + width]

    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    # RGBA -> BGR
    bgr = rgba[:, :, :3][:, :, ::-1].astype(np.float32)

    blended = roi.astype(np.float32) * (1.0 - alpha) + bgr * alpha
    roi[:] = blended.astype(np.uint8)

    return frame
