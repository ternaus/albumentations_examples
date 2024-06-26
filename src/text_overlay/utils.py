import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create a PCG64 bit generator
bit_gen = np.random.PCG64()

# Create a Generator instance
rng = np.random.Generator(bit_gen)

MIN_TEXT_LENGTH_FOR_REPLACEMENT = 50


def render_text(
    bbox_shape: tuple[int, int],
    text: str,
    font: ImageFont,
) -> np.ndarray:
    bbox_height, bbox_width = bbox_shape

    # Create an empty RGB image with the size of the bounding box
    bbox_img = Image.new("RGB", (bbox_width, bbox_height), color="white")
    draw = ImageDraw.Draw(bbox_img)

    # Draw the text in red
    draw.text((0, 0), text, fill="red", font=font)

    return np.array(bbox_img)


def normalized_top_left2albumentations(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_min, y_min, width, height = bbox
    return x_min, y_min, x_min + width, y_min + height
