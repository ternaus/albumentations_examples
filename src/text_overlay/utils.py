import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.text_overlay.aug import TextAugmenter

# Create a PCG64 bit generator
bit_gen = np.random.PCG64()

# Create a Generator instance
rng = np.random.Generator(bit_gen)

MIN_TEXT_LENGTH_FOR_REPLACEMENT = 50


def render_text(
    bbox_shape: tuple[int, int],
    text: str,
    font: ImageFont,
    aug: TextAugmenter | None = None,
) -> np.ndarray:
    bbox_height, bbox_width = bbox_shape

    # Create an empty RGB image with the size of the bounding box
    bbox_img = Image.new("RGB", (bbox_width, bbox_height), color="white")
    draw = ImageDraw.Draw(bbox_img)

    if aug is not None:
        old_text = text

        choices = ["swap", "deletion", "insertion", "kreplacement"]
        choice = random.choice(choices)

        if choice == "kreplacement" and len(old_text) <= MIN_TEXT_LENGTH_FOR_REPLACEMENT:
            choice = random.choice(choices[:-1])

        new_text = aug.random_aug(old_text, 0.10, choice)
    else:
        new_text = text

    # Draw the text in red
    draw.text((0, 0), new_text, fill="black", font=font)

    return np.array(bbox_img)


def prepare_metadata(
    image_shape: tuple[int, int],
    bboxes: list[tuple[float, float, float, float]],
    texts: list[str],
    fraction: float,
    font_path: str,
    aug: TextAugmenter | None = None,
) -> list[dict[str, Any]]:
    image_height, image_width = image_shape

    num_texts = len(texts)

    num_lines_to_modify = int(len(texts) * fraction)

    bbox_indices_to_update = rng.choice(range(num_texts), num_lines_to_modify)

    result = []

    for index in bbox_indices_to_update:
        selected_bbox = bboxes[index]
        text = texts[index]

        left, top, width_norm, height_norm = selected_bbox

        bbox_height = int(image_height * height_norm)
        bbox_width = int(image_width * width_norm)

        font = ImageFont.truetype(font_path, int(0.90 * bbox_height))

        overlay_image = render_text((bbox_height, bbox_width), text, font, aug)

        result += [
            {
                "image": overlay_image,
                "bbox": (left, top, left + width_norm, top + height_norm),
            },
        ]

    return result
