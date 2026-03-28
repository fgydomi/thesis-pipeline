from __future__ import annotations

from pathlib import Path

import numpy as np
import imageio.v3 as iio
from skimage.segmentation import find_boundaries


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or RGB image to uint8 RGB for visualization."""

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3:
        if image.shape[2] >= 3:
            image = image[:, :, :3]
        else:
            raise ValueError(f"Unsupported channel count for image shape: {image.shape}")
    else:
        raise ValueError(f"Unsupported image shape for visualization: {image.shape}")

    if image.dtype == np.uint8:
        return image.copy()

    image = image.astype(np.float32)
    value_min = float(image.min())
    value_max = float(image.max())

    if value_max <= value_min:
        return np.zeros_like(image, dtype=np.uint8)

    scaled = (255.0 * (image - value_min) / (value_max - value_min)).clip(0, 255)
    return scaled.astype(np.uint8)


def make_label_boundary_overlay(
    image: np.ndarray,
    label_image: np.ndarray,
    boundary_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Overlay instance boundaries on top of the input image."""

    if label_image.ndim != 2:
        raise ValueError("Label image must be 2D.")

    overlay = to_uint8_rgb(image)
    boundaries = find_boundaries(label_image, mode="outer")

    overlay[boundaries] = np.asarray(boundary_color, dtype=np.uint8)
    return overlay


def save_overlay_png(overlay: np.ndarray, output_path: Path) -> None:
    """Save an RGB overlay as PNG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, overlay)