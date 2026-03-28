from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile as tiff
from lxml import etree


def load_tiff_image(image_path: Path) -> np.ndarray:
    """Load a TIFF image from disk."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = tiff.imread(str(image_path))
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported TIFF shape {image.shape} for file: {image_path}")

    return image


def get_image_hw(image: np.ndarray) -> tuple[int, int]:
    """Return image height and width for 2D or 3D image arrays."""

    if image.ndim == 2:
        return image.shape

    if image.ndim == 3:
        return image.shape[0], image.shape[1]

    raise ValueError(f"Unsupported image shape: {image.shape}")


def parse_monuseg_xml_polygons(xml_path: Path) -> list[np.ndarray]:
    """
    Parse MoNuSeg XML annotations into polygon coordinate arrays.

    Each polygon is returned as a float32 NumPy array of shape (N, 2),
    where columns are ordered as (x, y).
    """

    if not xml_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {xml_path}")

    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    polygons: list[np.ndarray] = []

    for region in root.findall(".//Region"):
        vertices = region.findall(".//Vertices/Vertex")
        if not vertices:
            continue

        coords: list[tuple[float, float]] = []
        for vertex in vertices:
            x = vertex.get("X")
            y = vertex.get("Y")
            if x is None or y is None:
                continue

            coords.append((float(x), float(y)))

        if len(coords) >= 3:
            polygons.append(np.asarray(coords, dtype=np.float32))

    return polygons