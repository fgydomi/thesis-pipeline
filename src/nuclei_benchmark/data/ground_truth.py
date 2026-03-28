from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage.draw import polygon

from nuclei_benchmark.data.dataset import DatasetPair, SplitName, summarize_split
from nuclei_benchmark.data.io import get_image_hw, load_tiff_image, parse_monuseg_xml_polygons

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundTruthRecord:
    """Result of converting one MoNuSeg annotation into an instance label mask."""

    image_id: str
    output_path: Path
    instance_count: int


@dataclass(frozen=True)
class GroundTruthConversionSummary:
    """Summary of GT conversion for a dataset split."""

    split: SplitName
    converted_images: int
    total_instances: int
    output_dir: Path
    records: list[GroundTruthRecord]


def polygons_to_instance_label(
    polygons: list[np.ndarray],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """
    Rasterize polygon annotations into a single instance label image.

    Output convention:
    - background = 0
    - nucleus IDs = 1..N
    """

    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image dimensions must be positive.")

    if len(polygons) > np.iinfo(np.uint16).max:
        raise ValueError("Too many instances for uint16 label encoding.")

    label_image = np.zeros((image_height, image_width), dtype=np.uint16)

    for instance_id, polygon_coords in enumerate(polygons, start=1):
        if polygon_coords.ndim != 2 or polygon_coords.shape[1] != 2:
            raise ValueError(
                "Each polygon must be a NumPy array of shape (N, 2) with columns (x, y)."
            )

        x_coords = polygon_coords[:, 0]
        y_coords = polygon_coords[:, 1]

        rr, cc = polygon(y_coords, x_coords, shape=(image_height, image_width))
        label_image[rr, cc] = instance_id

    return label_image


def save_instance_label(label_image: np.ndarray, output_path: Path) -> None:
    """Save an instance label image as TIFF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(output_path), label_image)


def convert_pair_to_instance_mask(pair: DatasetPair, output_dir: Path) -> GroundTruthRecord:
    """Convert a single MoNuSeg image/XML pair into an instance label TIFF."""

    image = load_tiff_image(pair.image_path)
    image_height, image_width = get_image_hw(image)

    polygons = parse_monuseg_xml_polygons(pair.annotation_path)
    label_image = polygons_to_instance_label(polygons, image_height, image_width)

    output_path = output_dir / f"{pair.image_id}_inst.tif"
    save_instance_label(label_image, output_path)

    instance_count = int(label_image.max())
    return GroundTruthRecord(
        image_id=pair.image_id,
        output_path=output_path,
        instance_count=instance_count,
    )


def convert_split_ground_truth(
    raw_root: Path,
    split: SplitName,
    output_root: Path,
    limit: int | None = None,
) -> GroundTruthConversionSummary:
    """
    Convert all paired annotations in the requested split to instance label masks.

    Parameters
    ----------
    raw_root:
        Root data directory, typically `data/raw`.
    split:
        Either `train` or `test`.
    output_root:
        Root output directory for processed GT masks.
    limit:
        Optional limit for quick testing on the first N images.
    """

    pairing_summary = summarize_split(raw_root, split)
    pairs = pairing_summary.pairs[:limit] if limit is not None else pairing_summary.pairs

    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting GT conversion for split=%s | paired_images=%s | limit=%s",
        split,
        pairing_summary.paired_count,
        limit,
    )

    records: list[GroundTruthRecord] = []
    total_instances = 0

    for pair in pairs:
        record = convert_pair_to_instance_mask(pair, output_dir)
        records.append(record)
        total_instances += record.instance_count

        logger.info(
            "Converted %s -> %s | instances=%s",
            pair.image_id,
            record.output_path.name,
            record.instance_count,
        )

    summary = GroundTruthConversionSummary(
        split=split,
        converted_images=len(records),
        total_instances=total_instances,
        output_dir=output_dir,
        records=records,
    )

    logger.info(
        "Finished GT conversion for split=%s | converted_images=%s | total_instances=%s",
        summary.split,
        summary.converted_images,
        summary.total_instances,
    )

    return summary