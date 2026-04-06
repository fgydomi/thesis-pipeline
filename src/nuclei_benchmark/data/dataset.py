from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SplitName = Literal["train", "test"]


@dataclass(frozen=True)
class DatasetPair:
    """A matched image-annotation pair for one MoNuSeg sample."""

    image_id: str
    image_path: Path
    annotation_path: Path


@dataclass(frozen=True)
class PairingSummary:
    """Summary of image/XML pairing for one split."""

    split: SplitName
    image_count: int
    annotation_count: int
    paired_count: int
    missing_annotations: list[str]
    missing_images: list[str]
    pairs: list[DatasetPair]


def get_split_dirs(raw_root: Path, split: SplitName) -> tuple[Path, Path]:
    """Return the image and annotation directories for a split."""

    split_root = raw_root / f"monuseg_{split}"
    images_dir = split_root / "Tissue_Images"
    annotations_dir = split_root / "Annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    return images_dir, annotations_dir


def summarize_split(raw_root: Path, split: SplitName) -> PairingSummary:
    """Build a pairing summary for the requested MoNuSeg split."""

    images_dir, annotations_dir = get_split_dirs(raw_root, split)

    image_paths = sorted(images_dir.glob("*.tif"))
    annotation_paths = sorted(annotations_dir.glob("*.xml"))

    image_map = {path.stem: path for path in image_paths}
    annotation_map = {path.stem: path for path in annotation_paths}

    image_ids = set(image_map)
    annotation_ids = set(annotation_map)

    paired_ids = sorted(image_ids & annotation_ids)
    missing_annotations = sorted(image_ids - annotation_ids)
    missing_images = sorted(annotation_ids - image_ids)

    pairs = [
        DatasetPair(
            image_id=image_id,
            image_path=image_map[image_id],
            annotation_path=annotation_map[image_id],
        )
        for image_id in paired_ids
    ]

    return PairingSummary(
        split=split,
        image_count=len(image_paths),
        annotation_count=len(annotation_paths),
        paired_count=len(pairs),
        missing_annotations=missing_annotations,
        missing_images=missing_images,
        pairs=pairs,
    )