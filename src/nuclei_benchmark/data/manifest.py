from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from nuclei_benchmark.data.dataset import SplitName, summarize_split


@dataclass(frozen=True)
class ManifestRow:
    """One row in the MoNuSeg manifest table."""

    image_id: str
    split: str
    image_path: str
    annotation_path: str
    gt_instance_path: str
    image_exists: bool
    annotation_exists: bool
    gt_exists: bool


def build_split_manifest(
    raw_root: Path,
    gt_root: Path,
    split: SplitName,
) -> list[ManifestRow]:
    """
    Build manifest rows for one split.

    The manifest links raw images, XML annotations, and generated GT instance masks.
    """

    summary = summarize_split(raw_root, split)
    rows: list[ManifestRow] = []

    for pair in summary.pairs:
        gt_path = gt_root / split / f"{pair.image_id}_inst.tif"

        rows.append(
            ManifestRow(
                image_id=pair.image_id,
                split=split,
                image_path=str(pair.image_path),
                annotation_path=str(pair.annotation_path),
                gt_instance_path=str(gt_path),
                image_exists=pair.image_path.exists(),
                annotation_exists=pair.annotation_path.exists(),
                gt_exists=gt_path.exists(),
            )
        )

    return rows


def build_manifest(
    raw_root: Path,
    gt_root: Path,
    splits: tuple[SplitName, ...] = ("train", "test"),
) -> pd.DataFrame:
    """Build the full MoNuSeg manifest across the requested splits."""

    rows: list[ManifestRow] = []
    for split in splits:
        rows.extend(build_split_manifest(raw_root=raw_root, gt_root=gt_root, split=split))

    manifest = pd.DataFrame(asdict(row) for row in rows)
    manifest = manifest.sort_values(["split", "image_id"]).reset_index(drop=True)
    return manifest


def save_manifest(manifest: pd.DataFrame, output_path: Path) -> None:
    """Save the manifest as CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)