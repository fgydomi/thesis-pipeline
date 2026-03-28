from __future__ import annotations

import argparse
from pathlib import Path

import tifffile as tiff

from nuclei_benchmark.data.dataset import summarize_split
from nuclei_benchmark.data.io import load_tiff_image
from nuclei_benchmark.visualization.overlays import (
    make_label_boundary_overlay,
    save_overlay_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GT boundary overlay for one MoNuSeg image."
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--image-id", type=str, default=None)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("data/processed/gt_instances"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/figures/gt_overlays"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = summarize_split(args.raw_root, args.split)
    pairs = summary.pairs
    if not pairs:
        raise RuntimeError(f"No paired samples found for split={args.split}")

    if args.image_id is None:
        pair = pairs[0]
    else:
        matches = [p for p in pairs if p.image_id == args.image_id]
        if not matches:
            raise ValueError(f"Image ID not found in split '{args.split}': {args.image_id}")
        pair = matches[0]

    gt_path = args.gt_root / args.split / f"{pair.image_id}_inst.tif"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth instance mask not found: {gt_path}\n"
            f"Run the GT conversion step first."
        )

    image = load_tiff_image(pair.image_path)
    label_image = tiff.imread(str(gt_path))

    overlay = make_label_boundary_overlay(image=image, label_image=label_image)

    output_path = args.output_root / args.split / f"{pair.image_id}_gt_overlay.png"
    save_overlay_png(overlay, output_path)

    print(f"Saved overlay: {output_path}")


if __name__ == "__main__":
    main()