from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nuclei_benchmark.data.ground_truth import convert_split_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MoNuSeg XML annotations to instance label TIFF masks."
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/gt_instances"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N paired samples for quick testing.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    summary = convert_split_ground_truth(
        raw_root=args.raw_root,
        split=args.split,
        output_root=args.output_root,
        limit=args.limit,
    )

    print()
    print("GT conversion summary")
    print(f"Split: {summary.split}")
    print(f"Converted images: {summary.converted_images}")
    print(f"Total instances: {summary.total_instances}")
    print(f"Output directory: {summary.output_dir}")


if __name__ == "__main__":
    main()