from __future__ import annotations

import argparse
from pathlib import Path

from nuclei_benchmark.data.manifest import build_manifest, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MoNuSeg manifest CSV that links raw files and GT masks."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/processed/gt_instances"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/interim/monuseg_manifest.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = build_manifest(
        raw_root=args.raw_root,
        gt_root=args.gt_root,
        splits=("train", "test"),
    )
    save_manifest(manifest, args.output_path)

    print("Manifest build summary")
    print(f"Rows: {len(manifest)}")
    print(f"Train rows: {(manifest['split'] == 'train').sum()}")
    print(f"Test rows: {(manifest['split'] == 'test').sum()}")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()