from __future__ import annotations

from pathlib import Path

from nuclei_benchmark.data.dataset import summarize_split


RAW_ROOT = Path("data/raw")


def print_summary(split: str) -> None:
    summary = summarize_split(RAW_ROOT, split)

    print(f"== MoNuSeg {split.upper()} ==")
    print(f"Images (.tif): {summary.image_count}")
    print(f"Annotations (.xml): {summary.annotation_count}")
    print(f"Paired: {summary.paired_count}")

    if summary.missing_annotations:
        print(
            f"Missing XML for {len(summary.missing_annotations)} images "
            f"(showing up to 5): {summary.missing_annotations[:5]}"
        )

    if summary.missing_images:
        print(
            f"Missing image for {len(summary.missing_images)} XML files "
            f"(showing up to 5): {summary.missing_images[:5]}"
        )

    print()


def main() -> None:
    print_summary("train")
    print_summary("test")
    print("OK: dataset structure looks consistent.")


if __name__ == "__main__":
    main()