from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.color import label2rgb


DEFAULT_MANIFEST_PATH = Path("data/interim/monuseg_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/figures/qualitative_comparison")

CELLPOSE_ROOT = Path("outputs/predictions/cellpose_manifest")
STARDIST_ROOT = Path("outputs/predictions/stardist_manifest")
UNET_ROOT = Path("outputs/predictions/unet_watershed_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready qualitative comparison figures."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to manifest CSV file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of images to process when --image-ids is not provided.",
    )
    parser.add_argument(
        "--image-ids",
        nargs="*",
        default=None,
        help="Optional explicit image IDs to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be saved.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_manifest_path(path_str: str) -> Path:
    return Path(path_str.replace("\\", "/"))


def count_instances(label_mask: np.ndarray) -> int:
    unique_values = np.unique(label_mask)
    return int(np.sum(unique_values > 0))


def make_overlay(image: np.ndarray, label_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = label2rgb(label_mask, bg_label=0, bg_color=(0, 0, 0))
    image_float = image.astype(np.float32) / 255.0
    overlay = (1.0 - alpha) * image_float + alpha * colored
    return np.clip(overlay, 0.0, 1.0)


def load_manifest_rows(
    manifest_path: Path,
    split: str,
    image_ids: list[str] | None,
    limit: int,
) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_columns = ["image_id", "split", "image_path", "gt_instance_path"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Manifest must contain a '{column}' column.")

    df = df[df["split"] == split].copy()
    if df.empty:
        raise ValueError(f"No rows found for split='{split}'.")

    if image_ids:
        df = df[df["image_id"].isin(image_ids)].copy()
        if df.empty:
            raise ValueError("None of the requested image IDs were found in the manifest.")
    else:
        if limit > 0:
            df = df.head(limit)

    return df.to_dict(orient="records")


def load_prediction_mask(root: Path, split: str, filename: str) -> np.ndarray:
    path = root / split / filename
    if not path.exists():
        raise FileNotFoundError(f"Prediction mask not found: {path}")
    return tifffile.imread(path)


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir / args.split
    ensure_dir(output_dir)

    rows = load_manifest_rows(
        manifest_path=args.manifest,
        split=args.split,
        image_ids=args.image_ids,
        limit=args.limit,
    )

    print("Generating qualitative comparison figures")
    print(f"Split: {args.split}")
    print(f"Images selected: {len(rows)}")
    print(f"Output directory: {output_dir}")
    print()

    for index, row in enumerate(rows, start=1):
        image_id = str(row["image_id"])
        image_path = normalize_manifest_path(str(row["image_path"]))
        gt_path = normalize_manifest_path(str(row["gt_instance_path"]))

        print(f"[{index}/{len(rows)}] Processing: {image_id}")

        image = tifffile.imread(image_path)
        gt_mask = tifffile.imread(gt_path)

        cellpose_mask = load_prediction_mask(
            CELLPOSE_ROOT,
            args.split,
            f"{image_id}_cellpose_inst.tif",
        )
        stardist_mask = load_prediction_mask(
            STARDIST_ROOT,
            args.split,
            f"{image_id}_stardist_inst.tif",
        )
        unet_mask = load_prediction_mask(
            UNET_ROOT,
            args.split,
            f"{image_id}_unet_watershed_inst.tif",
        )

        gt_count = count_instances(gt_mask)
        cellpose_count = count_instances(cellpose_mask)
        stardist_count = count_instances(stardist_mask)
        unet_count = count_instances(unet_mask)

        gt_overlay = make_overlay(image, gt_mask)
        cellpose_overlay = make_overlay(image, cellpose_mask)
        stardist_overlay = make_overlay(image, stardist_mask)
        unet_overlay = make_overlay(image, unet_mask)

        figure = plt.figure(figsize=(16, 10))

        ax1 = figure.add_subplot(2, 3, 1)
        ax1.imshow(image)
        ax1.set_title("Original image")
        ax1.axis("off")

        ax2 = figure.add_subplot(2, 3, 2)
        ax2.imshow(gt_overlay)
        ax2.set_title(f"Ground truth overlay\ncount={gt_count}")
        ax2.axis("off")

        ax3 = figure.add_subplot(2, 3, 3)
        ax3.imshow(cellpose_overlay)
        ax3.set_title(f"Cellpose overlay\ncount={cellpose_count}")
        ax3.axis("off")

        ax4 = figure.add_subplot(2, 3, 4)
        ax4.imshow(stardist_overlay)
        ax4.set_title(f"StarDist overlay\ncount={stardist_count}")
        ax4.axis("off")

        ax5 = figure.add_subplot(2, 3, 5)
        ax5.imshow(unet_overlay)
        ax5.set_title(f"U-Net + Watershed overlay\ncount={unet_count}")
        ax5.axis("off")

        ax6 = figure.add_subplot(2, 3, 6)
        ax6.axis("off")
        ax6.text(
            0.0,
            0.95,
            "\n".join(
                [
                    f"Image ID: {image_id}",
                    f"GT count: {gt_count}",
                    f"Cellpose count: {cellpose_count}",
                    f"StarDist count: {stardist_count}",
                    f"U-Net + Watershed count: {unet_count}",
                ]
            ),
            va="top",
            ha="left",
            fontsize=11,
        )

        figure.suptitle(
            f"Qualitative comparison on MoNuSeg ({args.split}): {image_id}",
            fontsize=14,
        )
        figure.tight_layout()

        output_path = output_dir / f"{image_id}_qualitative_comparison.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

        print(f"  Saved: {output_path}")

    print()
    print("Done.")
    print("Next step: inspect the generated figures and choose 2–3 representative cases.")
    

if __name__ == "__main__":
    main()