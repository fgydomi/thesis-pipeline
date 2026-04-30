from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Rectangle
from skimage import measure


DEFAULT_MANIFEST_PATH = Path("data/interim/monuseg_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/figures/qualitative_roi")

CELLPOSE_ROOT = Path("outputs/predictions/cellpose_manifest")
STARDIST_ROOT = Path("outputs/predictions/stardist_manifest")
UNET_ROOT = Path("outputs/predictions/unet_watershed_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ROI-based qualitative comparison figures for the thesis."
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
        help="Dataset split.",
    )
    parser.add_argument(
        "--image-id",
        type=str,
        required=True,
        help="Image ID to visualize.",
    )
    parser.add_argument("--x", type=int, required=True, help="ROI top-left x coordinate.")
    parser.add_argument("--y", type=int, required=True, help="ROI top-left y coordinate.")
    parser.add_argument("--w", type=int, required=True, help="ROI width.")
    parser.add_argument("--h", type=int, required=True, help="ROI height.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved figures.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_manifest_path(path_str: str) -> Path:
    return Path(path_str.replace("\\", "/"))


def load_manifest_row(manifest_path: Path, split: str, image_id: str) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_columns = ["image_id", "split", "image_path", "gt_instance_path"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Manifest must contain a '{column}' column.")

    row_df = df[(df["split"] == split) & (df["image_id"] == image_id)].copy()
    if row_df.empty:
        raise ValueError(f"Image ID '{image_id}' not found in split '{split}'.")

    return row_df.iloc[0].to_dict()


def load_prediction_mask(root: Path, split: str, filename: str) -> np.ndarray:
    path = root / split / filename
    if not path.exists():
        raise FileNotFoundError(f"Prediction mask not found: {path}")
    return tifffile.imread(path)


def crop_image(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return image[y : y + h, x : x + w]


def crop_mask(mask: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return mask[y : y + h, x : x + w]


def count_instances(mask: np.ndarray) -> int:
    unique_values = np.unique(mask)
    return int(np.sum(unique_values > 0))


def draw_instance_contours(ax, image_crop: np.ndarray, mask_crop: np.ndarray, color: str) -> None:
    ax.imshow(image_crop)

    instance_ids = np.unique(mask_crop)
    instance_ids = instance_ids[instance_ids > 0]

    for instance_id in instance_ids:
        binary_mask = mask_crop == instance_id
        contours = measure.find_contours(binary_mask.astype(np.uint8), level=0.5)

        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.0)

    ax.axis("off")


def validate_roi(image: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    height, width = image.shape[:2]

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI: x={x}, y={y}, w={w}, h={h}")

    if x + w > width or y + h > height:
        raise ValueError(
            f"ROI exceeds image bounds. Image shape={image.shape}, ROI=({x}, {y}, {w}, {h})"
        )


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir / args.split
    ensure_dir(output_dir)

    row = load_manifest_row(
        manifest_path=args.manifest,
        split=args.split,
        image_id=args.image_id,
    )

    image_path = normalize_manifest_path(str(row["image_path"]))
    gt_path = normalize_manifest_path(str(row["gt_instance_path"]))

    image = tifffile.imread(image_path)
    gt_mask = tifffile.imread(gt_path)

    validate_roi(image=image, x=args.x, y=args.y, w=args.w, h=args.h)

    cellpose_mask = load_prediction_mask(
        CELLPOSE_ROOT,
        args.split,
        f"{args.image_id}_cellpose_inst.tif",
    )
    stardist_mask = load_prediction_mask(
        STARDIST_ROOT,
        args.split,
        f"{args.image_id}_stardist_inst.tif",
    )
    unet_mask = load_prediction_mask(
        UNET_ROOT,
        args.split,
        f"{args.image_id}_unet_watershed_inst.tif",
    )

    image_crop = crop_image(image, args.x, args.y, args.w, args.h)
    gt_crop = crop_mask(gt_mask, args.x, args.y, args.w, args.h)
    cellpose_crop = crop_mask(cellpose_mask, args.x, args.y, args.w, args.h)
    stardist_crop = crop_mask(stardist_mask, args.x, args.y, args.w, args.h)
    unet_crop = crop_mask(unet_mask, args.x, args.y, args.w, args.h)

    gt_count = count_instances(gt_crop)
    cellpose_count = count_instances(cellpose_crop)
    stardist_count = count_instances(stardist_crop)
    unet_count = count_instances(unet_crop)

    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, 4, height_ratios=[1.0, 1.1])

    ax_context = figure.add_subplot(grid[0, :])
    ax_context.imshow(image)
    ax_context.add_patch(
        Rectangle(
            (args.x, args.y),
            args.w,
            args.h,
            fill=False,
            edgecolor="red",
            linewidth=2.0,
        )
    )
    ax_context.set_title(
        f"Original image with selected ROI: {args.image_id}",
        fontsize=13,
    )
    ax_context.axis("off")

    ax_gt = figure.add_subplot(grid[1, 0])
    draw_instance_contours(ax_gt, image_crop, gt_crop, color="lime")
    ax_gt.set_title(f"Ground truth\nROI count={gt_count}")

    ax_cellpose = figure.add_subplot(grid[1, 1])
    draw_instance_contours(ax_cellpose, image_crop, cellpose_crop, color="deepskyblue")
    ax_cellpose.set_title(f"Cellpose\nROI count={cellpose_count}")

    ax_stardist = figure.add_subplot(grid[1, 2])
    draw_instance_contours(ax_stardist, image_crop, stardist_crop, color="orange")
    ax_stardist.set_title(f"StarDist\nROI count={stardist_count}")

    ax_unet = figure.add_subplot(grid[1, 3])
    draw_instance_contours(ax_unet, image_crop, unet_crop, color="magenta")
    ax_unet.set_title(f"U-Net + Watershed\nROI count={unet_count}")

    figure.suptitle(
        (
            f"Qualitative ROI comparison on MoNuSeg ({args.split})\n"
            f"ROI = x:{args.x}, y:{args.y}, w:{args.w}, h:{args.h}"
        ),
        fontsize=14,
    )
    figure.tight_layout()

    output_path = output_dir / (
        f"{args.image_id}_x{args.x}_y{args.y}_w{args.w}_h{args.h}_roi_comparison.png"
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print("ROI qualitative figure finished.")
    print(f"Image ID: {args.image_id}")
    print(f"ROI: x={args.x}, y={args.y}, w={args.w}, h={args.h}")
    print(f"GT ROI count: {gt_count}")
    print(f"Cellpose ROI count: {cellpose_count}")
    print(f"StarDist ROI count: {stardist_count}")
    print(f"U-Net + Watershed ROI count: {unet_count}")
    print(f"Output figure path: {output_path}")


if __name__ == "__main__":
    main()