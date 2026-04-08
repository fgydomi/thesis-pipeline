from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


DEFAULT_MANIFEST_PATH = Path("data/interim/monuseg_manifest.csv")
DEFAULT_PREDICTION_ROOT = Path("outputs/predictions/stardist_manifest")
DEFAULT_METRICS_ROOT = Path("outputs/metrics/stardist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate StarDist predictions against GT instance masks."
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
        default="train",
        choices=["train", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Directory containing StarDist prediction masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where metrics CSV will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of rows to evaluate. Use 0 for no limit.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_manifest_path(path_str: str) -> Path:
    return Path(path_str.replace("\\", "/"))


def load_manifest_rows(manifest_path: Path, split: str, limit: int) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_columns = [
        "image_id",
        "split",
        "gt_instance_path",
        "gt_exists",
    ]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Manifest must contain a '{column}' column.")

    split_df = df[df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split='{split}' in manifest.")

    split_df = split_df[split_df["gt_exists"] == True].copy()
    if split_df.empty:
        raise ValueError(
            f"No rows with gt_exists=True found for split='{split}' in manifest."
        )

    if limit > 0:
        split_df = split_df.head(limit)

    return split_df.to_dict(orient="records")


def load_label_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")
    return tifffile.imread(path)


def to_binary_foreground(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask > 0, dtype=bool)


def compute_binary_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    gt_fg = to_binary_foreground(gt_mask)
    pred_fg = to_binary_foreground(pred_mask)

    intersection = np.logical_and(gt_fg, pred_fg).sum()
    gt_sum = gt_fg.sum()
    pred_sum = pred_fg.sum()

    if gt_sum == 0 and pred_sum == 0:
        return 1.0

    denominator = gt_sum + pred_sum
    if denominator == 0:
        return 0.0

    return float((2.0 * intersection) / denominator)


def compute_binary_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    gt_fg = to_binary_foreground(gt_mask)
    pred_fg = to_binary_foreground(pred_mask)

    intersection = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def validate_same_shape(image_id: str, gt_mask: np.ndarray, pred_mask: np.ndarray) -> None:
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Shape mismatch for {image_id}: gt shape={gt_mask.shape}, pred shape={pred_mask.shape}"
        )


def build_metrics_row(
    image_id: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> dict[str, str]:
    dice = compute_binary_dice(gt_mask, pred_mask)
    iou = compute_binary_iou(gt_mask, pred_mask)

    gt_foreground_pixels = int((gt_mask > 0).sum())
    pred_foreground_pixels = int((pred_mask > 0).sum())

    return {
        "image_id": image_id,
        "status": "success",
        "shape": "x".join(str(dim) for dim in gt_mask.shape),
        "dice_fg": f"{dice:.6f}",
        "iou_fg": f"{iou:.6f}",
        "gt_foreground_pixels": str(gt_foreground_pixels),
        "pred_foreground_pixels": str(pred_foreground_pixels),
    }


def summarize_metrics(metrics_rows: list[dict[str, str]]) -> None:
    successful_rows = [row for row in metrics_rows if row["status"] == "success"]
    failed_rows = [row for row in metrics_rows if row["status"] != "success"]

    print()
    print("Evaluation summary")
    print(f"  Total rows: {len(metrics_rows)}")
    print(f"  Successful: {len(successful_rows)}")
    print(f"  Failed: {len(failed_rows)}")

    if successful_rows:
        dice_values = [float(row["dice_fg"]) for row in successful_rows]
        iou_values = [float(row["iou_fg"]) for row in successful_rows]

        print(f"  Mean Dice (foreground): {statistics.mean(dice_values):.6f}")
        print(f"  Mean IoU  (foreground): {statistics.mean(iou_values):.6f}")
        print(f"  Min Dice  (foreground): {min(dice_values):.6f}")
        print(f"  Max Dice  (foreground): {max(dice_values):.6f}")
        print(f"  Min IoU   (foreground): {min(iou_values):.6f}")
        print(f"  Max IoU   (foreground): {max(iou_values):.6f}")


def main() -> None:
    args = parse_args()

    prediction_dir = (
        args.prediction_dir
        if args.prediction_dir is not None
        else DEFAULT_PREDICTION_ROOT / args.split
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_METRICS_ROOT / args.split
    )
    metrics_csv_path = output_dir / "stardist_metrics.csv"

    ensure_dir(output_dir)

    manifest_rows = load_manifest_rows(
        manifest_path=args.manifest,
        split=args.split,
        limit=args.limit,
    )

    metrics_rows: list[dict[str, str]] = []

    print("Evaluating StarDist predictions")
    print(f"Manifest: {args.manifest}")
    print(f"Split: {args.split}")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Rows selected: {len(manifest_rows)}")
    print()

    for index, row in enumerate(manifest_rows, start=1):
        image_id = str(row["image_id"])
        gt_mask_path = normalize_manifest_path(str(row["gt_instance_path"]))
        pred_mask_path = prediction_dir / f"{image_id}_stardist_inst.tif"

        print(f"[{index}/{len(manifest_rows)}] Evaluating: {image_id}")
        print(f"  GT path:   {gt_mask_path}")
        print(f"  Pred path: {pred_mask_path}")

        try:
            gt_mask = load_label_mask(gt_mask_path)
            pred_mask = load_label_mask(pred_mask_path)

            validate_same_shape(image_id=image_id, gt_mask=gt_mask, pred_mask=pred_mask)

            metrics_row = build_metrics_row(
                image_id=image_id,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
            )
            metrics_rows.append(metrics_row)

            print("  Status: success")
            print(f"  Dice (fg): {metrics_row['dice_fg']}")
            print(f"  IoU  (fg): {metrics_row['iou_fg']}")
            print()

        except Exception as exc:
            metrics_rows.append(
                {
                    "image_id": image_id,
                    "status": f"failed: {exc}",
                    "shape": "",
                    "dice_fg": "",
                    "iou_fg": "",
                    "gt_foreground_pixels": "",
                    "pred_foreground_pixels": "",
                }
            )

            print("  Status: failed")
            print(f"  Error: {exc}")
            print()

    fieldnames = [
        "image_id",
        "status",
        "shape",
        "dice_fg",
        "iou_fg",
        "gt_foreground_pixels",
        "pred_foreground_pixels",
    ]

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    print("StarDist evaluation finished.")
    print(f"Metrics CSV: {metrics_csv_path}")
    summarize_metrics(metrics_rows)


if __name__ == "__main__":
    main()