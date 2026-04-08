from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import pandas as pd
import tifffile

from nuclei_benchmark.data.io import load_tiff_image
from nuclei_benchmark.models.stardist_model import StarDistModelWrapper


DEFAULT_CONFIG_PATH = Path("configs/stardist.yaml")
DEFAULT_MANIFEST_PATH = Path("data/interim/monuseg_manifest.csv")
DEFAULT_OUTPUT_ROOT = Path("outputs/predictions/stardist_manifest")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_label_mask(path: Path, mask) -> None:
    tifffile.imwrite(path, mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StarDist inference from a manifest CSV."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to manifest CSV file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to StarDist YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of images to process. Use 0 for no limit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, a split-specific folder is used.",
    )
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path, split: str, limit: int) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_columns = [
        "image_id",
        "split",
        "image_path",
        "image_exists",
    ]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Manifest must contain a '{column}' column.")

    split_df = df[df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split='{split}' in manifest.")

    split_df = split_df[split_df["image_exists"] == True].copy()
    if split_df.empty:
        raise ValueError(
            f"No rows with image_exists=True found for split='{split}' in manifest."
        )

    if limit > 0:
        split_df = split_df.head(limit)

    return split_df.to_dict(orient="records")


def build_runtime_row(
    image_id: str,
    image_shape: tuple[int, ...],
    elapsed_seconds: float,
    prediction,
    status: str,
) -> dict[str, str]:
    metadata = prediction.metadata if prediction is not None else {}

    return {
        "image_id": image_id,
        "status": status,
        "image_shape": "x".join(str(dim) for dim in image_shape),
        "elapsed_seconds": f"{elapsed_seconds:.6f}",
        "device_requested": str(metadata.get("device_requested", "")),
        "device_resolved": str(metadata.get("device_resolved", "")),
        "num_visible_gpus": str(metadata.get("num_visible_gpus", "")),
        "max_label": str(metadata.get("max_label", "")),
        "num_polygons": str(metadata.get("num_polygons", "")),
    }


def summarize_run(runtime_rows: list[dict[str, str]]) -> None:
    total_images = len(runtime_rows)
    successful_rows = [row for row in runtime_rows if row["status"] == "success"]
    failed_rows = [row for row in runtime_rows if row["status"] != "success"]

    print()
    print("Run summary")
    print(f"  Total images: {total_images}")
    print(f"  Successful: {len(successful_rows)}")
    print(f"  Failed: {len(failed_rows)}")

    if successful_rows:
        successful_times = [float(row["elapsed_seconds"]) for row in successful_rows]
        print(f"  Average runtime: {statistics.mean(successful_times):.4f} s")
        print(f"  Minimum runtime: {min(successful_times):.4f} s")
        print(f"  Maximum runtime: {max(successful_times):.4f} s")


def main() -> None:
    args = parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / args.split
    )
    runtime_log_path = output_dir / "stardist_runtime.csv"

    ensure_dir(output_dir)

    wrapper = StarDistModelWrapper(config_path=args.config)
    manifest_rows = load_manifest_rows(
        manifest_path=args.manifest,
        split=args.split,
        limit=args.limit,
    )

    runtime_rows: list[dict[str, str]] = []

    print("Running StarDist from manifest")
    print(f"Manifest: {args.manifest}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Rows selected: {len(manifest_rows)}")
    print(f"Output directory: {output_dir}")
    print()

    for index, row in enumerate(manifest_rows, start=1):
        image_id = str(row["image_id"])
        image_path = Path(str(row["image_path"]).replace("\\", "/"))

        print(f"[{index}/{len(manifest_rows)}] Processing: {image_id}")
        print(f"  Image path: {image_path}")

        image = load_tiff_image(image_path)
        start_time = time.perf_counter()

        try:
            prediction = wrapper.predict(image=image, image_id=image_id)
            elapsed_seconds = time.perf_counter() - start_time

            mask_path = output_dir / f"{image_id}_stardist_inst.tif"
            metadata_path = output_dir / f"{image_id}_stardist_metadata.json"

            save_label_mask(mask_path, prediction.instance_mask)
            save_json(metadata_path, prediction.metadata)

            runtime_rows.append(
                build_runtime_row(
                    image_id=image_id,
                    image_shape=image.shape,
                    elapsed_seconds=elapsed_seconds,
                    prediction=prediction,
                    status="success",
                )
            )

            print("  Status: success")
            print(f"  Time: {elapsed_seconds:.4f} s")
            print(f"  Mask path: {mask_path}")
            print(f"  Metadata path: {metadata_path}")
            print(f"  Max label: {prediction.metadata.get('max_label')}")
            print()

        except Exception as exc:
            elapsed_seconds = time.perf_counter() - start_time

            runtime_rows.append(
                build_runtime_row(
                    image_id=image_id,
                    image_shape=image.shape,
                    elapsed_seconds=elapsed_seconds,
                    prediction=None,
                    status=f"failed: {exc}",
                )
            )

            print("  Status: failed")
            print(f"  Time before failure: {elapsed_seconds:.4f} s")
            print(f"  Error: {exc}")
            print()

    fieldnames = [
        "image_id",
        "status",
        "image_shape",
        "elapsed_seconds",
        "device_requested",
        "device_resolved",
        "num_visible_gpus",
        "max_label",
        "num_polygons",
    ]

    with runtime_log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runtime_rows)

    print("StarDist manifest run finished.")
    print(f"Runtime log: {runtime_log_path}")
    summarize_run(runtime_rows)


if __name__ == "__main__":
    main()