from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import tifffile

from nuclei_benchmark.data.io import load_tiff_image
from nuclei_benchmark.models.cellpose_model import CellposeModelWrapper


CONFIG_PATH = Path("configs/cellpose.yaml")
IMAGE_DIR = Path("data/raw/monuseg_train/Tissue_Images")
OUTPUT_DIR = Path("outputs/predictions/cellpose_batch")
RUNTIME_LOG_PATH = OUTPUT_DIR / "cellpose_runtime.csv"

MAX_IMAGES = 5


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_label_mask(path: Path, mask) -> None:
    tifffile.imwrite(path, mask)


def iter_image_paths(image_dir: Path, max_images: int) -> list[Path]:
    image_paths = sorted(image_dir.glob("*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No .tif files found in {image_dir}")
    return image_paths[:max_images]


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
        "max_label": str(metadata.get("max_label", "")),
        "num_flow_entries": str(metadata.get("num_flow_entries", "")),
        "num_style_entries": str(metadata.get("num_style_entries", "")),
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
    ensure_dir(OUTPUT_DIR)

    wrapper = CellposeModelWrapper(config_path=CONFIG_PATH)
    image_paths = iter_image_paths(IMAGE_DIR, MAX_IMAGES)

    runtime_rows: list[dict[str, str]] = []

    print(f"Running Cellpose batch on {len(image_paths)} image(s)")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for index, image_path in enumerate(image_paths, start=1):
        image_id = image_path.stem

        print(f"[{index}/{len(image_paths)}] Processing: {image_path.name}")

        image = load_tiff_image(image_path)
        start_time = time.perf_counter()

        try:
            prediction = wrapper.predict(image=image, image_id=image_id)
            elapsed_seconds = time.perf_counter() - start_time

            mask_path = OUTPUT_DIR / f"{image_id}_cellpose_inst.tif"
            metadata_path = OUTPUT_DIR / f"{image_id}_cellpose_metadata.json"

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
        "max_label",
        "num_flow_entries",
        "num_style_entries",
    ]

    with RUNTIME_LOG_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runtime_rows)

    print("Cellpose batch run finished.")
    print(f"Runtime log: {RUNTIME_LOG_PATH}")
    summarize_run(runtime_rows)


if __name__ == "__main__":
    main()