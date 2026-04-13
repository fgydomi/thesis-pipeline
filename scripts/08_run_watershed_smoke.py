from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from nuclei_benchmark.postprocessing.watershed import foreground_to_instances


GT_MASK_PATH = Path("data/processed/gt_instances/train/TCGA-18-5592-01Z-00-DX1_inst.tif")
OUTPUT_DIR = Path("outputs/predictions/watershed_smoke")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    if not GT_MASK_PATH.exists():
        raise FileNotFoundError(f"GT mask file not found: {GT_MASK_PATH}")

    image_id = GT_MASK_PATH.stem.replace("_inst", "")
    gt_mask = tifffile.imread(GT_MASK_PATH)

    foreground_map = np.asarray(gt_mask > 0, dtype=np.float32)

    instance_mask = foreground_to_instances(
        foreground_map=foreground_map,
        threshold=0.5,
        min_size=16,
        min_distance=5,
    )

    output_mask_path = OUTPUT_DIR / f"{image_id}_watershed_inst.tif"
    output_metadata_path = OUTPUT_DIR / f"{image_id}_watershed_metadata.json"

    tifffile.imwrite(output_mask_path, instance_mask)

    metadata = {
        "image_id": image_id,
        "source": "gt_binary_foreground",
        "input_shape": list(gt_mask.shape),
        "input_foreground_pixels": int((foreground_map > 0).sum()),
        "output_max_label": int(instance_mask.max()),
        "output_unique_labels": int(len(np.unique(instance_mask))),
        "threshold": 0.5,
        "min_size": 16,
        "min_distance": 5,
    }

    save_json(output_metadata_path, metadata)

    print("Watershed smoke test finished.")
    print(f"Image ID: {image_id}")
    print(f"Input shape: {gt_mask.shape}")
    print(f"Foreground pixels: {int((foreground_map > 0).sum())}")
    print(f"Output mask path: {output_mask_path}")
    print(f"Output metadata path: {output_metadata_path}")
    print(f"Output max label: {int(instance_mask.max())}")
    print(f"Output unique labels: {int(len(np.unique(instance_mask)))}")


if __name__ == "__main__":
    main()