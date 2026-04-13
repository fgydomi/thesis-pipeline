from __future__ import annotations

import json
from pathlib import Path

import tifffile

from nuclei_benchmark.data.io import load_tiff_image
from nuclei_benchmark.models.unet_watershed_model import UNetWatershedModelWrapper


CONFIG_PATH = Path("configs/unet_watershed.yaml")
IMAGE_PATH = Path("data/raw/monuseg_train/Tissue_Images/TCGA-18-5592-01Z-00-DX1.tif")
OUTPUT_DIR = Path("outputs/predictions/unet_watershed_smoke")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

    image_id = IMAGE_PATH.stem
    image = load_tiff_image(IMAGE_PATH)

    wrapper = UNetWatershedModelWrapper(config_path=CONFIG_PATH)
    prediction = wrapper.predict(image=image, image_id=image_id)

    mask_path = OUTPUT_DIR / f"{image_id}_unet_watershed_inst.tif"
    metadata_path = OUTPUT_DIR / f"{image_id}_unet_watershed_metadata.json"

    tifffile.imwrite(mask_path, prediction.instance_mask)
    save_json(metadata_path, prediction.metadata)

    print("UNet + Watershed smoke test finished.")
    print(f"Image ID: {image_id}")
    print(f"Mask path: {mask_path}")
    print(f"Metadata path: {metadata_path}")
    print(f"Mask shape: {prediction.instance_mask.shape}")
    print(f"Max label: {prediction.metadata.get('max_label')}")
    print(f"Threshold: {prediction.metadata.get('threshold')}")
    print(f"Min size: {prediction.metadata.get('min_size')}")
    print(f"Min distance: {prediction.metadata.get('min_distance')}")


if __name__ == "__main__":
    main()