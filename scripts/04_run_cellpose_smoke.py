from __future__ import annotations

import argparse
import json
from pathlib import Path

import tifffile as tiff

from nuclei_benchmark.data.io import load_tiff_image
from nuclei_benchmark.models.cellpose_model import CellposeModelWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-image Cellpose smoke test through the benchmark wrapper."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path(r"data\raw\monuseg_train\Tissue_Images\TCGA-18-5592-01Z-00-DX1.tif"),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/cellpose.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions/cellpose_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = load_tiff_image(args.image_path)
    image_id = args.image_path.stem

    model = CellposeModelWrapper(args.config_path)
    prediction = model.predict(image=image, image_id=image_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = args.output_dir / f"{image_id}_cellpose_inst.tif"
    metadata_path = args.output_dir / f"{image_id}_cellpose_metadata.json"

    tiff.imwrite(str(mask_path), prediction.instance_mask)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(prediction.metadata, handle, indent=2)

    print("Cellpose smoke test finished")
    print(f"Image ID: {prediction.image_id}")
    print(f"Mask path: {mask_path}")
    print(f"Metadata path: {metadata_path}")
    print(f"Mask shape: {prediction.instance_mask.shape}")
    print(f"Max label: {int(prediction.instance_mask.max())}")


if __name__ == "__main__":
    main()