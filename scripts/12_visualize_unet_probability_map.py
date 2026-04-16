from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from nuclei_benchmark.data.io import load_tiff_image
from nuclei_benchmark.models.unet_watershed_model import UNetWatershedModelWrapper


CONFIG_PATH = Path("configs/unet_watershed.yaml")
IMAGE_PATH = Path("data/raw/monuseg_train/Tissue_Images/TCGA-18-5592-01Z-00-DX1.tif")
GT_MASK_PATH = Path("data/processed/gt_instances/train/TCGA-18-5592-01Z-00-DX1_inst.tif")
OUTPUT_DIR = Path("outputs/figures/unet_probability_map")

THRESHOLDS = [0.35, 0.40, 0.45, 0.50]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")
    if not GT_MASK_PATH.exists():
        raise FileNotFoundError(f"GT mask file not found: {GT_MASK_PATH}")

    image_id = IMAGE_PATH.stem

    image = load_tiff_image(IMAGE_PATH)
    gt_mask = tifffile.imread(GT_MASK_PATH)
    gt_binary = gt_mask > 0

    wrapper = UNetWatershedModelWrapper(config_path=CONFIG_PATH)
    foreground_map = wrapper.predict_foreground_map(image=image)

    figure = plt.figure(figsize=(18, 10))

    ax1 = figure.add_subplot(2, 4, 1)
    ax1.imshow(image)
    ax1.set_title("Original image")
    ax1.axis("off")

    ax2 = figure.add_subplot(2, 4, 2)
    ax2.imshow(gt_binary, cmap="gray")
    ax2.set_title(f"GT foreground\npixels={int(gt_binary.sum())}")
    ax2.axis("off")

    ax3 = figure.add_subplot(2, 4, 3)
    heatmap = ax3.imshow(foreground_map, cmap="viridis", vmin=0.0, vmax=1.0)
    ax3.set_title("U-Net foreground probability")
    ax3.axis("off")
    figure.colorbar(heatmap, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = figure.add_subplot(2, 4, 4)
    ax4.hist(foreground_map.ravel(), bins=50, range=(0.0, 1.0))
    ax4.set_title("Probability histogram")
    ax4.set_xlabel("Probability")
    ax4.set_ylabel("Pixel count")

    threshold_pixel_counts: dict[float, int] = {}

    for idx, threshold in enumerate(THRESHOLDS, start=5):
        binary_pred = foreground_map >= threshold
        threshold_pixel_counts[threshold] = int(binary_pred.sum())

        ax = figure.add_subplot(2, 4, idx)
        ax.imshow(binary_pred, cmap="gray")
        ax.set_title(
            f"Threshold {threshold:.2f}\n"
            f"pixels={threshold_pixel_counts[threshold]}"
        )
        ax.axis("off")

    figure.suptitle(
        f"U-Net probability map and threshold sweep: {image_id}",
        fontsize=14,
    )
    figure.tight_layout()

    output_path = OUTPUT_DIR / f"{image_id}_unet_probability_map.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print("U-Net probability map visualization finished.")
    print(f"Image ID: {image_id}")
    print(f"GT foreground pixels: {int(gt_binary.sum())}")
    print(f"Foreground map min: {float(foreground_map.min()):.6f}")
    print(f"Foreground map max: {float(foreground_map.max()):.6f}")
    print(f"Foreground map mean: {float(foreground_map.mean()):.6f}")
    for threshold in THRESHOLDS:
        print(
            f"Threshold {threshold:.2f} foreground pixels: "
            f"{threshold_pixel_counts[threshold]}"
        )
    print(f"Output figure path: {output_path}")


if __name__ == "__main__":
    main()