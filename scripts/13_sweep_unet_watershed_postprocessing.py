from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.color import label2rgb

from nuclei_benchmark.data.io import load_tiff_image
from nuclei_benchmark.models.unet_watershed_model import UNetWatershedModelWrapper


CONFIG_PATH = Path("configs/unet_watershed.yaml")
IMAGE_PATH = Path("data/raw/monuseg_train/Tissue_Images/TCGA-18-5592-01Z-00-DX1.tif")
GT_MASK_PATH = Path("data/processed/gt_instances/train/TCGA-18-5592-01Z-00-DX1_inst.tif")
OUTPUT_DIR = Path("outputs/figures/unet_watershed_sweep")

THRESHOLDS = [0.40, 0.45, 0.50]
MIN_DISTANCES = [9, 11]
MIN_SIZE = 16


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_instances(label_mask: np.ndarray) -> int:
    unique_values = np.unique(label_mask)
    return int(np.sum(unique_values > 0))


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")
    if not GT_MASK_PATH.exists():
        raise FileNotFoundError(f"GT mask file not found: {GT_MASK_PATH}")

    image_id = IMAGE_PATH.stem

    image = load_tiff_image(IMAGE_PATH)
    gt_mask = tifffile.imread(GT_MASK_PATH)
    gt_count = count_instances(gt_mask)

    wrapper = UNetWatershedModelWrapper(config_path=CONFIG_PATH)
    foreground_map = wrapper.predict_foreground_map(image=image)

    combinations = [(thr, md) for thr in THRESHOLDS for md in MIN_DISTANCES]

    figure = plt.figure(figsize=(20, 10))

    ax1 = figure.add_subplot(2, 5, 1)
    ax1.imshow(image)
    ax1.set_title("Original image")
    ax1.axis("off")

    ax2 = figure.add_subplot(2, 5, 2)
    ax2.imshow(label2rgb(gt_mask, bg_label=0, bg_color=(0, 0, 0)))
    ax2.set_title(f"GT instances\ncount={gt_count}")
    ax2.axis("off")

    ax3 = figure.add_subplot(2, 5, 3)
    heatmap = ax3.imshow(foreground_map, cmap="viridis", vmin=0.0, vmax=1.0)
    ax3.set_title("Foreground probability")
    ax3.axis("off")
    figure.colorbar(heatmap, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = figure.add_subplot(2, 5, 4)
    ax4.hist(foreground_map.ravel(), bins=50, range=(0.0, 1.0))
    ax4.set_title("Probability histogram")
    ax4.set_xlabel("Probability")
    ax4.set_ylabel("Pixel count")

    results = []

    for idx, (threshold, min_distance) in enumerate(combinations, start=5):
        prediction = wrapper.predict_from_foreground_map(
            foreground_map=foreground_map,
            image_id=image_id,
            threshold=threshold,
            min_size=MIN_SIZE,
            min_distance=min_distance,
        )

        pred_mask = prediction.instance_mask
        pred_count = count_instances(pred_mask)

        results.append(
            {
                "threshold": threshold,
                "min_distance": min_distance,
                "predicted_count": pred_count,
            }
        )

        ax = figure.add_subplot(2, 5, idx)
        ax.imshow(label2rgb(pred_mask, bg_label=0, bg_color=(0, 0, 0)))
        ax.set_title(
            f"thr={threshold:.2f}, md={min_distance}\ncount={pred_count}"
        )
        ax.axis("off")

    figure.suptitle(
        f"U-Net + Watershed post-processing sweep: {image_id}\nGT count={gt_count}",
        fontsize=14,
    )
    figure.tight_layout()

    output_path = OUTPUT_DIR / f"{image_id}_unet_watershed_sweep.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print("U-Net + Watershed sweep finished.")
    print(f"Image ID: {image_id}")
    print(f"GT instance count: {gt_count}")
    for result in results:
        print(
            f"threshold={result['threshold']:.2f}, "
            f"min_distance={result['min_distance']}, "
            f"predicted_count={result['predicted_count']}"
        )
    print(f"Output figure path: {output_path}")


if __name__ == "__main__":
    main()