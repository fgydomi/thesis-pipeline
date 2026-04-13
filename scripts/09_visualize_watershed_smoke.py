from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.color import label2rgb

from nuclei_benchmark.postprocessing.watershed import foreground_to_instances


IMAGE_PATH = Path("data/raw/monuseg_train/Tissue_Images/TCGA-18-5592-01Z-00-DX1.tif")
GT_MASK_PATH = Path("data/processed/gt_instances/train/TCGA-18-5592-01Z-00-DX1_inst.tif")
OUTPUT_DIR = Path("outputs/figures/watershed_smoke")


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

    image = tifffile.imread(IMAGE_PATH)
    gt_mask = tifffile.imread(GT_MASK_PATH)

    foreground_map = np.asarray(gt_mask > 0, dtype=np.float32)

    watershed_mask = foreground_to_instances(
        foreground_map=foreground_map,
        threshold=0.5,
        min_size=16,
        min_distance=9,
    )

    gt_binary = gt_mask > 0
    watershed_binary = watershed_mask > 0

    gt_colored = label2rgb(gt_mask, bg_label=0, bg_color=(0, 0, 0))
    watershed_colored = label2rgb(watershed_mask, bg_label=0, bg_color=(0, 0, 0))

    gt_count = count_instances(gt_mask)
    watershed_count = count_instances(watershed_mask)

    figure = plt.figure(figsize=(16, 10))

    ax1 = figure.add_subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original image")
    ax1.axis("off")

    ax2 = figure.add_subplot(2, 3, 2)
    ax2.imshow(gt_binary, cmap="gray")
    ax2.set_title("GT foreground")
    ax2.axis("off")

    ax3 = figure.add_subplot(2, 3, 3)
    ax3.imshow(watershed_binary, cmap="gray")
    ax3.set_title("Watershed foreground")
    ax3.axis("off")

    ax4 = figure.add_subplot(2, 3, 4)
    ax4.imshow(gt_colored)
    ax4.set_title(f"GT instances (count={gt_count})")
    ax4.axis("off")

    ax5 = figure.add_subplot(2, 3, 5)
    ax5.imshow(watershed_colored)
    ax5.set_title(f"Watershed instances (count={watershed_count})")
    ax5.axis("off")

    ax6 = figure.add_subplot(2, 3, 6)
    ax6.imshow(image)
    ax6.imshow(watershed_colored, alpha=0.45)
    ax6.set_title("Watershed overlay")
    ax6.axis("off")

    figure.suptitle(
        f"Watershed smoke visualization: {image_id}\n"
        f"GT instances={gt_count}, Watershed instances={watershed_count}",
        fontsize=14,
    )

    figure.tight_layout()

    output_path = OUTPUT_DIR / f"{image_id}_watershed_visualization.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print("Watershed visualization finished.")
    print(f"Image ID: {image_id}")
    print(f"GT instance count: {gt_count}")
    print(f"Watershed instance count: {watershed_count}")
    print(f"Output figure path: {output_path}")


if __name__ == "__main__":
    main()