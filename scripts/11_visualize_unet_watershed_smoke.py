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
OUTPUT_DIR = Path("outputs/figures/unet_watershed_smoke")


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

    wrapper = UNetWatershedModelWrapper(config_path=CONFIG_PATH)
    prediction = wrapper.predict(image=image, image_id=image_id)
    pred_mask = prediction.instance_mask

    gt_binary = gt_mask > 0
    pred_binary = pred_mask > 0

    gt_colored = label2rgb(gt_mask, bg_label=0, bg_color=(0, 0, 0))
    pred_colored = label2rgb(pred_mask, bg_label=0, bg_color=(0, 0, 0))

    gt_count = count_instances(gt_mask)
    pred_count = count_instances(pred_mask)

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
    ax3.imshow(pred_binary, cmap="gray")
    ax3.set_title("U-Net + Watershed foreground")
    ax3.axis("off")

    ax4 = figure.add_subplot(2, 3, 4)
    ax4.imshow(gt_colored)
    ax4.set_title(f"GT instances (count={gt_count})")
    ax4.axis("off")

    ax5 = figure.add_subplot(2, 3, 5)
    ax5.imshow(pred_colored)
    ax5.set_title(f"U-Net + Watershed instances (count={pred_count})")
    ax5.axis("off")

    ax6 = figure.add_subplot(2, 3, 6)
    ax6.imshow(image)
    ax6.imshow(pred_colored, alpha=0.45)
    ax6.set_title("U-Net + Watershed overlay")
    ax6.axis("off")

    figure.suptitle(
        f"U-Net + Watershed visualization: {image_id}\n"
        f"GT instances={gt_count}, Predicted instances={pred_count}",
        fontsize=14,
    )

    figure.tight_layout()

    output_path = OUTPUT_DIR / f"{image_id}_unet_watershed_visualization.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print("U-Net + Watershed visualization finished.")
    print(f"Image ID: {image_id}")
    print(f"GT instance count: {gt_count}")
    print(f"Predicted instance count: {pred_count}")
    print(f"Output figure path: {output_path}")


if __name__ == "__main__":
    main()