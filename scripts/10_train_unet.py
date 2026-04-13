from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nuclei_benchmark.data.unet_dataset import (
    FullImageUNetDataset,
    RandomPatchUNetDataset,
    load_pairs_from_manifest,
    split_train_val_pairs,
)
from nuclei_benchmark.models.unet_network import UNet
from nuclei_benchmark.training.losses import bce_dice_loss, binary_dice_from_logits
from nuclei_benchmark.utils.config import load_yaml_config


DEFAULT_CONFIG_PATH = Path("configs/unet_train.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline U-Net model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the U-Net training config file.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_history_csv(path: Path, history: list[dict[str, str]]) -> None:
    fieldnames = ["epoch", "train_loss", "val_dice"]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requests CUDA, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    raise ValueError(f"Unsupported device setting: {device_name}")


def make_tile_starts(length: int, tile_size: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, tile_size))
    last_start = length - tile_size

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


@torch.no_grad()
def predict_full_image_logits(
    model: UNet,
    image_tensor: torch.Tensor,
    tile_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run tiled inference on one full image tensor.

    image_tensor shape: (C, H, W)
    returns logits shape: (1, H, W)
    """
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image tensor shape (C, H, W), got {image_tensor.shape}")

    _, height, width = image_tensor.shape

    y_starts = make_tile_starts(height, tile_size)
    x_starts = make_tile_starts(width, tile_size)

    logits_sum = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    count_sum = torch.zeros((1, height, width), dtype=torch.float32, device=device)

    for top in y_starts:
        for left in x_starts:
            image_tile = image_tensor[:, top : top + tile_size, left : left + tile_size]
            image_tile = image_tile.unsqueeze(0).to(device)

            tile_logits = model(image_tile).squeeze(0)

            tile_height = image_tile.shape[2]
            tile_width = image_tile.shape[3]

            logits_sum[:, top : top + tile_height, left : left + tile_width] += tile_logits[
                :, :tile_height, :tile_width
            ]
            count_sum[:, top : top + tile_height, left : left + tile_width] += 1.0

    return logits_sum / count_sum.clamp_min(1.0)


def train_one_epoch(
    model: UNet,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    running_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = bce_dice_loss(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def validate_one_epoch(
    model: UNet,
    val_loader: DataLoader,
    tile_size: int,
    threshold: float,
    device: torch.device,
) -> float:
    model.eval()

    dice_values: list[float] = []

    for batch in val_loader:
        image = batch["image"].squeeze(0)
        target = batch["target"].to(device)

        logits = predict_full_image_logits(
            model=model,
            image_tensor=image,
            tile_size=tile_size,
            device=device,
        ).unsqueeze(0)

        dice_value = binary_dice_from_logits(
            logits=logits,
            targets=target,
            threshold=threshold,
        )
        dice_values.append(dice_value)

    if not dice_values:
        raise ValueError("Validation loader produced no samples.")

    return float(sum(dice_values) / len(dice_values))


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    device = resolve_device(config["training"].get("device", "auto"))

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    ensure_dir(checkpoint_dir)

    history_csv_path = checkpoint_dir / config["output"]["history_csv_name"]
    split_json_path = checkpoint_dir / config["output"]["split_json_name"]
    best_checkpoint_path = checkpoint_dir / config["output"]["best_checkpoint_name"]

    manifest_path = Path(config["data"]["manifest_path"])
    train_split = config["data"].get("train_split", "train")

    pairs = load_pairs_from_manifest(manifest_path=manifest_path, split=train_split)
    train_pairs, val_pairs = split_train_val_pairs(
        pairs=pairs,
        val_count=int(config["data"]["val_count"]),
        seed=int(config["data"]["split_seed"]),
    )

    train_dataset = RandomPatchUNetDataset(
        pairs=train_pairs,
        patch_size=int(config["data"]["patch_size"]),
        samples_per_epoch=int(config["data"]["samples_per_epoch"]),
        augment=True,
        preload=bool(config["data"].get("preload", True)),
    )

    val_dataset = FullImageUNetDataset(
        pairs=val_pairs,
        preload=bool(config["data"].get("preload", True)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(
        in_channels=int(config["model"]["in_channels"]),
        out_channels=int(config["model"]["out_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )

    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    tile_size = int(config["validation"]["tile_size"])
    threshold = float(config["validation"]["threshold"])

    save_json(
        split_json_path,
        {
            "train_image_ids": [pair.image_id for pair in train_pairs],
            "val_image_ids": [pair.image_id for pair in val_pairs],
        },
    )

    print("Starting U-Net training")
    print(f"Device: {device}")
    print(f"Train images: {len(train_pairs)}")
    print(f"Validation images: {len(val_pairs)}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print()

    best_val_dice = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, str]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_dice = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            tile_size=tile_size,
            threshold=threshold,
            device=device,
        )

        history.append(
            {
                "epoch": str(epoch),
                "train_loss": f"{train_loss:.6f}",
                "val_dice": f"{val_dice:.6f}",
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_dice={val_dice:.6f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "config_path": str(args.config),
                },
                best_checkpoint_path,
            )
            print(f"  New best checkpoint saved: {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement. Patience: {epochs_without_improvement}/{patience}")

        save_history_csv(history_csv_path, history)

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        print()

    save_history_csv(history_csv_path, history)

    print("Training finished.")
    print(f"Best validation Dice: {best_val_dice:.6f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Training history: {history_csv_path}")
    print(f"Split file: {split_json_path}")


if __name__ == "__main__":
    main()