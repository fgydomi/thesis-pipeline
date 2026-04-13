from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageMaskPair:
    """Container for one RGB image and its GT instance mask path."""

    image_id: str
    image_path: Path
    gt_instance_path: Path


def normalize_manifest_path(path_str: str) -> Path:
    """Normalize Windows-style manifest paths for the current platform."""
    return Path(path_str.replace("\\", "/"))


def load_pairs_from_manifest(
    manifest_path: Path,
    split: str = "train",
) -> list[ImageMaskPair]:
    """Load image/GT pairs from the dataset manifest."""
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_columns = [
        "image_id",
        "split",
        "image_path",
        "gt_instance_path",
        "image_exists",
        "gt_exists",
    ]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Manifest must contain a '{column}' column.")

    split_df = df[df["split"] == split].copy()
    split_df = split_df[(split_df["image_exists"] == True) & (split_df["gt_exists"] == True)]

    if split_df.empty:
        raise ValueError(f"No usable rows found for split='{split}'.")

    split_df = split_df.sort_values("image_id")

    pairs: list[ImageMaskPair] = []
    for row in split_df.itertuples(index=False):
        pairs.append(
            ImageMaskPair(
                image_id=str(row.image_id),
                image_path=normalize_manifest_path(str(row.image_path)),
                gt_instance_path=normalize_manifest_path(str(row.gt_instance_path)),
            )
        )

    return pairs


def split_train_val_pairs(
    pairs: list[ImageMaskPair],
    val_count: int = 8,
    seed: int = 42,
) -> tuple[list[ImageMaskPair], list[ImageMaskPair]]:
    """Create an image-level train/validation split."""
    if not pairs:
        raise ValueError("pairs must not be empty")

    if val_count <= 0 or val_count >= len(pairs):
        raise ValueError(
            f"val_count must be between 1 and len(pairs)-1. "
            f"Got val_count={val_count}, len(pairs)={len(pairs)}."
        )

    image_ids = [pair.image_id for pair in pairs]

    rng = random.Random(seed)
    shuffled_ids = image_ids[:]
    rng.shuffle(shuffled_ids)

    val_ids = set(shuffled_ids[:val_count])

    train_pairs = [pair for pair in pairs if pair.image_id not in val_ids]
    val_pairs = [pair for pair in pairs if pair.image_id in val_ids]

    return train_pairs, val_pairs


def instance_mask_to_binary_target(instance_mask: np.ndarray) -> np.ndarray:
    """Convert an instance mask to a binary foreground mask."""
    if instance_mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D instance mask, got shape: {instance_mask.shape}"
        )

    return np.asarray(instance_mask > 0, dtype=np.float32)


def load_image_and_target(pair: ImageMaskPair) -> dict[str, Any]:
    """Load one RGB image and its binary foreground target."""
    image = tifffile.imread(pair.image_path)
    gt_instance_mask = tifffile.imread(pair.gt_instance_path)

    if image.ndim != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, C), got: {image.shape}"
        )

    if gt_instance_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D GT instance mask, got: {gt_instance_mask.shape}"
        )

    if image.shape[:2] != gt_instance_mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch for {pair.image_id}: "
            f"image={image.shape[:2]}, mask={gt_instance_mask.shape}"
        )

    binary_target = instance_mask_to_binary_target(gt_instance_mask)

    return {
        "image_id": pair.image_id,
        "image": image,
        "target": binary_target,
    }


class RandomPatchUNetDataset(Dataset):
    """Random patch dataset for U-Net training."""

    def __init__(
        self,
        pairs: list[ImageMaskPair],
        patch_size: int = 256,
        samples_per_epoch: int = 512,
        augment: bool = True,
        preload: bool = True,
    ) -> None:
        if not pairs:
            raise ValueError("pairs must not be empty")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if samples_per_epoch <= 0:
            raise ValueError(
                f"samples_per_epoch must be positive, got {samples_per_epoch}"
            )

        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.preload = preload

        if preload:
            self.records: list[dict[str, Any] | ImageMaskPair] = [
                load_image_and_target(pair) for pair in pairs
            ]
        else:
            self.records = list(pairs)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_or_pair = self.records[index % len(self.records)]
        record = (
            load_image_and_target(record_or_pair)
            if isinstance(record_or_pair, ImageMaskPair)
            else record_or_pair
        )

        image = record["image"]
        target = record["target"]

        image_patch, target_patch = self._sample_random_patch(image, target)

        if self.augment:
            image_patch, target_patch = self._apply_augmentations(
                image_patch,
                target_patch,
            )

        image_tensor = self._image_to_tensor(image_patch)
        target_tensor = self._target_to_tensor(target_patch)

        return {
            "image_id": record["image_id"],
            "image": image_tensor,
            "target": target_tensor,
        }

    def _sample_random_patch(
        self,
        image: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = target.shape

        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Patch size {self.patch_size} is larger than image shape {target.shape}."
            )

        top = 0 if height == self.patch_size else np.random.randint(
            0, height - self.patch_size + 1
        )
        left = 0 if width == self.patch_size else np.random.randint(
            0, width - self.patch_size + 1
        )

        image_patch = image[top : top + self.patch_size, left : left + self.patch_size]
        target_patch = target[top : top + self.patch_size, left : left + self.patch_size]

        return image_patch, target_patch

    def _apply_augmentations(
        self,
        image_patch: np.ndarray,
        target_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image_patch = np.flip(image_patch, axis=1)
            target_patch = np.flip(target_patch, axis=1)

        if np.random.rand() < 0.5:
            image_patch = np.flip(image_patch, axis=0)
            target_patch = np.flip(target_patch, axis=0)

        rotation_k = np.random.randint(0, 4)
        if rotation_k > 0:
            image_patch = np.rot90(image_patch, k=rotation_k, axes=(0, 1))
            target_patch = np.rot90(target_patch, k=rotation_k, axes=(0, 1))

        return np.ascontiguousarray(image_patch), np.ascontiguousarray(target_patch)

    @staticmethod
    def _image_to_tensor(image_patch: np.ndarray) -> torch.Tensor:
        image_float = image_patch.astype(np.float32) / 255.0
        image_chw = np.transpose(image_float, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(image_chw))

    @staticmethod
    def _target_to_tensor(target_patch: np.ndarray) -> torch.Tensor:
        target_float = target_patch.astype(np.float32)[None, ...]
        return torch.from_numpy(np.ascontiguousarray(target_float))


class FullImageUNetDataset(Dataset):
    """Full-image dataset for validation and later tiled inference."""

    def __init__(
        self,
        pairs: list[ImageMaskPair],
        preload: bool = True,
    ) -> None:
        if not pairs:
            raise ValueError("pairs must not be empty")

        self.preload = preload
        if preload:
            self.records: list[dict[str, Any] | ImageMaskPair] = [
                load_image_and_target(pair) for pair in pairs
            ]
        else:
            self.records = list(pairs)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_or_pair = self.records[index]
        record = (
            load_image_and_target(record_or_pair)
            if isinstance(record_or_pair, ImageMaskPair)
            else record_or_pair
        )

        image = record["image"].astype(np.float32) / 255.0
        image_chw = np.transpose(image, (2, 0, 1))
        target = record["target"].astype(np.float32)[None, ...]

        return {
            "image_id": record["image_id"],
            "image": torch.from_numpy(np.ascontiguousarray(image_chw)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
        }