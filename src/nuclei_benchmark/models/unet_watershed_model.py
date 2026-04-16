from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from nuclei_benchmark.models.base import BaseSegmentationModel, ModelPrediction
from nuclei_benchmark.models.unet_network import UNet
from nuclei_benchmark.postprocessing.watershed import foreground_to_instances
from nuclei_benchmark.utils.config import load_yaml_config


class UNetWatershedModelWrapper(BaseSegmentationModel):
    """Config-driven wrapper for checkpoint-based U-Net + Watershed inference."""

    def __init__(self, config_path: Path) -> None:
        super().__init__(model_name="unet_watershed", config_path=config_path)

        self.config = load_yaml_config(config_path)
        self._validate_config()

        self.runtime_config: dict[str, Any] = self.config["runtime"]
        self.model_config: dict[str, Any] = self.config["model"]
        self.inference_config: dict[str, Any] = self.config["inference"]
        self.output_config: dict[str, Any] = self.config["output"]

        self.device = self._resolve_device()
        self.model = self._load_model()

    def _validate_config(self) -> None:
        expected_model_name = self.config.get("model_name")
        if expected_model_name != "unet_watershed":
            raise ValueError(
                f"Expected model_name='unet_watershed' in config, got: {expected_model_name}"
            )

        required_sections = ("runtime", "model", "inference", "output")
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

    def get_device_preference(self) -> str:
        device = self.runtime_config.get("device", "cpu")
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError(f"Unsupported device setting in config: {device}")
        return device

    def _resolve_device(self) -> torch.device:
        device_preference = self.get_device_preference()

        if device_preference == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device_preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Config requests CUDA, but torch.cuda.is_available() is False."
                )
            return torch.device("cuda")

        return torch.device("cpu")

    def _load_model(self) -> UNet:
        checkpoint_path = Path(self.model_config["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        model = UNet(
            in_channels=int(self.model_config.get("in_channels", 3)),
            out_channels=int(self.model_config.get("out_channels", 1)),
            base_channels=int(self.model_config.get("base_channels", 32)),
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model

    @staticmethod
    def _normalize_image(image: np.ndarray) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, C), got: {image.shape}"
            )

        image_float = image.astype(np.float32) / 255.0
        image_chw = np.transpose(image_float, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(image_chw))

    @staticmethod
    def _make_tile_starts(length: int, tile_size: int) -> list[int]:
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
    def _predict_full_image_logits(
        self,
        image_tensor: torch.Tensor,
        tile_size: int,
    ) -> torch.Tensor:
        """
        Run tiled inference on one full image tensor.

        image_tensor shape: (C, H, W)
        returns logits shape: (1, H, W)
        """
        if image_tensor.ndim != 3:
            raise ValueError(
                f"Expected image tensor shape (C, H, W), got {image_tensor.shape}"
            )

        _, height, width = image_tensor.shape

        y_starts = self._make_tile_starts(height, tile_size)
        x_starts = self._make_tile_starts(width, tile_size)

        logits_sum = torch.zeros(
            (1, height, width), dtype=torch.float32, device=self.device
        )
        count_sum = torch.zeros(
            (1, height, width), dtype=torch.float32, device=self.device
        )

        for top in y_starts:
            for left in x_starts:
                image_tile = image_tensor[
                    :, top : top + tile_size, left : left + tile_size
                ]
                image_tile = image_tile.unsqueeze(0).to(self.device)

                tile_logits = self.model(image_tile).squeeze(0)

                tile_height = image_tile.shape[2]
                tile_width = image_tile.shape[3]

                logits_sum[
                    :, top : top + tile_height, left : left + tile_width
                ] += tile_logits[:, :tile_height, :tile_width]
                count_sum[
                    :, top : top + tile_height, left : left + tile_width
                ] += 1.0

        return logits_sum / count_sum.clamp_min(1.0)

    @torch.no_grad()
    def predict_foreground_map(self, image: np.ndarray) -> np.ndarray:
        """Run checkpoint-based U-Net inference and return a foreground probability map."""
        image_tensor = self._normalize_image(image=image)

        tile_size = int(self.inference_config.get("tile_size", 256))

        logits = self._predict_full_image_logits(
            image_tensor=image_tensor,
            tile_size=tile_size,
        )

        foreground_map = (
            torch.sigmoid(logits)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return foreground_map

    def predict_from_foreground_map(
        self,
        foreground_map: np.ndarray,
        image_id: str,
        threshold: float,
        min_size: int,
        min_distance: int,
    ) -> ModelPrediction:
        """Build instance prediction from a precomputed foreground probability map."""
        instance_mask = foreground_to_instances(
            foreground_map=foreground_map,
            threshold=threshold,
            min_size=min_size,
            min_distance=min_distance,
        )

        metadata = {
            "model_name": self.model_name,
            "device_requested": self.get_device_preference(),
            "device_resolved": str(self.device),
            "checkpoint_path": str(self.model_config["checkpoint_path"]),
            "tile_size": int(self.inference_config.get("tile_size", 256)),
            "threshold": threshold,
            "min_size": min_size,
            "min_distance": min_distance,
            "max_label": int(instance_mask.max()),
        }

        return ModelPrediction(
            image_id=image_id,
            instance_mask=np.asarray(instance_mask, dtype=np.uint16),
            metadata=metadata,
        )

    def predict(self, image: np.ndarray, image_id: str) -> ModelPrediction:
        tile_size = int(self.inference_config.get("tile_size", 256))
        threshold = float(self.inference_config.get("threshold", 0.5))
        min_size = int(self.inference_config.get("min_size", 16))
        min_distance = int(self.inference_config.get("min_distance", 9))

        foreground_map = self.predict_foreground_map(image=image)

        return self.predict_from_foreground_map(
            foreground_map=foreground_map,
            image_id=image_id,
            threshold=threshold,
            min_size=min_size,
            min_distance=min_distance,
        )