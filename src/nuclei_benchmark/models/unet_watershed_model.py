from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nuclei_benchmark.models.base import BaseSegmentationModel, ModelPrediction
from nuclei_benchmark.postprocessing.watershed import foreground_to_instances
from nuclei_benchmark.utils.config import load_yaml_config


class UNetWatershedModelWrapper(BaseSegmentationModel):
    """Config-driven wrapper for a U-Net + Watershed style baseline."""

    def __init__(self, config_path: Path) -> None:
        super().__init__(model_name="unet_watershed", config_path=config_path)

        self.config = load_yaml_config(config_path)
        self._validate_config()

        self.runtime_config: dict[str, Any] = self.config["runtime"]
        self.model_config: dict[str, Any] = self.config["model"]
        self.inference_config: dict[str, Any] = self.config["inference"]
        self.output_config: dict[str, Any] = self.config["output"]

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
        if device not in {"cpu", "auto"}:
            raise ValueError(f"Unsupported device setting in config: {device}")
        return device

    def _build_foreground_map(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            grayscale = image.mean(axis=2)
        elif image.ndim == 2:
            grayscale = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        grayscale = grayscale.astype(np.float32)

        if grayscale.max() > 0:
            grayscale = grayscale / grayscale.max()

        foreground_map = 1.0 - grayscale
        return np.asarray(foreground_map, dtype=np.float32)

    def predict(self, image: np.ndarray, image_id: str) -> ModelPrediction:
        foreground_map = self._build_foreground_map(image)

        threshold = float(self.inference_config.get("threshold", 0.5))
        min_size = int(self.inference_config.get("min_size", 16))
        min_distance = int(self.inference_config.get("min_distance", 9))

        instance_mask = foreground_to_instances(
            foreground_map=foreground_map,
            threshold=threshold,
            min_size=min_size,
            min_distance=min_distance,
        )

        metadata = {
            "model_name": self.model_name,
            "device_requested": self.get_device_preference(),
            "device_resolved": "cpu",
            "foreground_source": self.model_config.get("foreground_source", "binary_mask"),
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