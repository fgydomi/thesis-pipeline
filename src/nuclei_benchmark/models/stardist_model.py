from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from csbdeep.utils import normalize

from nuclei_benchmark.models.base import BaseSegmentationModel, ModelPrediction
from nuclei_benchmark.utils.config import load_yaml_config


class StarDistModelWrapper(BaseSegmentationModel):
    """Config-driven wrapper for StarDist inference."""

    def __init__(self, config_path: Path) -> None:
        super().__init__(model_name="stardist", config_path=config_path)

        self.config = load_yaml_config(config_path)
        self._validate_config()

        self.runtime_config: dict[str, Any] = self.config["runtime"]
        self.model_config: dict[str, Any] = self.config["model"]
        self.inference_config: dict[str, Any] = self.config["inference"]
        self.output_config: dict[str, Any] = self.config["output"]

    def _validate_config(self) -> None:
        expected_model_name = self.config.get("model_name")
        if expected_model_name != "stardist":
            raise ValueError(
                f"Expected model_name='stardist' in config, got: {expected_model_name}"
            )

        required_sections = ("runtime", "model", "inference", "output")
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

    def get_device_preference(self) -> str:
        device = self.runtime_config.get("device", "cpu")
        if device not in {"cpu", "gpu", "auto"}:
            raise ValueError(f"Unsupported device setting in config: {device}")
        return device

    def _configure_tensorflow(self) -> list:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        return gpus

    def _resolve_runtime(self) -> tuple[str, list]:
        device_preference = self.get_device_preference()
        gpus = self._configure_tensorflow()

        if device_preference == "gpu":
            if not gpus:
                raise RuntimeError(
                    "Config requests device='gpu', but TensorFlow does not see any GPU."
                )
            return "gpu", gpus

        if device_preference == "cpu":
            return "cpu", []

        # auto
        if gpus:
            return "gpu", gpus
        return "cpu", []

    def _create_model(self):
        try:
            from stardist.models import StarDist2D
        except ImportError as exc:
            raise ImportError(
                "StarDist is not installed in the active environment."
            ) from exc

        pretrained_model = self.model_config.get("pretrained_model")
        if not pretrained_model:
            raise ValueError("StarDist config must define a pretrained_model.")

        return StarDist2D.from_pretrained(pretrained_model)

    def predict(self, image: np.ndarray, image_id: str) -> ModelPrediction:
        if image.ndim not in (2, 3):
            raise ValueError(f"Unsupported image shape for StarDist: {image.shape}")

        resolved_device, gpus = self._resolve_runtime()
        model = self._create_model()

        use_normalize = bool(self.inference_config.get("normalize", True))
        image_input = normalize(image) if use_normalize else image

        labels, details = model.predict_instances(image_input)

        instance_mask = np.asarray(labels, dtype=np.uint16)

        metadata = {
            "model_name": self.model_name,
            "device_requested": self.get_device_preference(),
            "device_resolved": resolved_device,
            "num_visible_gpus": len(gpus),
            "max_label": int(instance_mask.max()),
            "num_polygons": len(details["coord"]) if "coord" in details else 0,
        }

        return ModelPrediction(
            image_id=image_id,
            instance_mask=instance_mask,
            metadata=metadata,
        )