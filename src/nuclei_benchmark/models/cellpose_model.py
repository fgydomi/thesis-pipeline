from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from nuclei_benchmark.models.base import BaseSegmentationModel, ModelPrediction
from nuclei_benchmark.utils.config import load_yaml_config


class CellposeModelWrapper(BaseSegmentationModel):
    """Config-driven wrapper for Cellpose inference."""

    def __init__(self, config_path: Path) -> None:
        super().__init__(model_name="cellpose", config_path=config_path)

        self.config = load_yaml_config(config_path)
        self._validate_config()

        self.runtime_config: dict[str, Any] = self.config["runtime"]
        self.model_config: dict[str, Any] = self.config["model"]
        self.inference_config: dict[str, Any] = self.config["inference"]
        self.output_config: dict[str, Any] = self.config["output"]

    def _validate_config(self) -> None:
        expected_model_name = self.config.get("model_name")
        if expected_model_name != "cellpose":
            raise ValueError(
                f"Expected model_name='cellpose' in config, got: {expected_model_name}"
            )

        required_sections = ("runtime", "model", "inference", "output")
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

    def get_device_preference(self) -> str:
        device = self.runtime_config.get("device", "cpu")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unsupported device setting in config: {device}")
        return device

    def _resolve_runtime(self) -> tuple[str, bool]:
        device_preference = self.get_device_preference()
        cuda_available = torch.cuda.is_available()

        if device_preference == "cuda":
            if not cuda_available:
                raise RuntimeError(
                    "Config requests device='cuda', but CUDA is not available."
                )
            return "cuda", True

        if device_preference == "cpu":
            return "cpu", False

        # auto
        if cuda_available:
            return "cuda", True
        return "cpu", False

    def _create_model(self):
        try:
            from cellpose.models import CellposeModel
        except ImportError as exc:
            raise ImportError(
                "Cellpose is not installed in the active environment."
            ) from exc

        resolved_device, use_gpu = self._resolve_runtime()
        pretrained_model = self.model_config.get("pretrained_model")

        if pretrained_model in (None, "", "default"):
            model = CellposeModel(gpu=use_gpu)
        else:
            model = CellposeModel(gpu=use_gpu, pretrained_model=pretrained_model)

        return model, resolved_device

    def predict(self, image: np.ndarray, image_id: str) -> ModelPrediction:
        if image.ndim not in (2, 3):
            raise ValueError(f"Unsupported image shape for Cellpose: {image.shape}")

        model, resolved_device = self._create_model()

        masks, flows, styles = model.eval(
            [image],
            channel_axis=self.inference_config.get("channel_axis", -1),
            normalize=self.inference_config.get("normalize", True),
            diameter=self.inference_config.get("diameter", None),
        )

        instance_mask = np.asarray(masks[0], dtype=np.uint16)

        metadata = {
            "model_name": self.model_name,
            "device_requested": self.get_device_preference(),
            "device_resolved": resolved_device,
            "num_flow_entries": len(flows),
            "num_style_entries": len(styles),
            "max_label": int(instance_mask.max()),
        }

        return ModelPrediction(
            image_id=image_id,
            instance_mask=instance_mask,
            metadata=metadata,
        )