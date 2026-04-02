from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelPrediction:
    """Standardized prediction output for one image."""

    image_id: str
    instance_mask: np.ndarray
    metadata: dict[str, Any]


class BaseSegmentationModel(ABC):
    """Abstract base class for all benchmarked segmentation models."""

    def __init__(self, model_name: str, config_path: Path) -> None:
        self.model_name = model_name
        self.config_path = config_path

    @abstractmethod
    def predict(self, image: np.ndarray, image_id: str) -> ModelPrediction:
        """
        Run inference on a single image and return a standardized prediction object.
        """
        raise NotImplementedError