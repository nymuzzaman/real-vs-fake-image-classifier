"""Deepfake image classification (real vs. fake) with pretrained CNNs."""

from deepfake_detection.dataset import CLASSES, DeepfakeDataset
from deepfake_detection.models import MODEL_CONFIGS, build_model, get_transform

__all__ = ["CLASSES", "DeepfakeDataset", "MODEL_CONFIGS", "build_model", "get_transform"]
