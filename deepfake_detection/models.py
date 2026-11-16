"""Model definitions and the matching preprocessing for each backbone.

Keeping the transform next to the model guarantees that training, evaluation
and prediction all use the same input size and normalisation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelConfig:
    input_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    learning_rate: float  # default used in the original experiments


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "resnet50": ModelConfig(224, IMAGENET_MEAN, IMAGENET_STD, 1e-3),
    "efficientnet_b7": ModelConfig(224, IMAGENET_MEAN, IMAGENET_STD, 1e-4),
    "inception_v3": ModelConfig(299, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 1e-3),
}


def get_transform(model_name: str) -> Callable:
    cfg = MODEL_CONFIGS[model_name]
    return transforms.Compose(
        [
            transforms.Resize((cfg.input_size, cfg.input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )


def build_model(model_name: str, pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """Create a backbone with its classification head replaced for ``num_classes``."""
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b7":
        weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b7(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "inception_v3":
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        # aux_logits must be True to load the pretrained weights; the auxiliary
        # output is ignored (see `forward_logits`).
        model = models.inception_v3(weights=weights, aux_logits=True, init_weights=not pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {list(MODEL_CONFIGS)}")
    return model


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run the model and return the main logits (drops Inception's aux output)."""
    outputs = model(images)
    if isinstance(outputs, tuple):  # InceptionOutputs in train mode
        outputs = outputs[0]
    return outputs


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: str, device: torch.device) -> Tuple[nn.Module, str]:
    """Load a checkpoint written by ``train.py``. Returns (model, model_name)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_name = checkpoint["model_name"]
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), model_name
