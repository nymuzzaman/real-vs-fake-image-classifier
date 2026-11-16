"""Classify one or more images as real or fake.

    python -m deepfake_detection.predict --checkpoint checkpoints/resnet50_best.pth img1.png img2.png
"""

import argparse

import torch
from PIL import Image

from deepfake_detection.dataset import CLASSES
from deepfake_detection.models import get_device, get_transform, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict real/fake for images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("images", nargs="+", help="Image file(s) to classify.")
    args = parser.parse_args()

    device = get_device()
    model, model_name = load_checkpoint(args.checkpoint, device)
    transform = get_transform(model_name)

    for path in args.images:
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(image), dim=1)[0]
        label = int(probs.argmax())
        print(f"{path}: {CLASSES[label].capitalize()} ({probs[label] * 100:.2f}% confidence)")


if __name__ == "__main__":
    main()
