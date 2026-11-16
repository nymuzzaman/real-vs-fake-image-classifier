"""Evaluate a trained checkpoint on the test set (confusion matrix + report).

    python -m deepfake_detection.evaluate --checkpoint checkpoints/resnet50_best.pth
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from deepfake_detection.dataset import CLASSES, DeepfakeDataset  # noqa: E402
from deepfake_detection.models import get_device, get_transform, load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a deepfake detector.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    device = get_device()
    model, model_name = load_checkpoint(args.checkpoint, device)
    dataset = DeepfakeDataset(args.data_dir, get_transform(model_name))
    loader = DataLoader(dataset, batch_size=args.batch_size)

    preds, labels = [], []
    with torch.no_grad():
        for images, y in tqdm(loader, desc="Evaluating"):
            preds.extend(model(images.to(device)).argmax(1).cpu().tolist())
            labels.extend(y.tolist())

    names = [c.capitalize() for c in CLASSES]
    report = classification_report(labels, preds, target_names=names, zero_division=0)
    print(f"\nModel: {model_name}\n\n{report}")

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f"{model_name}_test_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.title(f"Confusion matrix — {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out = os.path.join(args.results_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
