"""Fine-tune a pretrained CNN to classify real vs. fake images.

    python -m deepfake_detection.train --model resnet50
"""

import argparse
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from deepfake_detection.dataset import DeepfakeDataset  # noqa: E402
from deepfake_detection.models import (  # noqa: E402
    MODEL_CONFIGS,
    build_model,
    forward_logits,
    get_device,
    get_transform,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc="train" if training else "val", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = forward_logits(model, images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def save_plot(history: dict, path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set(title="Loss", xlabel="epoch")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set(title="Accuracy (%)", xlabel="epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deepfake detector.")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    parser.add_argument("--data_dir", default="data", help="Folder with train/ and val/.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None, help="Default: per-model value.")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    lr = args.lr or MODEL_CONFIGS[args.model].learning_rate
    print(f"Model: {args.model} | device: {device} | lr: {lr}")

    transform = get_transform(args.model)
    loaders = {
        split: DataLoader(
            DeepfakeDataset(os.path.join(args.data_dir, split), transform),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
        )
        for split in ("train", "val")
    }

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{args.model}_best.pth")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:5.1f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:5.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": args.model,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(f"  -> saved best model ({val_acc:.1f}%) to {ckpt_path}")

    save_plot(history, os.path.join(args.results_dir, f"{args.model}_training.png"))
    with open(os.path.join(args.results_dir, f"{args.model}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val accuracy: {best_val_acc:.1f}%")


if __name__ == "__main__":
    main()
