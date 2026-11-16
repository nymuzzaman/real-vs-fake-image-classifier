"""Image-folder dataset: ``<root>/real/*.png`` -> 0, ``<root>/fake/*.png`` -> 1."""

import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["real", "fake"]  # index = label
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for label, class_name in enumerate(CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for name in sorted(os.listdir(class_dir)):
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    self.samples.append((os.path.join(class_dir, name), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
