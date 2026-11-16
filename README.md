# Deepfake Detection Using Deep Learning

Binary image classification (**real vs. fake**) on face crops from the
[WildDeepfake](https://huggingface.co/datasets/xingjunm/WildDeepfake) dataset.
Three ImageNet-pretrained CNNs are fine-tuned and compared:

| Model | Input size | Notes |
|-------|-----------|-------|
| **ResNet50** | 224×224 | 50-layer residual network with skip connections |
| **EfficientNet-B7** | 224×224 | Largest EfficientNet; compound-scaled depth/width/resolution |
| **Inception-V3** | 299×299 | Inception modules for multi-scale features (auxiliary head ignored) |

---

## Project structure

```
.
├── deepfake_detection/        # Reusable code
│   ├── dataset.py             # DeepfakeDataset: data/<split>/{real,fake}/*.png
│   ├── models.py              # Model builders + matching preprocessing per model
│   ├── train.py               # Fine-tune a model, save best checkpoint & curves
│   ├── evaluate.py            # Test-set report + confusion matrix
│   └── predict.py             # Classify individual images
├── notebooks/                 # Original Google Colab experiments (with outputs)
│   ├── resnet50.ipynb
│   ├── efficientnet_b7.ipynb
│   └── inception_v3.ipynb
├── data/
│   ├── train/{real,fake}/     # 200 + 200 images
│   ├── val/{real,fake}/       #  50 +  50 images
│   └── test/{real,fake}/      #  50 +  50 images
├── checkpoints/               # Saved models (created by train.py, git-ignored)
├── results/                   # Plots and reports (created by train/evaluate)
└── requirements.txt
```

---

## Dataset

- **Source:** [xingjunm/WildDeepfake](https://huggingface.co/datasets/xingjunm/WildDeepfake)
  on Hugging Face. These are real-world deepfake videos collected from the internet.
- **Subset used:** 600 face images, 224×224 PNG, balanced between classes.

| Split | Real | Fake | Total |
|-------|-----:|-----:|------:|
| train | 200 | 200 | 400 |
| val   |  50 |  50 | 100 |
| test  |  50 |  50 | 100 |

Labels come from the folder name: `real` → 0, `fake` → 1.

---

## Installation

```bash
git clone https://github.com/nymuzzaman/real-vs-fake-image-classifier.git
cd real-vs-fake-image-classifier

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is recommended. The code uses CUDA or Apple Silicon (MPS) automatically
when available and falls back to the CPU otherwise.

---

## Usage

All commands are run from the repository root.

### 1. Train

```bash
python -m deepfake_detection.train --model resnet50
python -m deepfake_detection.train --model efficientnet_b7
python -m deepfake_detection.train --model inception_v3
```

This saves the best checkpoint (by validation accuracy) to
`checkpoints/<model>_best.pth`, and saves the loss/accuracy curves to
`results/<model>_training.png`.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *required* | `resnet50`, `efficientnet_b7` or `inception_v3` |
| `--data_dir` | `data` | Folder containing `train/` and `val/` |
| `--epochs` | `10` | Number of epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | per model | 1e-3 (ResNet50, Inception-V3), 1e-4 (EfficientNet-B7) |
| `--seed` | `42` | Random seed |

### 2. Evaluate on the test set

```bash
python -m deepfake_detection.evaluate --checkpoint checkpoints/resnet50_best.pth
```

This prints precision, recall and F1 per class, and saves
`results/<model>_confusion_matrix.png` and `results/<model>_test_report.txt`.

### 3. Predict individual images

```bash
python -m deepfake_detection.predict --checkpoint checkpoints/resnet50_best.pth \
    data/test/real/311.png data/test/fake/64.png
# Output format: <image path>: Real/Fake (<confidence>% confidence)
```

### Notebooks

The `notebooks/` folder contains the original experiments, run on Google Colab
with the dataset stored on Google Drive (`/content/drive/MyDrive/deepfake/...`).
They are kept as a record of the results below. To run them locally, change the
three dataset paths at the top to `data/train`, `data/val` and `data/test`.

---

## Results

Results from the original notebook runs (10 epochs, Adam, ReduceLROnPlateau):

| Model | Train acc. (final) | Best val acc. | Test acc. | Test F1 (macro) |
|-------|-----:|-----:|-----:|-----:|
| ResNet50 | 100% | 93% | 33% | 0.25 |
| EfficientNet-B7 | 100% | 73% | 50% | 0.33 |
| Inception-V3 | 99.8% | 59% | 43% | 0.36 |

### Discussion

All three models reach almost 100% training accuracy, but none of them
generalises to the test set. Test accuracy is at or below the 50% chance
level, and EfficientNet-B7 predicts *Real* for every test image. The main
reasons:

- **Very little data.** 400 training images is far too few to fine-tune
  networks with 25 to 66 million parameters end-to-end, so the models memorise
  the training set.
- **Distribution shift between splits.** WildDeepfake frames come from a small
  number of video clips. The val and test images come from narrow, consecutive
  ID ranges (likely a few clips), so they look quite different from the
  training set.
- **No augmentation or regularisation.** No augmentation, no frozen layers and
  no early stopping were used, and the learning rate of 1e-3 is high for full
  fine-tuning.

### Possible improvements

- Use the full WildDeepfake dataset, split **by video** so no clip appears in
  two splits.
- Add data augmentation (flips, colour jitter, JPEG compression, blur).
- Freeze the backbone at first, then fine-tune with a lower learning rate (around 1e-5).
- Use early stopping on validation loss.
- Try architectures designed for deepfake detection, such as Xception or
  attention-based models, or frequency-domain features.

---

## Acknowledgements

- [WildDeepfake dataset](https://huggingface.co/datasets/xingjunm/WildDeepfake): Zi et al., *WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection*, ACM MM 2020
- [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/) pretrained models
