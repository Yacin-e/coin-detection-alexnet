# 🪙 Coin Classification with AlexNet

**Master 1 Informatique VMI — IFLBE055 Réseaux de neurones pour la vision par ordinateur**

Image classification of coins using a CNN (AlexNet), based on the [DL4CV Coin Classification](https://www.kaggle.com/competitions/dl4cv-coin-classification) Kaggle challenge.

> Given an image of a coin, predict its class — e.g. `"1 Cent,Australian dollar,australia"`

---

## 📁 Dataset Structure

Downloaded from Google Drive (link provided in the TP), the dataset is organized as:

```
coin_dataset/kaggle/
  train.csv        ← Id, Class (10 368 rows, 315 classes)
  test.csv         ← Id only   (1 282 rows)
  train/           ← {Id}.jpg
  test/            ← {Id}.jpg
```

The training set is further split 85/15 into train (8 804 images) and validation (1 554 images).

---

## 🏗️ Model

**AlexNet** implemented from scratch, faithful to the original paper:

> Krizhevsky, Sutskever & Hinton — *ImageNet Classification with Deep Convolutional Neural Networks* (NIPS 2012)

| Layer  | Details |
|--------|---------|
| Conv1  | 96 kernels 11×11, stride=4 → ReLU → LRN → MaxPool |
| Conv2  | 256 kernels 5×5 → ReLU → LRN → MaxPool |
| Conv3  | 384 kernels 3×3 → ReLU |
| Conv4  | 384 kernels 3×3 → ReLU |
| Conv5  | 256 kernels 3×3 → ReLU → MaxPool |
| FC6-7  | 4096 neurons → ReLU → Dropout(0.5) |
| FC8    | 315 classes |

A **fine-tuned variant** (ImageNet pretrained weights) is also included as an improvement.

---

## ⚙️ Training

| Setting | Scratch | Fine-tuned |
|---------|---------|------------|
| Optimizer | Adam | SGD |
| Learning rate | 1e-4 | 1e-3 |
| Epochs | 50 | 30 |
| Scheduler | ReduceLROnPlateau | CosineAnnealing |
| Loss | CrossEntropyLoss | CrossEntropyLoss |

**Data augmentation** (coin-specific):
- `RandomRotation(180)` — coins are rotationally symmetric
- `RandomHorizontalFlip`, `RandomVerticalFlip`
- `ColorJitter`
- `RandomCrop(224)` from 256×256

---

## 📊 Results

| Model | Val Accuracy |
|-------|-------------|
| AlexNet scratch (10 epochs) | ~10% |
| AlexNet fine-tuned (13 epochs) | ~72% |
| Kaggle Leaderboard #1 | 90.17% |

The scratch model converges slowly due to the small dataset size (~8 800 images vs 1.2M in the original paper). Fine-tuning with ImageNet weights gives a much stronger baseline.

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Enable GPU: *Runtime > Change runtime type > T4 GPU*
3. Set your Kaggle credentials in cell 2 (or upload `kaggle.json`)
4. Run all cells in order

---

## 📦 Requirements

All handled automatically in the notebook:
- PyTorch / torchvision
- scikit-learn
- pandas, matplotlib, seaborn
- gdown, kaggle

---

## 🧠 Possible Improvements

- **Better architecture** — ResNet-50, EfficientNet-B4, ConvNeXt
- **Advanced augmentation** — MixUp, CutMix, RandAugment
- **Longer fine-tuning** — more epochs with cosine annealing
- **Ensemble** — average predictions from multiple models
- **TTA** — Test-Time Augmentation (10 crops, as in the original paper)
