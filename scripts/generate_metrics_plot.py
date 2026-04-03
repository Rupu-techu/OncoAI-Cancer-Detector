"""
Generate simple train/validation performance graphs for the saved RF model.
Outputs:
  - static/reports/train_val_accuracy.png
  - static/reports/confusion_matrix.png
  - static/reports/metrics.json
"""
from pathlib import Path
import json

import matplotlib

# Use headless backend so plotting works without a GUI (prevents TclError on Windows servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# --- Config (keep in sync with train_model.py) --- #
DATA_DIR = Path("data/images_multiclass")
MODELS_DIR = Path("models")
OUT_DIR = Path("static/reports")
BATCH_SIZE = 64
VAL_SPLIT = 0.2
RANDOM_STATE = 42
NUM_WORKERS = 0
QUICK_MAX_PER_CLASS = 300  # cap per class for speed; 0 disables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_extractor():
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    feat = torch.nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
    feat.eval()
    norm_mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    norm_std = weights.meta.get("std", [0.229, 0.224, 0.225])
    return feat, norm_mean, norm_std


def extract_features(loader, feat_model):
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            out = feat_model(imgs)
            out = out.view(out.size(0), -1).cpu().numpy()
            feats.append(out)
            labels.append(lbls.numpy())
    return np.vstack(feats), np.concatenate(labels)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_DIR}")

    # load artifacts
    rf_model = joblib.load(MODELS_DIR / "model.pkl")
    rf_model.n_jobs = 1  # avoid multiprocessing issues in restricted environments
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    pca = joblib.load(MODELS_DIR / "pca.pkl")
    with open(MODELS_DIR / "classes.json") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    feat_model, norm_mean, norm_std = get_feature_extractor()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # cap per class for consistency with training
    if QUICK_MAX_PER_CLASS:
        per_class = {c: 0 for c in dataset.class_to_idx.values()}
        filtered = []
        for i, (_, target) in enumerate(dataset.samples):
            if per_class[target] < QUICK_MAX_PER_CLASS:
                filtered.append(i)
                per_class[target] += 1
        idx = np.array(filtered)
    else:
        idx = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(
        idx,
        test_size=VAL_SPLIT,
        stratify=[dataset.targets[i] for i in idx],
        random_state=RANDOM_STATE,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # feature extraction -> scaling -> pca -> predict
    X_train, y_train = extract_features(train_loader, feat_model)
    X_val, y_val = extract_features(val_loader, feat_model)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)

    train_pred = rf_model.predict(X_train)
    val_pred = rf_model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Accuracy bar chart --- #
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Train", "Validation"], [train_acc * 100, val_acc * 100],
                  color=["#0B3C5D", "#5BC0EB"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy (Train vs Validation)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "train_val_accuracy.png", dpi=160)
    plt.close(fig)

    # --- Confusion matrix on validation --- #
    cm = confusion_matrix(y_val, val_pred, labels=sorted(idx_to_class.keys()))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(idx_to_class)))
    ax.set_yticks(range(len(idx_to_class)))
    ax.set_xticklabels([idx_to_class[i] for i in sorted(idx_to_class.keys())], rotation=35, ha="right")
    ax.set_yticklabels([idx_to_class[i] for i in sorted(idx_to_class.keys())])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    metrics = {
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "classes": idx_to_class,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved plots and metrics to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
