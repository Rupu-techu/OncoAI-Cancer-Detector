"""
Hybrid training: EfficientNet-B0 pretrained feature extractor + PCA + RandomForest.
Dataset expected at data/images_multiclass/<class>/image.jpg
Saves model.pkl, scaler.pkl, pca.pkl, classes.json to models/.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import matplotlib

# Use headless backend so plots render without a GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------- Config ---------------- #
DATA_DIR = Path("data/images_multiclass")
MODELS_DIR = Path("models")
OUT_DIR = Path("static/reports")
BATCH_SIZE = 64
VAL_SPLIT = 0.2
RANDOM_STATE = 42
N_PCA = 100
N_TREES = 200
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK_MAX_PER_CLASS = 300  # cap per class to speed up; set 0 to disable
PLOT_METRICS = True


def get_feature_extractor():
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    modules = list(model.children())[:-1]  # remove classifier
    feat = nn.Sequential(*modules).to(DEVICE)
    feat.eval()
    feat_dim = model.classifier[1].in_features  # 1280
    norm_mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    norm_std = weights.meta.get("std", [0.229, 0.224, 0.225])
    return feat, feat_dim, norm_mean, norm_std


def extract_features(loader, feat_model):
    feats = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            out = feat_model(imgs)  # (B, 2048,1,1)
            out = out.view(out.size(0), -1).cpu().numpy()
            feats.append(out)
            labels.append(lbls.numpy())
    return np.vstack(feats), np.concatenate(labels)


def main():
    start = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    if PLOT_METRICS:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    feat_model, feat_dim, norm_mean, norm_std = get_feature_extractor()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
    print(f"Classes: {idx_to_class}")

    # Optional cap per class for faster runs
    if QUICK_MAX_PER_CLASS:
        per_class = {c: 0 for c in class_to_idx.values()}
        filtered = []
        for i, (_, target) in enumerate(dataset.samples):
            if per_class[target] < QUICK_MAX_PER_CLASS:
                filtered.append(i)
                per_class[target] += 1
        idx = np.array(filtered)
    else:
        idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        idx, test_size=VAL_SPLIT, stratify=[dataset.targets[i] for i in idx], random_state=RANDOM_STATE
    )
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print("Extracting train features...")
    X_train, y_train = extract_features(train_loader, feat_model)
    print("Extracting val features...")
    X_val, y_val = extract_features(val_loader, feat_model)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    pca = PCA(n_components=min(N_PCA, feat_dim), random_state=RANDOM_STATE)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)

    # class weights for balanced training (warm_start safe)
    classes = np.unique(y_train)
    cw_vals = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {cls: w for cls, w in zip(classes, cw_vals)}

    print("Training RandomForest...")
    # Incremental fit to record curves similar to epoch plots
    rf = RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        class_weight=class_weights,
        random_state=RANDOM_STATE,
        n_jobs=1  # avoid multiprocessing issues on some Windows setups
    )

    step = max(5, N_TREES // 10)  # ~10 points
    tree_steps = list(range(step, N_TREES + 1, step))
    if tree_steps[-1] != N_TREES:
        tree_steps.append(N_TREES)

    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    for n in tree_steps:
        rf.set_params(n_estimators=n)
        rf.fit(X_train, y_train)
        train_proba = rf.predict_proba(X_train)
        val_proba = rf.predict_proba(X_val)
        train_pred = train_proba.argmax(axis=1)
        val_pred = val_proba.argmax(axis=1)
        train_acc_hist.append(accuracy_score(y_train, train_pred))
        val_acc_hist.append(accuracy_score(y_val, val_pred))
        train_loss_hist.append(log_loss(y_train, train_proba, labels=list(range(num_classes))))
        val_loss_hist.append(log_loss(y_val, val_proba, labels=list(range(num_classes))))

    # final metrics from last step
    train_acc = train_acc_hist[-1]
    val_acc = val_acc_hist[-1]
    cm = confusion_matrix(y_val, val_pred, labels=list(range(num_classes)))
    print("\nTrain classification report:")
    print(classification_report(y_train, train_pred, target_names=[idx_to_class[i] for i in range(num_classes)]))
    print("\nVal classification report:")
    print(classification_report(y_val, val_pred, target_names=[idx_to_class[i] for i in range(num_classes)]))

    if PLOT_METRICS:
        # training curves (accuracy and log loss vs number of trees)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(tree_steps, train_acc_hist, label="Train", color="#1f77b4")
        axes[0].plot(tree_steps, val_acc_hist, label="Validation", color="#ff7f0e")
        axes[0].set_xlabel("Number of Trees")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy vs Trees")
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(alpha=0.3, linestyle="--")
        axes[0].legend()

        axes[1].plot(tree_steps, train_loss_hist, label="Train", color="#1f77b4")
        axes[1].plot(tree_steps, val_loss_hist, label="Validation", color="#ff7f0e")
        axes[1].set_xlabel("Number of Trees")
        axes[1].set_ylabel("Log Loss")
        axes[1].set_title("Loss vs Trees")
        axes[1].grid(alpha=0.3, linestyle="--")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR / "train_val_curves.png", dpi=160)
        plt.close(fig)

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

        # confusion matrix (validation)
        cm = confusion_matrix(y_val, val_pred, labels=list(range(num_classes)))
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels([idx_to_class[i] for i in range(num_classes)], rotation=35, ha="right")
        ax.set_yticklabels([idx_to_class[i] for i in range(num_classes)])
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
            "confusion_matrix": {
                "labels": [idx_to_class[i] for i in range(num_classes)],
                "counts": cm.tolist(),
            },
            "curve": {
                "trees": tree_steps,
                "train_accuracy": [round(x, 4) for x in train_acc_hist],
                "val_accuracy": [round(x, 4) for x in val_acc_hist],
                "train_loss": [round(x, 6) for x in train_loss_hist],
                "val_loss": [round(x, 6) for x in val_loss_hist],
            },
        }
        with open(OUT_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved plots to {OUT_DIR / 'train_val_accuracy.png'}, {OUT_DIR / 'confusion_matrix.png'}, and {OUT_DIR / 'train_val_curves.png'}")
        print(f"Metrics JSON: {OUT_DIR / 'metrics.json'}")

    joblib.dump(rf, MODELS_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(pca, MODELS_DIR / "pca.pkl")
    with open(MODELS_DIR / "classes.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"\nSaved artifacts to {MODELS_DIR}")
    print(f"Total time: {(time.time() - start):.1f}s")


if __name__ == "__main__":
    main()
