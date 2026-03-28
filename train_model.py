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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------- Config ---------------- #
DATA_DIR = Path("data/images_multiclass")
MODELS_DIR = Path("models")
BATCH_SIZE = 64
VAL_SPLIT = 0.2
RANDOM_STATE = 42
N_PCA = 100
N_TREES = 200
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK_MAX_PER_CLASS = 300  # cap per class to speed up; set 0 to disable


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

    print("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)
    val_pred = rf.predict(X_val)
    print("\nTrain classification report:")
    print(classification_report(y_train, train_pred, target_names=[idx_to_class[i] for i in range(num_classes)]))
    print("\nVal classification report:")
    print(classification_report(y_val, val_pred, target_names=[idx_to_class[i] for i in range(num_classes)]))

    joblib.dump(rf, MODELS_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(pca, MODELS_DIR / "pca.pkl")
    with open(MODELS_DIR / "classes.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"\nSaved artifacts to {MODELS_DIR}")
    print(f"Total time: {(time.time() - start):.1f}s")


if __name__ == "__main__":
    main()
