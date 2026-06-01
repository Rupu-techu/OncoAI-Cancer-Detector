"""
Shared helpers for the OncoAI research module.

This module keeps the research pipeline separate from the Flask app while
reusing the same EfficientNet-B0 feature extractor and saved Random Forest
artifacts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:  # seaborn is optional but preferred for publication-quality heatmaps
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sns = None

try:  # xgboost is optional
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = RESEARCH_ROOT / "outputs"
FIGURES_DIR = RESEARCH_ROOT / "paper_assets" / "figures"
TABLES_DIR = RESEARCH_ROOT / "paper_assets" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "images_multiclass",
    PROJECT_ROOT / "data" / "images_binary",
]

RANDOM_STATE = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MAX_PER_CLASS = 300
DEFAULT_NUM_WORKERS = 0


def ensure_research_directories() -> None:
    """Create all expected research output directories."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dataset_dir(preferred: Optional[Path] = None) -> Path:
    """
    Resolve the dataset directory used for research experiments.

    Preference order:
    1) an explicitly supplied path
    2) `data/images_multiclass`
    3) `data/images_binary`
    """
    if preferred and preferred.exists():
        return preferred
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No supported dataset directory found. Expected one of: "
        f"{', '.join(str(path) for path in DATASET_CANDIDATES)}"
    )


def get_image_transform():
    """Build the EfficientNet-B0 normalization pipeline."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.meta.get("mean", [0.485, 0.456, 0.406]),
                std=weights.meta.get("std", [0.229, 0.224, 0.225]),
            ),
        ]
    )


def get_feature_extractor(device: Optional[torch.device] = None):
    """Return the pretrained EfficientNet-B0 feature extractor."""
    device = device or get_device()
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    backbone = models.efficientnet_b0(weights=weights)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(device)
    feature_extractor.eval()
    feature_dim = backbone.classifier[1].in_features
    return feature_extractor, feature_dim


def load_dataset(dataset_dir: Optional[Path] = None):
    """Load the image dataset with the standard transform."""
    dataset_dir = resolve_dataset_dir(dataset_dir)
    return datasets.ImageFolder(dataset_dir, transform=get_image_transform())


def cap_indices_by_class(dataset, max_per_class: int = DEFAULT_MAX_PER_CLASS) -> np.ndarray:
    """
    Optionally cap the number of examples per class to mirror the training script.

    Set `max_per_class=0` to disable capping.
    """
    if not max_per_class:
        return np.arange(len(dataset))

    per_class = {class_index: 0 for class_index in dataset.class_to_idx.values()}
    selected: List[int] = []
    for sample_index, (_, target) in enumerate(dataset.samples):
        if per_class[target] < max_per_class:
            selected.append(sample_index)
            per_class[target] += 1
    return np.asarray(selected)


def build_stratified_split(
    labels: Sequence[int],
    indices: Sequence[int],
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic train/test indices with stratification."""
    label_subset = [labels[i] for i in indices]
    train_idx, test_idx = train_test_split(
        np.asarray(indices),
        test_size=test_size,
        stratify=label_subset,
        random_state=random_state,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def class_names_from_mapping(class_to_idx: Dict[str, int]) -> List[str]:
    """Return class labels ordered by index."""
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def positive_class_index(class_to_idx: Dict[str, int], positive_label: str = "malignant") -> int:
    """Locate the malignant class index with a safe fallback."""
    if positive_label in class_to_idx:
        return int(class_to_idx[positive_label])
    return max(class_to_idx.values())


def binary_label_indices(
    class_to_idx: Dict[str, int],
    positive_label: str = "malignant",
) -> Tuple[int, int]:
    """Return `(negative_index, positive_index)` for a binary label mapping."""
    positive_index = positive_class_index(class_to_idx, positive_label=positive_label)
    all_indices = sorted(int(index) for index in class_to_idx.values())
    negative_index = next((index for index in all_indices if index != positive_index), positive_index)
    return negative_index, positive_index


@dataclass
class ResearchArtifacts:
    """Container for the shared model artifacts used in research scripts."""

    rf_model: object
    scaler: StandardScaler
    pca: PCA
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    feature_extractor: torch.nn.Module
    device: torch.device


def load_artifacts(
    models_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> ResearchArtifacts:
    """Load the saved Random Forest, scaler, PCA, and EfficientNet feature extractor."""
    device = device or get_device()
    models_dir = models_dir or MODELS_DIR

    rf_model = joblib.load(models_dir / "model.pkl")
    if hasattr(rf_model, "n_jobs"):
        rf_model.n_jobs = 1

    scaler = joblib.load(models_dir / "scaler.pkl")
    pca = joblib.load(models_dir / "pca.pkl")

    classes_path = models_dir / "classes.json"
    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as handle:
            class_to_idx = json.load(handle)
    else:
        dataset = load_dataset()
        class_to_idx = dataset.class_to_idx

    idx_to_class = {int(index): name for name, index in class_to_idx.items()}
    feature_extractor, _ = get_feature_extractor(device=device)
    return ResearchArtifacts(
        rf_model=rf_model,
        scaler=scaler,
        pca=pca,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        feature_extractor=feature_extractor,
        device=device,
    )


def extract_features(
    dataset,
    indices: Sequence[int],
    feature_extractor: torch.nn.Module,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    device: Optional[torch.device] = None,
):
    """Extract EfficientNet feature vectors and labels for the requested indices."""
    device = device or get_device()
    subset = Subset(dataset, list(indices))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = feature_extractor(images)
            outputs = outputs.view(outputs.size(0), -1).cpu().numpy()
            features.append(outputs)
            labels.append(targets.numpy())

    if not features:
        return np.empty((0, 0)), np.empty((0,), dtype=int)
    return np.vstack(features), np.concatenate(labels)


def prepare_split(
    dataset=None,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Prepare a deterministic train/test split and return the dataset plus indices."""
    dataset = dataset or load_dataset()
    indices = cap_indices_by_class(dataset, max_per_class=max_per_class)
    train_idx, test_idx = build_stratified_split(dataset.targets, indices, test_size=test_size, random_state=random_state)
    return dataset, train_idx, test_idx


def predict_positive_probabilities(model, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict labels and the positive-class score for any sklearn-compatible classifier.
    """
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            positive_scores = probabilities[:, 1]
        else:
            positive_scores = probabilities.reshape(-1)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        positive_scores = 1 / (1 + np.exp(-decision))
    else:
        positive_scores = predictions.astype(float)
    return np.asarray(predictions), np.asarray(positive_scores)


def binary_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    positive_label: int = 1,
    negative_label: Optional[int] = None,
) -> Dict[str, float]:
    """Compute the core binary research metrics used in the module."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    if negative_label is None:
        negative_label = 1 - positive_label if positive_label in {0, 1} else min(int(y_true.min()), int(positive_label))

    cm = confusion_matrix(y_true, y_pred, labels=[negative_label, positive_label])
    if cm.shape != (2, 2):
        tn = fp = fn = tp = 0
    else:
        tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def save_json(data: Dict, path: Path) -> None:
    """Persist a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_text(text: str, path: Path) -> None:
    """Persist plain text output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def save_csv(rows: List[Dict], path: Path, fieldnames: Sequence[str]) -> None:
    """Persist a list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_publication_style() -> None:
    """Apply publication-friendly matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#d0d7de",
            "grid.color": "#d9e2ec",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
    )


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_labels: Sequence[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a publication-ready confusion matrix."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            square=True,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            linewidths=0.5,
            linecolor="#ffffff",
        )
    else:  # pragma: no cover - fallback when seaborn is unavailable
        image = ax.imshow(confusion, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                ax.text(j, i, int(confusion[i, j]), ha="center", va="center", fontsize=11)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, pad=12, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    output_path: Path,
    title: str = "Receiver Operating Characteristic",
) -> Dict[str, List[float]]:
    """Plot ROC curve and return the sampled points for later regeneration."""
    set_publication_style()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#0B3C5D", linewidth=2.5, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, pad=12, weight="bold")
    ax.grid(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "fpr": [float(value) for value in fpr],
        "tpr": [float(value) for value in tpr],
        "thresholds": [float(value) for value in thresholds],
        "roc_auc": float(roc_auc),
    }


def plot_precision_recall_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    output_path: Path,
    title: str = "Precision-Recall Curve",
) -> Dict[str, List[float]]:
    """Plot precision-recall curve and return the sampled points for later regeneration."""
    set_publication_style()
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#0B3C5D", linewidth=2.5, label=f"Average Precision = {pr_auc:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, pad=12, weight="bold")
    ax.grid(True)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "precision": [float(value) for value in precision],
        "recall": [float(value) for value in recall],
        "thresholds": [float(value) for value in thresholds],
        "average_precision": float(pr_auc),
    }


def plot_bar_metrics(
    rows: Sequence[Dict[str, float]],
    metric_name: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Create a ranked bar chart for model-comparison metrics."""
    set_publication_style()
    ordered = sorted(rows, key=lambda row: row.get(metric_name, 0.0), reverse=True)
    labels = [row["model"] for row in ordered]
    values = [row.get(metric_name, 0.0) for row in ordered]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#0B3C5D" if index == 0 else "#5BC0EB" for index in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, edgecolor="#0b3c5d", linewidth=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, weight="bold")
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def safe_float(value) -> float:
    """Convert values to float while handling missing/NA inputs."""
    try:
        if value is None:
            return float("nan")
        if isinstance(value, str) and value.lower() in {"na", "nan", ""}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")
