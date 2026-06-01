"""Regenerate publication-ready graphs from saved research outputs."""

from __future__ import annotations

import sys
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research.utils import OUTPUTS_DIR, ensure_research_directories, plot_bar_metrics, plot_confusion_matrix, set_publication_style

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def regenerate_confusion_matrix(evaluation_data: Dict) -> None:
    labels = evaluation_data["confusion_matrix"]["labels"]
    matrix = np.asarray(evaluation_data["confusion_matrix"]["counts"])
    plot_confusion_matrix(
        matrix,
        labels,
        OUTPUTS_DIR / "confusion_matrix.png",
        title="OncoAI Confusion Matrix",
    )


def regenerate_roc_curve(evaluation_data: Dict) -> None:
    set_publication_style()
    roc_curve = evaluation_data["roc_curve"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(roc_curve["fpr"], roc_curve["tpr"], color="#0B3C5D", linewidth=2.5, label=f"ROC-AUC = {roc_curve['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OncoAI ROC Curve", pad=12, weight="bold")
    ax.grid(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close(fig)


def regenerate_precision_recall_curve(evaluation_data: Dict) -> None:
    set_publication_style()
    pr_curve = evaluation_data["precision_recall_curve"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pr_curve["recall"], pr_curve["precision"], color="#0B3C5D", linewidth=2.5, label=f"Average Precision = {pr_curve['average_precision']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("OncoAI Precision-Recall Curve", pad=12, weight="bold")
    ax.grid(True)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "precision_recall_curve.png", bbox_inches="tight")
    plt.close(fig)


def regenerate_metric_bars(comparison_rows: List[Dict[str, str]]) -> None:
    rows = []
    for row in comparison_rows:
        rows.append(
            {
                "model": row["model"],
                "accuracy": float(row["accuracy"]),
                "f1_score": float(row["f1_score"]),
                "recall": float(row["recall"]),
            }
        )
    plot_bar_metrics(
        rows,
        metric_name="accuracy",
        output_path=OUTPUTS_DIR / "accuracy_scores.png",
        title="Model Accuracy Comparison",
        ylabel="Accuracy",
    )
    plot_bar_metrics(
        rows,
        metric_name="f1_score",
        output_path=OUTPUTS_DIR / "f1_scores.png",
        title="Model F1-Score Comparison",
        ylabel="F1-Score",
    )
    plot_bar_metrics(
        rows,
        metric_name="recall",
        output_path=OUTPUTS_DIR / "recall_scores.png",
        title="Model Recall Comparison",
        ylabel="Recall",
    )


def main() -> None:
    """Regenerate every output graph from saved metrics files."""
    ensure_research_directories()
    evaluation_path = OUTPUTS_DIR / "evaluation_metrics.json"
    comparison_path = OUTPUTS_DIR / "model_comparison.csv"

    if evaluation_path.exists():
        evaluation_data = load_json(evaluation_path)
        regenerate_confusion_matrix(evaluation_data)
        regenerate_roc_curve(evaluation_data)
        regenerate_precision_recall_curve(evaluation_data)
    else:
        raise FileNotFoundError(f"Missing metrics file: {evaluation_path}")

    if comparison_path.exists():
        regenerate_metric_bars(load_csv(comparison_path))
    else:
        raise FileNotFoundError(f"Missing comparison file: {comparison_path}")

    print("Research graphs regenerated successfully.")


if __name__ == "__main__":
    main()
