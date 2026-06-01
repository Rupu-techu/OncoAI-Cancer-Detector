"""Evaluate the saved OncoAI hybrid model on a deterministic hold-out split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research.utils import (
    DEFAULT_MAX_PER_CLASS,
    DEFAULT_TEST_SIZE,
    OUTPUTS_DIR,
    RANDOM_STATE,
    binary_label_indices,
    binary_classification_metrics,
    class_names_from_mapping,
    ensure_research_directories,
    extract_features,
    load_artifacts,
    load_dataset,
    positive_class_index,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    prepare_split,
    save_json,
    save_text,
)


def main() -> None:
    """Run the complete research evaluation pipeline."""
    ensure_research_directories()

    dataset, train_idx, test_idx = prepare_split(
        dataset=load_dataset(),
        max_per_class=DEFAULT_MAX_PER_CLASS,
        test_size=DEFAULT_TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    artifacts = load_artifacts()
    class_labels = class_names_from_mapping(artifacts.class_to_idx)
    negative_index, positive_index = binary_label_indices(artifacts.class_to_idx, positive_label="malignant")

    test_features, y_test = extract_features(
        dataset,
        test_idx,
        artifacts.feature_extractor,
        device=artifacts.device,
    )
    test_features = artifacts.scaler.transform(test_features)
    test_features = artifacts.pca.transform(test_features)

    y_pred = artifacts.rf_model.predict(test_features)
    if hasattr(artifacts.rf_model, "predict_proba"):
        y_proba = artifacts.rf_model.predict_proba(test_features)[:, positive_index]
    else:
        y_proba = y_pred.astype(float)

    metrics = binary_classification_metrics(
        y_test,
        y_pred,
        y_proba,
        positive_label=positive_index,
        negative_label=negative_index,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[negative_index, positive_index])
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_labels,
        zero_division=0,
        digits=4,
    )

    roc_payload = plot_roc_curve(
        y_test,
        y_proba,
        OUTPUTS_DIR / "roc_curve.png",
        title="OncoAI ROC Curve",
    )
    pr_payload = plot_precision_recall_curve(
        y_test,
        y_proba,
        OUTPUTS_DIR / "precision_recall_curve.png",
        title="OncoAI Precision-Recall Curve",
    )
    plot_confusion_matrix(
        cm,
        class_labels,
        OUTPUTS_DIR / "confusion_matrix.png",
        title="OncoAI Confusion Matrix",
    )

    save_text(report, OUTPUTS_DIR / "classification_report.txt")
    save_json(
        {
            "dataset": str(dataset.root),
            "split": {
                "random_state": RANDOM_STATE,
                "test_size": DEFAULT_TEST_SIZE,
                "max_per_class": DEFAULT_MAX_PER_CLASS,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            },
            "classes": class_labels,
            "metrics": metrics,
            "confusion_matrix": {
                "labels": class_labels,
                "counts": cm.tolist(),
            },
            "roc_curve": roc_payload,
            "precision_recall_curve": pr_payload,
        },
        OUTPUTS_DIR / "evaluation_metrics.json",
    )

    print("OncoAI evaluation complete.")
    print(f"Outputs saved in: {OUTPUTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
