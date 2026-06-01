"""Compare several classical models against the EfficientNet + Random Forest hybrid."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research.utils import (
    DEFAULT_MAX_PER_CLASS,
    DEFAULT_TEST_SIZE,
    OUTPUTS_DIR,
    RANDOM_STATE,
    XGBClassifier,
    binary_label_indices,
    binary_classification_metrics,
    class_names_from_mapping,
    ensure_research_directories,
    extract_features,
    load_artifacts,
    load_dataset,
    plot_bar_metrics,
    prepare_split,
    save_csv,
)


def build_models(random_state: int = RANDOM_STATE) -> Dict[str, object]:
    """Create the comparison models used in the research experiments."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "SVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    return models


def main() -> None:
    """Train and compare models on the same deterministic split."""
    ensure_research_directories()

    dataset = load_dataset()
    dataset, train_idx, test_idx = prepare_split(
        dataset=dataset,
        max_per_class=DEFAULT_MAX_PER_CLASS,
        test_size=DEFAULT_TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    artifacts = load_artifacts()
    class_labels = class_names_from_mapping(artifacts.class_to_idx)
    negative_index, positive_index = binary_label_indices(artifacts.class_to_idx, positive_label="malignant")

    X_train_raw, y_train = extract_features(
        dataset,
        train_idx,
        artifacts.feature_extractor,
        device=artifacts.device,
    )
    X_test_raw, y_test = extract_features(
        dataset,
        test_idx,
        artifacts.feature_extractor,
        device=artifacts.device,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=min(100, X_train_scaled.shape[1]), random_state=RANDOM_STATE)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

    comparison_rows: List[Dict[str, object]] = []
    trained_models = build_models()

    for model_name, model in trained_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, positive_index]
        else:
            y_score = y_pred.astype(float)

        metrics = binary_classification_metrics(
            y_test,
            y_pred,
            y_score,
            positive_label=positive_index,
            negative_label=negative_index,
        )
        comparison_rows.append(
            {
                "model": model_name,
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1_score"], 4),
                "roc_auc": round(metrics["roc_auc"], 4),
                "sensitivity": round(metrics["sensitivity"], 4),
                "specificity": round(metrics["specificity"], 4),
                "status": "trained",
            }
        )

    # Include the saved hybrid artifact for reference on the same split
    hybrid_pred = artifacts.rf_model.predict(artifacts.pca.transform(artifacts.scaler.transform(X_test_raw)))
    hybrid_score = artifacts.rf_model.predict_proba(artifacts.pca.transform(artifacts.scaler.transform(X_test_raw)))[:, positive_index]
    hybrid_metrics = binary_classification_metrics(
        y_test,
        hybrid_pred,
        hybrid_score,
        positive_label=positive_index,
        negative_label=negative_index,
    )
    comparison_rows.append(
        {
            "model": "Saved Hybrid Model",
            "accuracy": round(hybrid_metrics["accuracy"], 4),
            "precision": round(hybrid_metrics["precision"], 4),
            "recall": round(hybrid_metrics["recall"], 4),
            "f1_score": round(hybrid_metrics["f1_score"], 4),
            "roc_auc": round(hybrid_metrics["roc_auc"], 4),
            "sensitivity": round(hybrid_metrics["sensitivity"], 4),
            "specificity": round(hybrid_metrics["specificity"], 4),
            "status": "artifact",
        }
    )

    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    save_csv(
        comparison_rows,
        csv_path,
        fieldnames=[
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "sensitivity",
            "specificity",
            "status",
        ],
    )

    plot_bar_metrics(
        comparison_rows,
        metric_name="accuracy",
        output_path=OUTPUTS_DIR / "accuracy_scores.png",
        title="Model Accuracy Comparison",
        ylabel="Accuracy",
    )
    plot_bar_metrics(
        comparison_rows,
        metric_name="f1_score",
        output_path=OUTPUTS_DIR / "f1_scores.png",
        title="Model F1-Score Comparison",
        ylabel="F1-Score",
    )
    plot_bar_metrics(
        comparison_rows,
        metric_name="recall",
        output_path=OUTPUTS_DIR / "recall_scores.png",
        title="Model Recall Comparison",
        ylabel="Recall",
    )

    print(f"Model comparison saved to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
