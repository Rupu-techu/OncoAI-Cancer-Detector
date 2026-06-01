# OncoAI Research Module

This folder contains a separate research and evaluation pipeline for the existing OncoAI hybrid model:

- EfficientNet-B0 feature extraction
- PCA feature compression
- Random Forest classification
- Optional baseline model comparison
- Publication-ready plots and tables

The module is intentionally separate from the Flask application so it does not affect the UI, routes, or runtime behavior of the web app.

## Folder Structure

```text
research/
├── evaluation_metrics.py
├── model_comparison.py
├── generate_graphs.py
├── utils.py
├── outputs/
└── paper_assets/
```

## How to Run

Run the research evaluation on the saved hybrid model:

```bash
python research/evaluation_metrics.py
```

Run model comparison experiments:

```bash
python research/model_comparison.py
```

Regenerate all graphs from saved outputs:

```bash
python research/generate_graphs.py
```

## Generated Outputs

All outputs are saved to `research/outputs/`:

- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `classification_report.txt`
- `evaluation_metrics.json`
- `model_comparison.csv`
- `accuracy_scores.png`
- `f1_scores.png`
- `recall_scores.png`

## Metrics Explained

- **Accuracy**: Overall fraction of correct predictions.
- **Precision**: Of the predicted positive cases, how many were correct.
- **Recall / Sensitivity**: Of the actual positive cases, how many were detected.
- **F1-score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Measures how well the model separates the two classes across thresholds.
- **Specificity**: Of the actual negative cases, how many were correctly identified.

## Confusion Matrix

The confusion matrix summarizes prediction outcomes:

- **True Negative**: benign correctly predicted as benign
- **False Positive**: benign predicted as malignant
- **False Negative**: malignant predicted as benign
- **True Positive**: malignant correctly predicted as malignant

It is useful for understanding which error types occur most often.

## ROC-AUC

ROC-AUC is a threshold-independent measure of class separability. Higher values indicate that the model is better at ranking malignant samples above benign samples.

## Publication Usage

The generated PNG figures are formatted for:

- IEEE-style figures
- Springer-style manuscripts
- conference slide decks
- thesis chapters and appendices

The figures are saved at high DPI with clean white backgrounds, readable labels, and publication-friendly styling.

## Notes

- The evaluation pipeline uses a deterministic split with `random_state=42`.
- The dataset is capped per class by default to mirror the training configuration.
- XGBoost is optional; if it is not installed, the comparison script skips it gracefully.

