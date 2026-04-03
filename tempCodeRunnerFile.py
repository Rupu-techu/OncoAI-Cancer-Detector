import os
import uuid
from pathlib import Path

import joblib
import json
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image
from torchvision import models, transforms

# -------------------- Paths & constants -------------------- #
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
HEATMAP_FOLDER = BASE_DIR / "static" / "heatmaps"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
HEATMAP_FOLDER.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Load RF pipeline + feature extractor -------------------- #

model_path = MODELS_DIR / "model.pkl"
scaler_path = MODELS_DIR / "scaler.pkl"
pca_path = MODELS_DIR / "pca.pkl"
classes_path = MODELS_DIR / "classes.json"

if not model_path.exists():
    raise FileNotFoundError("model.pkl not found. Run train_model.py first.")

rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)

with open(classes_path, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# feature extractor (EfficientNet-B0)
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
enet = models.efficientnet_b0(weights=weights)
feat_extractor = torch.nn.Sequential(*list(enet.children())[:-1]).to(DEVICE)
feat_extractor.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.meta.get("mean", [0.485, 0.456, 0.406]),
        std=weights.meta.get("std", [0.229, 0.224, 0.225]),
    ),
])


def extract_features(pil_img):
    img = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feat_extractor(img)
        feat = feat.view(feat.size(0), -1).cpu().numpy()
    return feat


def run_inference(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    feats = extract_features(pil_img)
    feats = scaler.transform(feats)
    feats = pca.transform(feats)
    probs = rf_model.predict_proba(feats)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    prediction = idx_to_class[pred_idx]
    risk = float(probs[class_to_idx.get("malignant", pred_idx)])
    return prediction, confidence, risk


# -------------------- Flask app -------------------- #
app = Flask(__name__)
app.secret_key = "oncoai-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


def store_result(data):
    session["last_result"] = data


@app.route("/", methods=["GET", "POST"])
def dashboard():
    if "total_scans" not in session:
        session["total_scans"] = 0
    result = session.get("last_result")
    return render_template(
        "dashboard.html",
        result=result,
        total_scans=session.get("total_scans", 0),
        model_type="RF on EfficientNet-B0 features",
        model_accuracy=None,
    )


@app.route("/report", methods=["GET", "POST"])
def report():
    if "total_scans" not in session:
        session["total_scans"] = 0

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)

            pred_label, conf, risk = run_inference(filepath)

            status = "Cancerous" if pred_label.lower() == "malignant" else "Non-Cancerous"
            badge_class = "badge-danger" if status == "Cancerous" else "badge-success"

            session["total_scans"] += 1
            result = {
                "filename": filename,
                "filepath": str(filepath.relative_to(BASE_DIR)),
                "prediction": status,
                "confidence": round(conf * 100, 2),
                "risk": round(risk * 100, 2),
                "badge_class": badge_class,
                "model_type": "RF on EfficientNet-B0 features",
                "accuracy": "—",
            }
            store_result(result)
            return redirect(url_for("report"))

    result = session.get("last_result")
    analysis = {
        "summary": "AI reviewed the uploaded image and derived a confidence score based on learned patterns.",
        "causes": [
            "Genetic predisposition",
            "Hormonal factors",
            "Environmental exposure",
        ],
        "precautions": [
            "Schedule regular screenings",
            "Maintain healthy BMI and active lifestyle",
            "Avoid smoking and limit alcohol intake",
        ],
        "actions": [
            "Consult a specialist for follow-up",
            "Consider further diagnostic imaging or biopsy",
            "Track any new symptoms and report promptly",
        ],
        "faq": [
            ("How accurate is this result?",
             "Accuracy depends on image quality and training data; always confirm with a clinician."),
            ("Does a Non-Cancerous result guarantee safety?",
             "No. It reduces likelihood but clinical evaluation is essential."),
            ("Can lifestyle change risk?",
             "Yes. Diet, exercise, and avoiding smoking can reduce risk."),
        ],
    }
    return render_template("report.html", result=result, analysis=analysis)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
