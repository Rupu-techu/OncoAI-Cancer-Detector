import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from torchcam.methods import GradCAM, SmoothGradCAMpp
from torchvision import models, transforms
from werkzeug.utils import secure_filename

# -------------------- Paths & constants -------------------- #
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
GRADCAM_FOLDER = UPLOAD_FOLDER / "gradcam"
REPORTS_FOLDER = BASE_DIR / "static" / "reports"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
GRADCAM_FOLDER.mkdir(parents=True, exist_ok=True)
REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

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

# Feature extractors
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
enet = models.efficientnet_b0(weights=weights)
feat_extractor = torch.nn.Sequential(*list(enet.children())[:-1]).to(DEVICE)
feat_extractor.eval()

attention_model = models.efficientnet_b0(weights=weights).to(DEVICE)
attention_model.eval()

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
    prediction = idx_to_class[pred_idx]
    malignancy_probability = float(probs[class_to_idx.get("malignant", pred_idx)])
    return prediction, malignancy_probability


def calculate_risk_category(malignancy_probability_percent):
    if malignancy_probability_percent <= 25:
        return "Low Concern", "risk-low"
    if malignancy_probability_percent <= 50:
        return "Mild Concern", "risk-mild"
    if malignancy_probability_percent <= 75:
        return "Moderate Concern", "risk-moderate"
    return "High Concern", "risk-high"


def build_cam_extractor():
    target_layer = attention_model.features[-1]
    try:
        return SmoothGradCAMpp(attention_model, target_layer=target_layer)
    except Exception:
        return GradCAM(attention_model, target_layer=target_layer)


def generate_gradcam_overlay(img_path, output_stem):
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    extractor = build_cam_extractor()

    try:
        attention_model.zero_grad(set_to_none=True)
        scores = attention_model(input_tensor)
        class_idx = int(scores.argmax(dim=1).item())
        cam = extractor(class_idx=class_idx, scores=scores)[0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, pil_img.size)
        original = np.array(pil_img)
        overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)

        gradcam_filename = f"{output_stem}_gradcam.png"
        gradcam_path = GRADCAM_FOLDER / gradcam_filename
        Image.fromarray(overlay).save(gradcam_path)
        return str(gradcam_path.relative_to(BASE_DIR)), gradcam_filename
    finally:
        extractor.remove_hooks()


def pdf_image(path, max_width, max_height):
    reader = ImageReader(str(path))
    image_width, image_height = reader.getSize()
    scale = min(max_width / image_width, max_height / image_height)
    return RLImage(str(path), width=image_width * scale, height=image_height * scale)


def risk_badge_text(risk_class):
    return {
        "risk-low": "Low Concern",
        "risk-mild": "Mild Concern",
        "risk-moderate": "Moderate Concern",
        "risk-high": "High Concern",
    }.get(risk_class, "Low Concern")


def generate_pdf_report(result, analysis, original_image_path, gradcam_path):
    report_filename = f"oncoai_report_{result['analysis_token']}.pdf"
    pdf_path = REPORTS_FOLDER / report_filename

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="OncoTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0d3b66"),
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="OncoHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0d3b66"),
        fontSize=14,
        leading=18,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="OncoBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#24344d"),
    ))
    styles.add(ParagraphStyle(
        name="OncoSmall",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#5f6f86"),
    ))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=(8.27 * inch, 11.69 * inch),
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    story = []
    story.append(Paragraph("OncoAI", styles["OncoTitle"]))
    story.append(Paragraph("AI Screening Report", styles["Heading2"]))
    story.append(Paragraph(f"Analysis Date and Time: {result['analysis_date']}", styles["OncoSmall"]))
    story.append(Spacer(1, 0.12 * inch))

    summary_table = Table(
        [
            ["AI Screening Result", result["prediction_label"]],
            ["Malignancy Probability", f"{result['malignancy_probability']}%"],
            ["Risk Category", risk_badge_text(result["risk_badge_class"])],
        ],
        colWidths=[2.15 * inch, 4.45 * inch],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef5fb")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#24344d")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#c9d9ea")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe6f1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LEADING", (0, 0), (-1, -1), 14),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fbff")]),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph("Model Attention Analysis", styles["OncoHeading"]))
    story.append(Paragraph(
        "The highlighted regions indicate image areas that contributed most strongly to the AI assessment.",
        styles["OncoBody"],
    ))
    story.append(Spacer(1, 0.10 * inch))

    original_block = [Paragraph("Original Image", styles["OncoSmall"]), Spacer(1, 0.05 * inch),
                      pdf_image(original_image_path, 2.8 * inch, 2.3 * inch)]
    if gradcam_path and Path(gradcam_path).exists():
        gradcam_block = [Paragraph("Grad-CAM Visualization", styles["OncoSmall"]), Spacer(1, 0.05 * inch),
                         pdf_image(gradcam_path, 2.8 * inch, 2.3 * inch)]
    else:
        gradcam_block = [Paragraph("Grad-CAM Visualization", styles["OncoSmall"]), Spacer(1, 0.05 * inch),
                         Paragraph("Attention analysis could not be generated for this image.", styles["OncoBody"])]

    visuals = Table([[original_block, gradcam_block]], colWidths=[3.0 * inch, 3.0 * inch], hAlign="CENTER")
    visuals.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#c9d9ea")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe6f1")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(visuals)
    story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph("Recommendations", styles["OncoHeading"]))
    for item in analysis.get("possible_findings", []):
        story.append(Paragraph(f"- {item}", styles["OncoBody"]))
    if analysis.get("possible_findings"):
        story.append(Spacer(1, 0.06 * inch))
    for item in analysis.get("recommendations", []):
        story.append(Paragraph(f"- {item}", styles["OncoBody"]))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Medical Disclaimer", styles["OncoHeading"]))
    story.append(Paragraph(
        "This system is an AI-assisted screening tool designed for educational and research purposes. Results are generated through image pattern analysis and should not be considered a medical interpretation. Always consult a qualified healthcare professional for clinical evaluation and treatment decisions.",
        styles["OncoBody"],
    ))

    doc.build(story)
    return report_filename


# -------------------- Flask app -------------------- #
app = Flask(__name__)
app.secret_key = "oncoai-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ANALYSIS_DEFAULT = {
    "summary": "Upload an image to view the AI-assisted screening summary.",
    "possible_findings": [],
    "recommendations": [],
    "faq": [
        ("How accurate is this result?",
         "Accuracy depends on image quality and training data; always confirm with a clinician."),
        ("Does a Benign result guarantee safety?",
         "No. It reduces likelihood but clinical evaluation is essential."),
        ("Can lifestyle change risk?",
         "Yes. Diet, exercise, and avoiding smoking can reduce risk."),
    ],
}


def build_analysis(prediction):
    if prediction == "malignant":
        return {
            "summary": "Malignant tissue patterns detected. Prioritize specialist review.",
            "possible_findings": [
                "Malignant tissue patterns detected",
                "Abnormal cellular morphology",
                "Suspicious histopathological features",
            ],
            "recommendations": [
                "Seek medical consultation promptly",
                "Consider additional clinical evaluation",
                "Discuss findings with a qualified specialist",
            ],
            "faq": ANALYSIS_DEFAULT["faq"],
        }
    if prediction == "benign":
        return {
            "summary": "Benign tissue patterns detected. Continue routine screening and monitoring.",
            "possible_findings": [
                "Benign tissue changes",
                "Fibroadenoma-like patterns",
                "Non-malignant cellular structures",
            ],
            "recommendations": [
                "Continue routine screening",
                "Monitor changes over time",
                "Consult healthcare professional if symptoms exist",
            ],
            "faq": ANALYSIS_DEFAULT["faq"],
        }
    return ANALYSIS_DEFAULT


def store_result(result, analysis):
    session["last_result"] = result
    session["last_analysis"] = analysis


def ensure_session_initialized():
    if "total_scans" not in session:
        session["total_scans"] = 0
    if not session.get("initialized"):
        session["initialized"] = True
        session.pop("last_result", None)


@app.route("/", methods=["GET", "POST"])
def dashboard():
    ensure_session_initialized()
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
    ensure_session_initialized()

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)

            pred_label, malignancy_probability = run_inference(filepath)
            pred_label = pred_label.lower()
            is_malignant = pred_label == "malignant"
            risk_category, risk_badge_class = calculate_risk_category(round(malignancy_probability * 100, 2))
            analysis_timestamp = datetime.now().strftime("%B %d, %Y %I:%M %p")
            analysis_token = uuid.uuid4().hex

            gradcam_filepath = None
            gradcam_filename = None
            try:
                gradcam_filepath, gradcam_filename = generate_gradcam_overlay(filepath, analysis_token)
            except Exception:
                gradcam_filepath = None
                gradcam_filename = None

            session["total_scans"] += 1
            result = {
                "filename": filename,
                "filepath": str(filepath.relative_to(BASE_DIR)),
                "prediction": pred_label,
                "prediction_label": "Malignant Pattern Detected" if is_malignant else "Benign Pattern Detected",
                "prediction_accent_class": "result-badge-danger" if is_malignant else "result-badge-success",
                "malignancy_probability": round(malignancy_probability * 100, 2),
                "risk_category": risk_category,
                "risk_badge_class": risk_badge_class,
                "analysis_date": analysis_timestamp,
                "analysis_token": analysis_token,
                "gradcam_filename": gradcam_filename,
                "gradcam_filepath": gradcam_filepath,
                "model_type": "RF on EfficientNet-B0 features",
                "accuracy": "-",
            }
            analysis = build_analysis(pred_label)

            try:
                report_filename = generate_pdf_report(result, analysis, filepath, gradcam_filepath)
                result["report_filename"] = report_filename
                result["report_filepath"] = str((REPORTS_FOLDER / report_filename).relative_to(BASE_DIR))
            except Exception:
                result["report_filename"] = None
                result["report_filepath"] = None

            store_result(result, analysis)
            return redirect(url_for("report"))

    result = session.get("last_result")
    analysis = build_analysis(result["prediction"]) if result else build_analysis(None)
    return render_template("report.html", result=result, analysis=analysis)


@app.route("/reports/<path:report_filename>")
def download_report(report_filename):
    return send_from_directory(REPORTS_FOLDER, report_filename, as_attachment=True)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
