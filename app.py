import json
import uuid
from datetime import datetime
from pathlib import Path
import traceback

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
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
    malignant_probability = float(probs[class_to_idx.get("malignant", pred_idx)])
    benign_probability = float(probs[class_to_idx.get("benign", pred_idx)])
    probability_breakdown = {
        "malignant": round(malignant_probability * 100, 2),
        "benign": round(benign_probability * 100, 2),
    }
    return prediction, malignant_probability, probability_breakdown


def calculate_risk_category(malignancy_probability_percent):
    if malignancy_probability_percent <= 25:
        return "Low Concern", "risk-low"
    if malignancy_probability_percent <= 50:
        return "Mild Concern", "risk-mild"
    if malignancy_probability_percent <= 75:
        return "Moderate Concern", "risk-moderate"
    return "High Concern", "risk-high"


def calculate_evidence_level(probability_percent):
    if probability_percent <= 60:
        return "Low Evidence", "evidence-low"
    if probability_percent <= 75:
        return "Moderate Evidence", "evidence-moderate"
    if probability_percent <= 90:
        return "Strong Evidence", "evidence-strong"
    return "Very Strong Evidence", "evidence-very-strong"


def select_gradcam_target_layer(model):
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv
    return model.features[-1]


def build_patient_explanation(prediction):
    if prediction == "malignant":
        observed_text = (
            "The AI identified image patterns that differ from those commonly observed in lower-risk samples. "
            "These findings may warrant additional review by a healthcare professional."
        )
        reviewed_text = (
            "The highlighted regions indicate image areas that most influenced the AI assessment. These highlights "
            "do not confirm disease and should be reviewed alongside clinical evaluation."
        )
        next_steps = [
            "Consult a healthcare professional.",
            "Consider additional evaluation if recommended.",
            "Continue monitoring according to medical advice.",
        ]
        assessment = "Pattern Requiring Further Evaluation"
    else:
        observed_text = (
            "The AI identified image patterns that are more consistent with lower-risk samples seen during model "
            "training. Continued routine monitoring is recommended."
        )
        reviewed_text = (
            "The highlighted regions indicate image areas that most influenced the AI assessment. These highlights "
            "do not confirm disease and should be reviewed alongside clinical evaluation."
        )
        next_steps = [
            "Continue routine screening.",
            "Monitor for changes over time.",
            "Seek medical advice if symptoms develop.",
        ]
        assessment = "No High-Risk Pattern Identified"

    return {
        "assessment": assessment,
        "observed_text": observed_text,
        "reviewed_text": reviewed_text,
        "next_steps": next_steps,
    }


def build_technical_appendix(probability_breakdown, risk_category):
    return {
        "feature_extraction": "EfficientNet-B0",
        "classification": "Random Forest",
        "visual_explanation": "Grad-CAM",
        "risk_summary": (
            f"Malignant probability is mapped to the concern level shown in the report. Current level of concern: "
            f"{risk_category}."
        ),
        "image_review_note": (
            "Highlighted regions show where the AI focused while reviewing the image."
        ),
    }


def build_cam_extractor():
    target_layer = select_gradcam_target_layer(attention_model)
    print(f"[GRADCAM] Selected target layer: {target_layer.__class__.__name__}")
    try:
        return SmoothGradCAMpp(attention_model, target_layer=target_layer)
    except Exception as error:
        print("[GRADCAM] SmoothGradCAM++ initialization failed, falling back to GradCAM")
        print(error)
        traceback.print_exc()
        return GradCAM(attention_model, target_layer=target_layer)


def generate_gradcam_overlay(img_path, output_stem):
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor: torch.Tensor = preprocess(pil_img)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    extractor = build_cam_extractor()

    try:
        print("[GRADCAM] Input shape:", input_tensor.shape)
        attention_model.zero_grad(set_to_none=True)
        scores = attention_model(input_tensor)
        print("[GRADCAM] Scores shape:", scores.shape)
        class_idx = int(scores.argmax(dim=1).item())
        print("[GRADCAM] Class index:", class_idx)
        cam_output = extractor(class_idx=class_idx, scores=scores)
        if isinstance(cam_output, (list, tuple)):
            cam = cam_output[0]
        else:
            cam = cam_output
        if isinstance(cam, (list, tuple)):
            cam = cam[0]
        if hasattr(cam, "detach"):
            cam = cam.detach().cpu().numpy()
        else:
            cam = np.asarray(cam)
        print("[GRADCAM] CAM shape:", cam.shape)
        if cam.size == 0:
            raise ValueError("Grad-CAM returned an empty map")
        cam = np.squeeze(cam)
        if cam.ndim != 2:
            raise ValueError(f"Grad-CAM map must be 2D after squeeze, got shape {cam.shape}")
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        else:
            raise ValueError("Grad-CAM map could not be normalized because max value is 0")

        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, pil_img.size)
        original = np.array(pil_img)
        if heatmap.shape[:2] != original.shape[:2]:
            raise ValueError(
                f"Heatmap size mismatch: heatmap {heatmap.shape[:2]} vs original {original.shape[:2]}"
            )
        overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)
        if overlay.size == 0:
            raise ValueError("Generated Grad-CAM overlay is empty")

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


def generate_pdf_report(result, patient_explanation, technical_appendix, original_image_path, gradcam_path):
    report_filename = f"oncoai_report_{result['analysis_token']}.pdf"
    pdf_path = REPORTS_FOLDER / report_filename

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="OncoTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0d3b66"),
        fontSize=21,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="OncoHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0d3b66"),
        fontSize=12.5,
        leading=14.5,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="OncoBody",
        parent=styles["BodyText"],
        fontSize=8.8,
        leading=11.2,
        textColor=colors.HexColor("#24344d"),
    ))
    styles.add(ParagraphStyle(
        name="OncoSmall",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#5f6f86"),
    ))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=(8.27 * inch, 11.69 * inch),
        leftMargin=0.40 * inch,
        rightMargin=0.40 * inch,
        topMargin=0.34 * inch,
        bottomMargin=0.30 * inch,
    )

    story = []
    story.append(Paragraph("ONCOAI SCREENING REPORT", styles["OncoTitle"]))
    meta_table = Table(
        [[
            Paragraph(f"Date & Time: {result['analysis_date']}", styles["OncoSmall"]),
            Paragraph(f"Analysis ID: {result['analysis_token']}", styles["OncoSmall"]),
        ]],
        colWidths=[3.35 * inch, 3.35 * inch],
    )
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.0, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.08 * inch))

    screening_result_text = (
        "Pattern Requiring Further Evaluation" if result["prediction"] == "malignant"
        else "No High-Risk Pattern Identified"
    )

    summary_table = Table(
        [
            ["Screening Result", screening_result_text],
            ["Level of Concern", risk_badge_text(result["risk_badge_class"])],
        ],
        colWidths=[2.20 * inch, 4.50 * inch],
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
    story.append(Spacer(1, 0.10 * inch))

    story.append(Paragraph("Image Review", styles["OncoHeading"]))
    story.append(Paragraph(
        "The highlighted regions indicate image areas that most influenced the AI assessment. These highlights do not confirm disease and should be reviewed alongside clinical evaluation.",
        styles["OncoBody"],
    ))
    story.append(Spacer(1, 0.04 * inch))

    original_block = [Paragraph("Original Image", styles["OncoSmall"]), Spacer(1, 0.03 * inch),
                      pdf_image(original_image_path, 2.95 * inch, 2.15 * inch)]
    if gradcam_path and Path(gradcam_path).exists():
        gradcam_block = [Paragraph("AI Highlighted Image", styles["OncoSmall"]), Spacer(1, 0.03 * inch),
                         pdf_image(gradcam_path, 2.95 * inch, 2.15 * inch)]
    else:
        gradcam_block = [Paragraph("AI Highlighted Image", styles["OncoSmall"]), Spacer(1, 0.03 * inch),
                         Paragraph("Attention analysis could not be generated for this image.", styles["OncoBody"])]

    visuals = Table([[original_block, gradcam_block]], colWidths=[3.12 * inch, 3.12 * inch], hAlign="CENTER")
    visuals.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#c9d9ea")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe6f1")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(visuals)
    story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph("What the AI Observed", styles["OncoHeading"]))
    story.append(Paragraph(patient_explanation["observed_text"], styles["OncoBody"]))
    story.append(Spacer(1, 0.04 * inch))

    story.append(Paragraph("Recommended Actions", styles["OncoHeading"]))
    for item in patient_explanation["next_steps"]:
        story.append(Paragraph(f"- {item}", styles["OncoBody"]))
    story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph("Important Notice", styles["OncoHeading"]))
    story.append(Paragraph(
        "This report is generated by an AI-assisted image screening system and is intended for educational and research purposes. It does not provide a medical diagnosis. Clinical decisions should always be made by qualified healthcare professionals.",
        styles["OncoBody"],
    ))
    story.append(Spacer(1, 0.05 * inch))

    footer_table = Table(
        [[
            Paragraph(
                f"<b>AI System Information</b><br/>Feature Extraction: {technical_appendix['feature_extraction']}<br/>Classification: {technical_appendix['classification']}<br/>Visual Explanation: {technical_appendix['visual_explanation']}",
                styles["OncoSmall"],
            ),
        ]],
        colWidths=[6.55 * inch],
    )
    footer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f9fd")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d7e3ef")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(footer_table)

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

            pred_label, malignancy_probability, probability_breakdown = run_inference(filepath)
            pred_label = pred_label.lower()
            is_malignant = pred_label == "malignant"
            risk_category, risk_badge_class = calculate_risk_category(round(malignancy_probability * 100, 2))
            patient_explanation = build_patient_explanation(pred_label)
            technical_appendix = build_technical_appendix(
                probability_breakdown,
                risk_category,
            )
            analysis_timestamp = datetime.now().strftime("%B %d, %Y %I:%M %p")
            analysis_token = uuid.uuid4().hex

            gradcam_filepath = None
            gradcam_filename = None
            try:
                gradcam_filepath, gradcam_filename = generate_gradcam_overlay(filepath, analysis_token)
            except Exception as error:
                print("\nGRADCAM ERROR:")
                print(error)
                traceback.print_exc()
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
                "probability_breakdown": probability_breakdown,
                "patient_explanation": patient_explanation,
                "technical_appendix": technical_appendix,
                "model_type": "RF on EfficientNet-B0 features",
                "accuracy": "-",
            }
            result["explanation"] = patient_explanation
            result["technical_appendix"] = technical_appendix
            analysis = build_analysis(pred_label)

            try:
                report_filename = generate_pdf_report(result, patient_explanation, technical_appendix, filepath, gradcam_filepath)
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
