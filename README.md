# 🧠 OncoAI – Cancer Detection & Medical Insight System

> AI-powered web application for cancer detection using medical images, enhanced with intelligent reporting and a modern doctor-style dashboard.

---

## 🌐 Overview

OncoAI is a machine learning-based web application designed to assist in early cancer detection using medical image analysis. It combines deep learning feature extraction with classical ML techniques to deliver accurate predictions along with user-friendly medical insights.

---

## ✨ Features

- 🧬 Cancer Detection (Image-based)
- 📊 Dashboard UI (Doctor-style interface)
- 📄 Automated Medical Report Generation
- ⚡ EfficientNet Feature Extraction
- 📉 PCA + Random Forest Model
- 🧠 Explainability-ready (Grad-CAM integration planned)
- 💡 Clean and interactive UI (glassmorphism + blue theme)

---

## 🖼️ Website Preview

### 🔷 Dashboard
![Dashboard Screenshot](assets/dashboard.png)

---

### 📄 Report Page
![Report Screenshot](assets/report.png)

---

## 🎥 Demo Video

[![Watch the demo](assets/demo-thumbnail.png)](assets/demo-video.mp4)
User Upload Image
↓
Preprocessing (Resize, Normalize)
↓
EfficientNet (Feature Extraction)
↓
PCA (Dimensionality Reduction)
↓
Random Forest Classifier
↓
Prediction + Confidence
↓
Report Generation (UI)


---

## ⚙️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **ML Models:**  
  - EfficientNet-B0  
  - PCA  
  - Random Forest  
- **Libraries:** OpenCV, NumPy, scikit-learn, PyTorch  

---

## 📁 Project Structure


OncoAI-Cancer-Detector/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── templates/
│ ├── dashboard.html
│ └── report.html
│
├── static/
│ ├── css/
│ └── js/
│
├── scripts/
├── notebook/


---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Rupu-techu/OncoAI-Cancer-Detector.git

# Navigate into project
cd OncoAI-Cancer-Detector

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

```

## 🏗️ Project Architecture
 Important Note

Dataset and trained model files are not included due to size limitations.

🧠 Future Enhancements
🔥 Grad-CAM visualization (Explainable AI)
🧬 Multi-cancer detection (lung, skin, brain)
📊 Severity prediction (Stage classification)
🤖 AI-powered medical recommendation system
☁️ Cloud deployment (AWS / Render)
👩‍⚕️ Use Case

This system can assist:

Medical students
Researchers
Healthcare AI prototypes
Early-stage screening tools
💙 Acknowledgement

Built with passion to contribute towards AI in healthcare.
