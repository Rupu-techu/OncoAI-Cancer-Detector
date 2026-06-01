# 🧠 OncoAI – Cancer Detection & Medical Insight System

> AI-powered web application for early-stage cancer detection using medical images, enhanced with intelligent reporting and a modern doctor-style dashboard.

---

## 🌐 Overview

OncoAI is a machine learning-based web application designed to assist in early breast-cancer detection using medical image analysis. It combines deep learning feature extraction with classical ML techniques to deliver accurate predictions along with user-friendly medical insights.

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

###  Dashboard

<img width="1919" height="926" alt="Screenshot 2026-04-04 010113" src="https://github.com/user-attachments/assets/9fd38bfe-8d50-4cef-8b75-5b2c46f0c4a6" />



---

###  Report Page

<img width="1919" height="930" alt="Screenshot 2026-04-04 010134" src="https://github.com/user-attachments/assets/4e240642-19a3-4d59-b947-b7afb2e86379" />


---
<img width="1915" height="930" alt="Screenshot 2026-04-04 003915" src="https://github.com/user-attachments/assets/49644a7b-a7ad-4104-9d00-fe2e80be2b72" />


--- 
###  About Page

<img width="1919" height="927" alt="Screenshot 2026-04-04 004155" src="https://github.com/user-attachments/assets/06b3118d-d526-4d68-b976-60a7eb9442ae" />

---

## 🎥 Demo Video






https://github.com/user-attachments/assets/ef9e4bc7-5d38-462b-8a4a-6c6f877f3fe9



---

## 🏗️ Project Architecture

---

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

AI Screening Result + Malignancy Probability

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

```
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


```

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


## Important Note

Dataset and trained model files are not included due to size limitations.

---

🧠 Future Enhancements

🧬 Multi-cancer detection (lung, skin, brain)

📊 Severity prediction (Stage classification)

🤖 AI-powered medical recommendation system

☁️ Cloud deployment (AWS / Render)

👩‍⚕️ Use Case

This system can assist:

Medical students,
Researchers,
Healthcare AI prototypes,
Early-stage screening tools

---

 ## Acknowledgement

Built with passion to contribute towards AI in healthcare.
