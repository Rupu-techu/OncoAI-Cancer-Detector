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

<img width="1919" height="1079" alt="Screenshot 2026-06-03 090534" src="https://github.com/user-attachments/assets/064b8004-a8cb-44e8-bd10-9ff58d2004f8" />


---

###  Report Page

<img width="1919" height="1079" alt="Screenshot 2026-06-03 092058" src="https://github.com/user-attachments/assets/a2ae79a5-dfaf-4be0-8e3b-b3c682f2a09d" />



---
<img width="1919" height="1079" alt="Screenshot 2026-06-03 091818" src="https://github.com/user-attachments/assets/9aac77b2-029b-411e-a9d8-8e5046411613" />

---
<img width="1918" height="1079" alt="Screenshot 2026-06-03 091842" src="https://github.com/user-attachments/assets/c7ef83bd-91c5-4e9f-ae08-af6772e59372" />

--- 
###  About Page

<img width="1919" height="1079" alt="Screenshot 2026-06-03 090534" src="https://github.com/user-attachments/assets/21d2d267-e54c-45cd-a7e0-18394851ce8a" />

---
###  PDF report
<img width="1919" height="1079" alt="Screenshot 2026-06-03 091227" src="https://github.com/user-attachments/assets/7376ca50-2bf2-44be-a179-88c34c3756f9" />

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
