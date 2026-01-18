# OncoAI-Cancer-Detector
 CANCER PREDICTION WEB APPLICATION (Extensible ML System)

This project is a **Machine Learning–based Cancer Prediction System** with a **web interface** built using **Flask**.

**Currently implemented for Breast Cancer prediction**  
**Designed to be extended to other cancer types** such as lung, skin etc.

The system focuses on building a **scalable ML pipeline** that can support multiple cancer datasets and prediction models in the future.



## Project Objective

The objective of this project is to:
- Build an end-to-end **machine learning pipeline**
- Develop a **web-based prediction interface**
- Create a **modular and extensible architecture** for cancer prediction
- Demonstrate real-world ML deployment skills


##  Key Features

- ✔ Machine Learning–based cancer prediction
- ✔ Modular design (easy to add new cancer types)
- ✔ Feature scaling and dimensionality reduction (PCA)
- ✔ Flask-based web application
- ✔ User-friendly interface for predictions
- ✔ Clean and professional project structure

---

##  Current Implementation

### Supported Cancer Type (Current)
- **Breast Cancer**
  - Based on structured clinical / extracted features
  - Supervised classification model

###  Planned Extensions
- Lung Cancer
- Skin Cancer
- Multi-class cancer classification

The architecture allows new datasets and models to be added with minimal changes.

---

##  Project Structure

cancer-prediction-project/
│
├── app.py # Flask web application
├── train_model.py # Model training script
├── save_dataset.py # Dataset preprocessing
├── organize_images.py # Image organization utility
├── requirements.txt # Required Python libraries
├── README.md # Project documentation
├── .gitignore
│
├── notebook/ # Jupyter notebooks (experiments)
├── scripts/ # Helper scripts
├── templates/ # HTML files (Flask frontend)
├── static/ # CSS / static assets


## Dataset Information

Due to **GitHub file size limits**, datasets and trained model files are **not included** in this repository.

### Datasets Used (Current)
- **Breast Cancer Wisconsin Dataset**
- **BreaKHis Histopathological Image Dataset**

Datasets can be obtained from:
- https://www.kaggle.com/

---

## Note on Large Files

The following files/folders are intentionally excluded:
- `data/`
- `.npy` files
- `.pkl` trained model files

These files are generated locally during training and are not required to understand the project structure.

## Model Training (Optional)

To preprocess data and train the model locally:

```bash
python save_dataset.py
python train_model.py
