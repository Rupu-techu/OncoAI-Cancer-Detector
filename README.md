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

-  Machine Learning–based cancer prediction
-  Modular design (easy to add new cancer types)
-  Feature scaling and dimensionality reduction (PCA)
-  Flask-based web application
-  User-friendly interface for predictions
-  Clean and professional project structure


##  Current Implementation

### Supported Cancer Type 
- **Breast Cancer**
  - Based on structured clinical / extracted features
  - Supervised classification model

###  Planned Extensions
- Lung Cancer
- Skin Cancer
- Multi-class cancer classification

The architecture allows new datasets and models to be added with minimal changes.

##  Project Structure

cancer-prediction-project/
│
├── app.py 
├── train_model.py 
├── save_dataset.py 
├── organize_images.py
├── requirements.txt 
├── README.md 
├── .gitignore
│
├── notebook/ 
├── scripts/ 
├── templates/
├── static/


## Dataset Information

Due to **GitHub file size limits**, datasets and trained model files are **not included** in this repository.

### Datasets Used 
- **Breast Cancer Wisconsin Dataset**
- **BreaKHis Histopathological Image Dataset**

Datasets can be obtained from:
- https://www.kaggle.com/

## Note on Large Files

The following files/folders are intentionally excluded:
- `data/`
- `.npy` files
- `.pkl` trained model files

These files are generated locally during training and are not required to understand the project structure.
## Demo Video 

A full working demonstration of the web application is available in the video below:
https://drive.google.com/drive/u/0/folders/1ZmCdam4Bv4xaIsBR_XBkG1LW2_BRVmv8

## Running and Using the Web Application

This project includes a Flask-based web interface that can be executed locally using **Visual Studio Code**.  
The application runs on a local server and can be accessed through a web browser.

### Execution Environment
- The application is developed and tested using **Visual Studio Code**
- Python is used as the backend runtime
- Flask is used to serve the web interface

### How the Website is Accessed

1. The project folder is opened in **Visual Studio Code**
2. Required Python dependencies are installed using the provided `requirements.txt`
3. The Flask application is started by running `app.py`
4. Once the server starts, the website becomes available on a local URL
5. Users can open the URL in any web browser and interact with the application

### Local Access URL

When the application is running, it is accessible at:
http://127.0.0.1:5000

## Model Training 
To preprocess data and train the model locally:
```bash
python save_dataset.py
python train_model.py
