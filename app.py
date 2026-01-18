import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # ================= IMAGE PREPROCESSING =================
            img = cv2.imread(filepath)

            if img is None:
                raise ValueError("Image could not be read")

            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
            img = cv2.resize(img, (128, 128))  

            
            img = img.reshape(1, -1)

            print("Feature count:", img.shape[1])  

            
            img = scaler.transform(img)
            img = pca.transform(img)
            probs = model.predict_proba(img)[0]
            print("Probabilities [Class0, Class1]:", probs)


           
        probs = model.predict_proba(img)[0]

        cancer_prob = probs[1]   
        non_cancer_prob = probs[0]

        print("Cancer probability:", cancer_prob)


        if cancer_prob >= 0.5:
            prediction = "Cancerous"
        else:
            prediction = "Non-Cancerous"

        confidence = round(
            cancer_prob * 100 if prediction == "Cancerous"
            else non_cancer_prob * 100,
            2
        )


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
