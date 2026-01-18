import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("Starting optimized model training...\n")
start_time = time.time()


print("Loading dataset...")
X = np.load("X_images.npy")
y = np.load("y_labels.npy")

print("Original X shape:", X.shape)
print("y shape:", y.shape)


X = X.reshape(X.shape[0], -1)
print("Flattened X shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


print("\nScaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



print("\nApplying PCA for dimensionality reduction...")
pca = PCA(n_components=200, random_state=42)

pca_start = time.time()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
pca_end = time.time()

print("PCA completed!")
print("Reduced feature size:", X_train.shape[1])
print(f"PCA time: {pca_end - pca_start:.2f} seconds")


print("\nTraining Random Forest model...")
rf_start = time.time()

model = RandomForestClassifier(
    n_estimators=300,          
    max_depth=None,
    class_weight="balanced",   
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

rf_end = time.time()
print("Model training completed!")
print(f"Training time: {rf_end - rf_start:.2f} seconds")


print("\nEvaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n================ RESULTS ================")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
import joblib

# Save trained objects
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("Model, scaler, and PCA saved successfully!")



