from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
import joblib

# Load test data
data = fetch_california_housing()
X, y = data.data, data.target

# Load model
model = joblib.load("model.joblib")

# Predict
y_pred = model.predict(X)
print("R² Score (Predict.py):", r2_score(y, y_pred))