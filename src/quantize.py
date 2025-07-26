import joblib
import torch
import numpy as np

# Load sklearn model
model = joblib.load("model.joblib")
weights = model.coef_
bias = model.intercept_

# Save unquantized
unquant_params = {"weights": weights, "bias": bias}
joblib.dump(unquant_params, "unquant_params.joblib")

# Quantize manually to uint8
scale = 255 / (weights.max() - weights.min())
zero_point = -weights.min() * scale

weights_q = np.round(weights * scale + zero_point).astype(np.uint8)
bias_q = np.round(bias * scale + zero_point).astype(np.uint8)

quant_params = {"weights_q": weights_q, "bias_q": bias_q, "scale": scale, "zero_point": zero_point}
joblib.dump(quant_params, "quant_params.joblib")

# Dequantize
weights_dq = (weights_q.astype(np.float32) - zero_point) / scale
bias_dq = (bias_q.astype(np.float32) - zero_point) / scale

# PyTorch single-layer model
class SimpleModel(torch.nn.Module):
    def _init_(self, weights, bias):
        super()._init_()
        self.linear = torch.nn.Linear(len(weights), 1)
        self.linear.weight.data = torch.tensor([weights], dtype=torch.float32)
        self.linear.bias.data = torch.tensor([bias], dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)

# Load dataset
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

data = fetch_california_housing()
X = data.data
y = data.target

# Torch inference
model = SimpleModel(weights_dq, bias_dq)
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X, dtype=torch.float32)).numpy()

print("R² Score (Quantized):", r2_score(y,  y_pred))