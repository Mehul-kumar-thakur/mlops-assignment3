# Base image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Train the model during image build
RUN python src/train.py

CMD ["python", "src/predict.py"]
