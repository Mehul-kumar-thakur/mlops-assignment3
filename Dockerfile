# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
WORKDIR /app/src

# Default command
CMD ["python", "predict.py"]