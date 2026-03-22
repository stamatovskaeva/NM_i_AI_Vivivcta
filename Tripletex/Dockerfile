# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY .env.example .env

# Expose port (Cloud Run uses 8080)
EXPOSE 8080

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
