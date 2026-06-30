# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY src/ src/
COPY config/ config/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "depression_companion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]