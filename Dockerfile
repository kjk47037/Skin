# Use Python slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install CPU-only PyTorch first (much smaller than CUDA version ~200MB vs ~2GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0+cpu torchvision==0.16.0+cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
# Create models directory (model can be downloaded at runtime via MODEL_DOWNLOAD_URL)
RUN mkdir -p ./models
# Copy models directory if it exists (optional - model can be downloaded at runtime)
# If models/ doesn't exist, this will be skipped during build
COPY models/ ./models/

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

