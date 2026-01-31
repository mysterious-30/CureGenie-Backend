FROM python:3.11-slim

# Install system dependencies for barcode scanning
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libzbar-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Increased timeout for heavy packages like tensorflow
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy application code (including human_main.h5)
COPY . .

# Expose port
EXPOSE 10000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
