# # Use Python 3.10 slim image
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*
    
# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Create necessary directories
# RUN mkdir -p data/raw data/processed data/embeddings logs models

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV FLASK_APP=src/api/app.py

# # Expose port
# EXPOSE 5000

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD python -c "import requests; requests.get('http://localhost:5000/api/health')"

    

# # Run the application
# CMD ["python", "src/api/app.py"]


# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/embeddings logs models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app        
ENV FLASK_APP=src/api/app.py

# Expose port
EXPOSE 5000

# Health check (optional, can remove for local testing)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health')" || exit 1

# Run the application as a module (best practice)
CMD ["python", "-m", "src.api.app"]

