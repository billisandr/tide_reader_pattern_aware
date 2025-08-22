FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config.yaml .

# Create necessary directories
RUN mkdir -p /app/data/input /app/data/processed /app/data/calibration /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENCV_LOG_LEVEL=ERROR

# Run the application
CMD ["python", "-u", "src/main.py"]