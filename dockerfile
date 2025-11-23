# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by OpenCV and unstructured
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    ghostscript \
    fonts-dejavu-core \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*



# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir "numpy<2.0"
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app
ENV PATH="/usr/bin:/usr/local/bin:${PATH}"

# Default command
CMD ["python", "app.py"]docker compose build --no-cache
