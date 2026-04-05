FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Hugging Face requires port 7860
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "backend.api_server:app"]