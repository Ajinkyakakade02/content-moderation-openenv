FROM python:3.9-slim

WORKDIR /app

# Install minimal system dependencies (skip heavy graphics libraries)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies with timeout and retry
RUN pip install --no-cache-dir --default-timeout=100 \
    flask==2.3.0 \
    flask-cors==4.0.0 \
    gunicorn==21.2.0 \
    pillow==10.1.0 \
    numpy==1.24.3 \
    pydantic==2.5.0 \
    gymnasium==0.29.1 \
    python-dotenv==1.0.0 \
    scikit-learn==1.3.2 \
    openai==1.0.0

# Copy all application files
COPY . .

# Expose the port
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "backend.api_server:app"]
