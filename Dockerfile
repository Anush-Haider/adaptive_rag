# Use an official lightweight Python runtime
FROM python:3.11-slim

# Set system environment optimizations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface

# Set working directory inside the container
WORKDIR /app

# Install essential system utilities needed for building dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker's caching layer
COPY requirements.txt .

# Install dependencies directly to system Python (no venv needed inside isolation)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application codebase
COPY . .

# Expose port 8000 in case we want to attach an API layer later
EXPOSE 8000

# Execute our main graph runner
CMD ["python", "main.py"]