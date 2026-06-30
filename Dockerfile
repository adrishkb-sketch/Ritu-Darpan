# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed for shapely/geospatial libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgeos-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY dashboard/ /app/dashboard/
COPY data-pipeline/ /app/data-pipeline/

# Expose Streamlit port
EXPOSE 8501

# Run command
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
