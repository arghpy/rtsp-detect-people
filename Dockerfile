# Start from the official Ultralytics image
FROM ultralytics/ultralytics:8.3.95-nvidia-cuda

# If there wouldn't be any detection
# FROM python:3.11-slim

# Install ffmpeg (Debian-based image)
RUN apt-get update\
      && apt-get install -y ffmpeg --no-install-recommends\
      && rm -rf /var/lib/apt/lists/*

# If there wouldn't be any detection
# COPY requirements.txt .
# RUN pip install -r requirements.txt
