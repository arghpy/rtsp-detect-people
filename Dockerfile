# Start from the official Ultralytics image
FROM ultralytics/ultralytics:latest

# Install ffmpeg (Debian-based image)
RUN apt-get update\
      && apt-get install -y ffmpeg\
      && rm -rf /var/lib/apt/lists/*
