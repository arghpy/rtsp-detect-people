# Start from the official Ultralytics image
FROM ultralytics/ultralytics:8.3.95-nvidia-cuda

# Install ffmpeg (Debian-based image)
RUN apt-get update\
      && apt-get install -y ffmpeg --no-install-recommends\
      && rm -rf /var/lib/apt/lists/* \
      && pip install --no-cache-dir flask
