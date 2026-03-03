# Start from the official Ultralytics image
FROM ultralytics/ultralytics:8.3.95-nvidia-cuda

# Install ffmpeg (Debian-based image)
RUN apt-get update\
      && apt-get install -y ffmpeg --no-install-recommends\
      && rm -rf /var/lib/apt/lists/*

# REMOVE GUI OpenCV
RUN pip uninstall -y opencv-python opencv-python-headless || true

# INSTALL headless OpenCV
RUN pip install --no-cache-dir opencv-python-headless

# Install your extras
RUN pip install --no-cache-dir flask
