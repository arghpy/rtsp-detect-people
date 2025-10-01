# Start from the official Ultralytics image
FROM ultralytics/ultralytics@sha256:6a2b333adc02a93cce839a5685d76ddae09ed0eead816b796bd69fab3ae39eab

# Install ffmpeg (Debian-based image)
RUN apt-get update\
      && apt-get install -y ffmpeg --no-install-recommends\
      && rm -rf /var/lib/apt/lists/*
