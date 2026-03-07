import os
import subprocess

from flask import Flask, send_from_directory
import app.yolo.detection

application = Flask(__name__)

HLS_DIR = "/tmp/hls"


@application.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>RTSP HLS Stream</title>
  <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
</head>
<body style="margin:0; background:black">
  <video id="video" controls autoplay muted playsinline width="100%"></video>

  <script>
    const video = document.getElementById('video');
    const src = '/hls/stream.m3u8';

    if (Hls.isSupported()) {
      const hls = new Hls({
        liveSyncDuration: 0.2,   // only keep ~0.2s behind live
        lowLatencyMode: true,
        backBufferLength: 30
      });
      hls.loadSource(src);
      hls.attachMedia(video);
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = src; // Safari
    }
  </script>
</body>
</html>
"""


@application.route("/hls/<path:filename>")
def hls_files(filename):
    return send_from_directory(HLS_DIR, filename)


@application.after_request
def disable_hls_cache(response):
    if response.mimetype in (
        "application/vnd.apple.mpegurl",
        "video/mp2t",
    ):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def start_web_server(web_port):
    """Run Flask app on separate thread"""
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    application.run(
        host="0.0.0.0", port=web_port, threaded=True, debug=False, use_reloader=False
    )


def hls_writer(output_dir, width, height, fps):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]

    if app.yolo.detection.CUDA_ENABLED:
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "llhp"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency"])

    cmd.extend(
        [
            "-g",
            str(int(fps * 0.5)),
            "-keyint_min",
            str(int(fps * 0.5)),
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "hls",
            "-hls_time",
            "0.5",
            "-hls_list_size",
            "2",
            "-hls_flags",
            "delete_segments+append_list+independent_segments",
            "-hls_allow_cache",
            "0",
            os.path.join(output_dir, "stream.m3u8"),
        ]
    )

    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
