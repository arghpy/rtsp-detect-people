from flask import Flask, send_from_directory

HLS_DIR = "/tmp/hls"

app = Flask(__name__)


@app.route("/")
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


@app.route("/hls/<path:filename>")
def hls_files(filename):
    return send_from_directory(HLS_DIR, filename)


@app.after_request
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
    app.run(
        host="0.0.0.0", port=web_port, threaded=True, debug=False, use_reloader=False
    )
