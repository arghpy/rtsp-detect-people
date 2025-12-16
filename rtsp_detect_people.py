#!/usr/bin/env python3
"""Detect people from an RTSP stream"""
# pylint: disable=c-extension-no-member

import json
import os
import queue
import select
import signal
import smtplib
import subprocess
import sys
import threading
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage

import cv2
import numpy as np
from flask import Flask, send_from_directory

# pylint: disable=import-error
from ultralytics import YOLO

# Load YOLOv8n model (it will auto-download if missing)
model = YOLO("yolov8n.pt")
CUDA_ENABLED = False
try:
    model.to("cuda")        # Move model to GPU
    CUDA_ENABLED = True
except Exception as e:
    print(f"[ERROR] Failed to initialize YOLO model with nvidia: {e}", file=sys.stderr)
    print("Continuing with cpu detection.", file=sys.stderr)

# Font settings
FONT_NAME = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Args
SHOW_DISPLAY = False
SAVE_VIDEO = False
SEND_EMAIL = False
CONFIGURATION_FILE = None
ENABLE_WEB = False
ENABLE_DETECTION = False
WEB_PORT = None

# Other globals
MAX_FRAME_DROPS = 5
BOX_COLOR = (0, 255, 0)  # (B, G, R) colors - Green
CONFIDENCE_MIN = None
HLS_DIR = "/tmp/hls"
HLS_WRITER = None

# --- WEB SERVER GLOBALS ---
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
  <video id="video" autoplay muted playsinline width="100%"></video>

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


# --- END WEB SERVER INTEGRATION ---

def hls_writer(output_dir, width, height, fps):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
    ]

    if CUDA_ENABLED:
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "llhp"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency"])

    cmd.extend([
        "-pix_fmt", "yuv420p",
        "-f", "hls",
        "-hls_time", "1",
        "-hls_list_size", "3",
        "-hls_allow_cache", "0",
        "-hls_flags", "delete_segments+append_list",
        "-hls_playlist_type", "event",
        os.path.join(output_dir, "stream.m3u8"),
    ])

    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# pylint: disable=unused-argument
def handle_signals(signum, exec_frame):
    """Respond to different signals"""
    global STOP_EVENT

    signame = signal.Signals(signum).name
    pprint(f"Received {signame}({signum})")

    # Release and close threading
    executor.shutdown(wait=True)

    # Stop reader
    STOP_EVENT.set()
    stream_reader_thread.join(timeout=2)
    web_thread.join(timeout=2)

    # Stop writer
    if SAVE_VIDEO:
        OUT_VIDEO_WRITER.stdin.close()
        OUT_VIDEO_WRITER.wait()

    # Destroy window if display was set
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()

    shutil.rmtree(HLS_DIR, ignore_errors=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_signals)
signal.signal(signal.SIGINT, handle_signals)


def eprint(s):
    """Print to stderr with current time"""
    print(f"{datetime.now()}: {s}", file=sys.stderr, flush=True)


def pprint(s):
    """Print to stdout with current time"""
    print(f"{datetime.now()}: {s}", file=sys.stdout, flush=True)


def send_email_report(save_image_path, save_image_type, config):
    """Send email based on the environment variables"""
    pprint("Person detected. Sending email")

    # Create the container email message.
    msg = EmailMessage()
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S")
    msg["Subject"] = config["email"]["subject"] + f": {current_time}"
    msg["From"] = config["email"]["user"]
    msg["To"] = ", ".join(config["email"]["recipients"])

    # Open the image in binary mode
    with open(save_image_path, "rb") as fp:
        img_data = fp.read()
        msg.add_attachment(
            img_data,
            maintype="image",
            subtype=save_image_type,
            filename=save_image_path,
        )

    with smtplib.SMTP_SSL(config["email"]["server"], config["email"]["port"]) as s:
        s.login(config["email"]["user"], config["email"]["password"])
        s.send_message(msg)


def writer_stream(video_path, width, height, fps) -> subprocess.Popen:
    """Write stream to file"""
    pprint(f"Saving video to {video_path}")

    writer_cmd = [
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
        f"{fps}",
        "-i",
        "-",
        "-an",  # no audio
    ]
    if CUDA_ENABLED:
        writer_cmd.extend(["-c:v", "h264_nvenc", "-preset", "llhp"])
    else:
        writer_cmd.extend(["-c:v", "libx264", "-preset", "veryfast"])

    writer_cmd.extend([
        "-g",
        f"{fps*2}",  # keyframe every 2 seconds
        "-x264-params",
        f"keyint={fps*2}:min-keyint={fps}",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "matroska",
        video_path,
    ])
    # pylint: disable=consider-using-with
    writer = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE)
    return writer


def read_frame(pipe: subprocess.Popen, width, height) -> np.ndarray | None:
    """Read frame from reader"""
    size = width * height * 3

    # Wait until data is available or timeout expires
    rlist, _, _ = select.select([pipe], [], [], 2)  # wait 2 seconds for data
    if not rlist:
        return None  # no data

    raw = pipe.read(size)
    if raw is None or len(raw) != size:
        return None

    return np.frombuffer(raw, np.uint8).reshape((height, width, 3))


def reader_stream(rtsp_url, fps) -> subprocess.Popen:
    """Continuously get frames from stream"""
    pprint("Starting ffmpeg reader")

    reader_cmd = [
        "ffmpeg",
    ]
    if CUDA_ENABLED:
        reader_cmd.extend(["-hwaccel", "cuda"])
    reader_cmd.extend([
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-loglevel",
        "error",
        "-vf", f"fps={fps}",
        "-an",
        "-sn",  # disable audio and subs
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ])

    # pylint: disable=consider-using-with
    reader = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE)
    return reader


def terminate_pipe_process(pipe: subprocess.Popen):
    """Safely terminate the pipe"""
    wait_timeout = 5
    pipe.stdout.close()
    pipe.terminate()

    try:
        pipe.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        eprint(f"Waited for {wait_timeout}. Killing process")
        pipe.kill()


def reconnect_pipe_process(pipe: subprocess.Popen, rtsp_url, fps):
    """Safely reconnect to stream, retry until ffmpeg is alive."""
    terminate_pipe_process(pipe)

    attempt = 0
    while True:
        attempt += 1
        eprint(f"Reconnecting attempt #{attempt}...")

        new_pipe = reader_stream(rtsp_url, fps)
        time.sleep(1.0)  # give ffmpeg a moment to start

        if new_pipe and new_pipe.poll() is None and new_pipe.stdout:
            pprint("Successfully reconnected")
            return new_pipe

        eprint("ffmpeg failed to start or exited immediately")
        terminate_pipe_process(new_pipe)
        time.sleep(2)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def reader_frames_thread(frame_queue, width, height, fps, rtsp_url, stop_event):
    """Continuously add frames in queue to be processed"""
    pprint("Reader thread started")

    pipe = reader_stream(rtsp_url, fps)
    if pipe is None or pipe.returncode is not None:
        stop_event.set()

    dropped_frames = 0

    while not stop_event.is_set():
        frame = None
        try:
            frame = read_frame(pipe.stdout, width, height)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            eprint(f"Exception reading frame: {e}")
            dropped_frames += 1
            continue

        if frame is None:
            dropped_frames += 1

            if dropped_frames >= fps * 2:
                eprint(f"{dropped_frames} consecutive frames missing. Reconnecting")
                pipe = reconnect_pipe_process(pipe, rtsp_url, fps)
                dropped_frames = 0
        else:
            dropped_frames = 0

            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                # Queue full → drop frame to avoid blocking
                pass

    terminate_pipe_process(pipe)


def probe_stream(rtsp_url) -> tuple[int, int, int]:
    """Probe the stream to get data"""
    pprint("Probing stream info")

    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=s=x:p=0",
        rtsp_url,
    ]

    while True:
        probe = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        probe_stdout = probe.stdout.strip()

        if probe.returncode != 0 or not probe_stdout:
            eprint("Failed to probe RTSP stream info")
            time.sleep(0.1)
            continue

        parts = probe_stdout.split("x")
        if len(parts) != 3:
            eprint(f"Unexpected ffprobe output: {probe.stdout.strip()}")
            continue
        break

    width = int(parts[0])
    height = int(parts[1])
    fps_str = parts[2]

    # Convert fps string like "25/1" or "30000/1001" to float
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = int(float(num) / float(den))
    else:
        fps = int(fps_str)

    pprint(f"Stream resolution: {width}x{height}, FPS: {fps:.2f}")
    return width, height, fps


def process_frame(frame):
    """Process frame with yolo model"""
    person_detected = False

    # half=True - Enable FP16 for faster inference
    results = model(frame, conf=CONFIDENCE_MIN, verbose=False, half=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            if model.names[cls] == "person":
                person_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Person: {confidence*100:.2f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
    return frame, person_detected


def load_json_file(file):
    """Load json file"""
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            content.strip()  # Remove whitespaces
            json_content = json.loads(content)
    except json.JSONDecodeError:
        eprint(f"File is not in JSON format: {file}")
        sys.exit(1)
    except FileNotFoundError:
        eprint(f"File does not exist: {file}")
        sys.exit(0)
    return json_content


def usage_description(description):
    """Formatting for description"""
    s = f"\n\nDESCRIPTION" f"\n\t{description}"
    return s


def usage_header(header):
    """Formatting for header"""
    s = f"\n\n{header.upper()}"
    return s


def add_option(option, description):
    """Add formatted option in description"""
    s = f"\n\n{option}," f"\n\t{description}"
    return s


def usage(argv):
    """Print program usage"""
    options = (
        ("-c/--config FILE", "specify configuration file", True),
        ("-h/--help", "print this help message", False),
        ("-d/--display", "view footage live", False),
        ("-s/--save", "save live footage", False),
        ("-e/--email", "send email", False),
        ("-w/--web PORT", "Start web server on port", False),
        ("--detect", "detect people", False),
    )

    str_options = " ".join(
        [f"{opt[0]}" if opt[2] is True else f"[{opt[0]}]" for opt in options]
    )
    program_usage = f"{argv[0]} {str_options}"
    program_description = usage_description("Detect people from RTSP stream.")
    program_options = "".join([add_option(opt[0], opt[1]) for opt in options])

    program_help = (
        program_usage,
        program_description,
        usage_header("options"),
        program_options,
    )
    print("".join(program_help))


def parse_arguments(argv):
    """Parse command line arguments"""
    # pylint: disable=global-statement
    global SHOW_DISPLAY, SAVE_VIDEO, SEND_EMAIL, CONFIGURATION_FILE, WEB_PORT, ENABLE_WEB, ENABLE_DETECTION

    passed_args = argv[1:]

    while len(passed_args) > 0:
        if passed_args[0] == "-h" or passed_args[0] == "--help":
            usage(argv)
            sys.exit(0)
        elif passed_args[0] == "-d" or passed_args[0] == "--display":
            SHOW_DISPLAY = True
        elif passed_args[0] == "-s" or passed_args[0] == "--save":
            SAVE_VIDEO = True
        elif passed_args[0] == "-e" or passed_args[0] == "--email":
            SEND_EMAIL = True
        elif passed_args[0] == "-c" or passed_args[0] == "--config":
            passed_args.pop(0)
            CONFIGURATION_FILE = passed_args[0]
        elif passed_args[0] == "-w" or passed_args[0] == "--web":
            ENABLE_WEB = True
            passed_args.pop(0)
            WEB_PORT = passed_args[0]
        elif passed_args[0] == "--detect":
            ENABLE_DETECTION = True
        else:
            eprint(f"Invalid option: {passed_args[0]}")
            usage(argv)
            sys.exit(0)
        passed_args.pop(0)


if __name__ == "__main__":
    parse_arguments(sys.argv)

    # Set up used variables
    now = datetime.now()

    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second

    SAVE_IMAGE_TYPE = "jpeg"
    SAVE_IMAGE_PATH = None
    OUT_VIDEO_WRITER = None
    # pylint: disable=invalid-name
    email_sent = False
    email_future = None
    start_timeout = 0
    STOP_EVENT = threading.Event()

    # Create executor
    executor = ThreadPoolExecutor(max_workers=1)

    if CONFIGURATION_FILE is None:
        eprint("Configuration not specified.")
        usage(sys.argv)
        sys.exit(1)

    configuration = load_json_file(CONFIGURATION_FILE)
    PERSON_DETECTED = False
    TIMEOUT = int(configuration["timeout"])  # Secs
    CONFIDENCE_MIN = float(configuration["confidence"])
    RTSP_USER = configuration["rtsp"]["user"]
    RTSP_PASSWORD = configuration["rtsp"]["password"]
    RTSP_FEED = configuration["rtsp"]["feed"]
    RTSP_URL = f"rtsp://{RTSP_USER}:{RTSP_PASSWORD}@{RTSP_FEED}"

    # Frame and properties
    video_width, video_height, video_fps = probe_stream(RTSP_URL)
    try:
        video_fps = int(configuration["rtsp"]["save_video"]["optional_force_fps"])
        pprint(f"FPS was overridden by the config to {video_fps}")
    except KeyError:
        pass
    FRAME_QUEUE = queue.Queue(maxsize=video_fps * 2)
    stream_reader_thread = threading.Thread(
        target=reader_frames_thread,
        args=(
            FRAME_QUEUE,
            video_width,
            video_height,
            video_fps,
            RTSP_URL,
            STOP_EVENT,
        ),
        daemon=True,
    )
    stream_reader_thread.start()

    if ENABLE_WEB:
        HLS_WRITER = hls_writer(HLS_DIR, video_width, video_height, video_fps)
        web_thread = threading.Thread(
            target=start_web_server, args=(WEB_PORT,), daemon=True
        )
        web_thread.start()

    if SAVE_VIDEO:
        output_video_name = configuration["rtsp"]["save_video"]["name"]
        output_video_path = configuration["rtsp"]["save_video"]["path"]
        output_video_path = (
            f"{output_video_path}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
        )
        output_video_format = "mkv"
        output_video_name = (
            f"{output_video_name}_{year}"
            f"-{month}"
            f"-{day}"
            f"_{hour}"
            f"-{minute}"
            f"-{second}"
            f".{output_video_format}"
        )

        output_video = f"{output_video_path}/{output_video_name}"

        SAVE_IMAGE_PATH = f"{output_video_path}/captures"

        try:
            os.makedirs(SAVE_IMAGE_PATH)
        except FileExistsError:
            pass

        OUT_VIDEO_WRITER = writer_stream(
            output_video, video_width, video_height, video_fps
        )

    # MAIN LOOP
    while True:
        video_frame = FRAME_QUEUE.get(block=True)  # Wait until a frame is available

        # Check for corrupt frame
        if (
            video_frame.size == 0
            or video_frame.shape[0] < 50
            or video_frame.shape[1] < 50
        ):
            eprint("[WARN] Corrupt frame detected, skipping...")
            continue

        # Run model on frame
        if ENABLE_DETECTION:
            video_frame, PERSON_DETECTED = process_frame(video_frame)

        # Send email
        if SEND_EMAIL:
            if email_sent and email_future is not None:
                if email_future.done():
                    pprint("Email sent")
                    email_sent = False
                    email_future = None

        if PERSON_DETECTED and (time.time() - start_timeout) > TIMEOUT:
            now = datetime.now()
            minute = now.minute
            second = now.second

            SAVE_IMAGE_PATH = f"{output_video_path}/captures"
            SAVE_IMAGE_NAME = configuration["rtsp"]["save_video"]["name"]
            SAVE_IMAGE_NAME = (
                f"{SAVE_IMAGE_NAME}" f"_{minute}" f":{second}" f".{SAVE_IMAGE_TYPE}"
            )
            SAVE_IMAGE = f"{SAVE_IMAGE_PATH}/{SAVE_IMAGE_NAME}"
            rc = cv2.imwrite(SAVE_IMAGE, video_frame)
            if rc:
                pprint(f"Saved image to {SAVE_IMAGE}")
            else:
                eprint(f"Failed to save image to {SAVE_IMAGE}")

            if SEND_EMAIL:
                email_future = executor.submit(
                    send_email_report,
                    SAVE_IMAGE,
                    SAVE_IMAGE_TYPE,
                    configuration,
                )
                email_sent = True

            start_timeout = time.time()

        if ENABLE_WEB:
            HLS_WRITER.stdin.write(video_frame.tobytes())

        # Show display
        if SHOW_DISPLAY:
            cv2.imshow(configuration["rtsp"]["feed"], video_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Save video
        if SAVE_VIDEO and OUT_VIDEO_WRITER is not None:
            now = datetime.now()

            # Change every hour
            if now.hour != hour:
                # Release before reconstructing
                OUT_VIDEO_WRITER.stdin.close()
                OUT_VIDEO_WRITER.wait()

                year = now.year
                month = now.month
                day = now.day
                hour = now.hour
                minute = now.minute
                second = now.second

                output_video_name = configuration["rtsp"]["save_video"]["name"]
                output_video_path = configuration["rtsp"]["save_video"]["path"]
                output_video_path = (
                    f"{output_video_path}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
                )

                output_video_name = (
                    f"{output_video_name}_{year}"
                    f"-{month}"
                    f"-{day}"
                    f"_{hour}"
                    f"-{minute}"
                    f"-{second}"
                    f".{output_video_format}"
                )

                output_video = f"{output_video_path}/{output_video_name}"

                SAVE_IMAGE_PATH = f"{output_video_path}/captures"

                try:
                    os.makedirs(SAVE_IMAGE_PATH)
                except FileExistsError:
                    pass

                OUT_VIDEO_WRITER = writer_stream(
                    output_video, video_width, video_height, video_fps
                )
            OUT_VIDEO_WRITER.stdin.write(video_frame.tobytes())

    # Release and close threading
    executor.shutdown(wait=True)

    # Stop reader
    STOP_EVENT.set()
    stream_reader_thread.join(timeout=2)
    web_thread.join(timeout=2)

    # Stop writer
    if SAVE_VIDEO:
        OUT_VIDEO_WRITER.stdin.close()
        OUT_VIDEO_WRITER.wait()

    # Destroy window if display was set
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()

    shutil.rmtree(HLS_DIR, ignore_errors=True)
