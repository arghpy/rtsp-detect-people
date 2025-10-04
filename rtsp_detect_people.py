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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage

import cv2
import numpy as np

# pylint: disable=import-error
from ultralytics import YOLO

# Load YOLOv8n model (it will auto-download if missing)
model = YOLO("yolov8n.pt")

# Font settings
FONT_NAME = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Args
SHOW_DISPLAY = False
SAVE_VIDEO = False
SEND_EMAIL = False
CONFIGURATION_FILE = None

# Other globals
MAX_FRAME_DROPS = 5
BOX_COLOR = (0, 255, 0)  # (B, G, R) colors - Green
CONFIDENCE_MIN = 0.45


# pylint: disable=unused-argument
def handle_signals(signum, exec_frame):
    """Respond to different signals"""
    signame = signal.Signals(signum).name
    pprint(f"Received {signame}({signum})")

    if signum == signal.SIGINT:
        pprint("Exiting")
        sys.exit(1)


signal.signal(signal.SIGINT, handle_signals)


def eprint(s):
    """Print to stderr with current time"""
    print(f"{datetime.now()}: {s}", file=sys.stderr, flush=True)


def pprint(s):
    """Print to stdout with current time"""
    print(f"{datetime.now()}: {s}", file=sys.stdout, flush=True)


def send_email_report(frame, image_type, config):
    """Send email based on the environment variables"""
    pprint("Person detected. Sending email")

    save_image_path = f"person.{image_type}"
    cv2.imwrite(save_image_path, frame)

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
            subtype=image_type,
            filename=save_image_path,
        )

    with smtplib.SMTP_SSL(config["email"]["server"], config["email"]["port"]) as s:
        s.login(config["email"]["user"], config["email"]["password"])
        s.send_message(msg)

    os.remove(save_image_path)


def writer_stream(video_path, width, height, fps) -> subprocess.Popen:
    """Write stream to file"""
    pprint(f"Saving to {video_path}")

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
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-g",
        f"{fps*2}",  # keyframe every 2 seconds
        "-x264-params",
        f"keyint={fps*2}:min-keyint={fps}",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "matroska",
        video_path,
    ]
    # pylint: disable=consider-using-with
    writer = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE)
    return writer


def read_frame(pipe, width, height) -> np.ndarray | None:
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


# pylint: disable=too-many-arguments,too-many-positional-arguments
def reader_frames_thread(frame_queue, width, height, fps, rtsp_url, stop_event):
    """Continuously add frames in queue to be processed"""
    pprint("Reader thread started")

    pipe = reader_stream(rtsp_url)
    dropped_frames = 0
    tried_to_reconnect = False

    while not stop_event.is_set():
        try:
            frame = read_frame(pipe.stdout, width, height)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            eprint(f"Exception reading frame: {e}")
            pipe.kill()  # terminate old ffmpeg
            pipe.stdout.close()
            pipe = reader_stream(rtsp_url)  # restart reader
            dropped_frames = 0
            continue

        if frame is None:
            dropped_frames += 1
            if dropped_frames >= fps * 2:
                eprint(f"{dropped_frames} consecutive frames missing. Reconnecting")
                time.sleep(2)
                pipe.stdout.close()
                pipe.kill()  # terminate old ffmpeg
                pipe = reader_stream(rtsp_url)  # restart reader
                dropped_frames = 0
                tried_to_reconnect = True
        else:
            if tried_to_reconnect:
                pprint("Successfully reconnected")
                tried_to_reconnect = False

            dropped_frames = 0
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                # Queue full → drop frame to avoid blocking
                pass

    pipe.stdout.close()
    pipe.terminate()


def reader_stream(rtsp_url) -> subprocess.Popen:
    """Continuously get frames from stream"""
    pprint("Starting ffmpeg reader")

    reader_cmd = [
        "ffmpeg",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-loglevel",
        "error",
        "-an",
        "-sn",  # disable audio and subs
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    # pylint: disable=consider-using-with
    reader = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE)
    return reader


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
    global SHOW_DISPLAY, SAVE_VIDEO, SEND_EMAIL, CONFIGURATION_FILE

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
    PERSON_DETECTED = False
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
    TIMEOUT = int(configuration["timeout"])  # Secs
    rtsp_user = configuration["rtsp"]["user"]
    rtsp_password = configuration["rtsp"]["password"]
    rtsp_feed = configuration["rtsp"]["feed"]
    RTSP_URL = f"rtsp://{rtsp_user}:{rtsp_password}@{rtsp_feed}"

    # Frame and properties
    video_width, video_height, video_fps = probe_stream(RTSP_URL)
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

    if SAVE_VIDEO:
        output_video_path = configuration["rtsp"]["save_video"]["path"]
        output_video_name = configuration["rtsp"]["save_video"]["name"]
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

        try:
            os.makedirs(output_video_path)
        except FileExistsError:
            # directory already exists
            pass
        OUT_VIDEO_WRITER = writer_stream(
            output_video, video_width, video_height, video_fps
        )

    # MAIN LOOP
    while True:
        try:
            video_frame = FRAME_QUEUE.get(timeout=1)
            video_frame = video_frame.copy()
        except queue.Empty:
            continue

        # Check for corrupt frame
        if (
            video_frame.size == 0
            or video_frame.shape[0] < 50
            or video_frame.shape[1] < 50
        ):
            eprint("[WARN] Corrupt frame detected, skipping...")
            continue

        # Run model on frame
        results = model(video_frame, conf=CONFIDENCE_MIN, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                if model.names[cls] != "person":
                    PERSON_DETECTED = False
                else:
                    PERSON_DETECTED = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        video_frame,
                        f"Person: {confidence*100:.2f}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # Send email
        if SEND_EMAIL:
            if email_sent and email_future is not None:
                if email_future.done():
                    pprint("Email sent")
                    email_sent = False
                    email_future = None

            if PERSON_DETECTED and (time.time() - start_timeout) > TIMEOUT:
                email_future = executor.submit(
                    send_email_report,
                    video_frame.copy(),
                    SAVE_IMAGE_TYPE,
                    configuration,
                )
                email_sent = True
                start_timeout = time.time()

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

                output_video_path = configuration["rtsp"]["save_video"]["path"]
                output_video_path = (
                    f"{output_video_path}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
                )

                output_video_name = (
                    f"front_from_{year}"
                    f"-{month}"
                    f"-{day}"
                    f"_{hour}"
                    f"-{minute}"
                    f"-{second}"
                    f".{output_video_format}"
                )

                output_video = f"{output_video_path}/{output_video_name}"

                try:
                    os.makedirs(output_video_path)
                except FileExistsError:
                    # directory already exists
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

    # Stop writer
    if SAVE_VIDEO:
        OUT_VIDEO_WRITER.stdin.close()
        OUT_VIDEO_WRITER.wait()

    # Destroy window if display was set
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
