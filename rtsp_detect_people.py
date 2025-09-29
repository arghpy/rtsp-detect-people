#!/usr/bin/env python3
"""Detect people from an RTSP stream"""
# pylint: disable=c-extension-no-member

import json
import os
import signal
import smtplib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage

import cv2
from ultralytics import YOLO

# Load YOLOv8n model (it will auto-download if missing)
model = YOLO("yolov8n.pt")

# Confidence parameters
# (B, G, R) colors
BOX_COLOR = (0, 255, 0)  # Green
# Levels to check
CONFIDENCE_MIN = 0.45

# Font settings
FONT_NAME = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Args
SHOW_DISPLAY = False
SAVE_VIDEO = False
SEND_EMAIL = False
out = None
CONFIGURATION_FILE = None
PLAY_VIDEO = None


# pylint: disable=unused-argument
def handle_signals(signum, exec_frame):
    """Respond to different signals"""
    signame = signal.Signals(signum).name
    print(f"Received {signame}({signum})", flush=True)

    if signum == signal.SIGINT:
        print("Exiting...", flush=True)
        sys.exit(1)


signal.signal(signal.SIGINT, handle_signals)


def eprint(s):
    """Print to stderr"""
    print(f"{s}", file=sys.stderr, flush=True)


def send_email_report(frame, image_type, config):
    """Send email based on the environment variables"""
    print("Person detected. Sending email...", flush=True)

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


def try_to_connect_stream(config):
    """Try to reconnect the stream"""
    wait_sec = 5

    while True:
        if PLAY_VIDEO is not None:
            cap = cv2.VideoCapture(
                PLAY_VIDEO,
                cv2.CAP_FFMPEG,
            )
        else:
            rtsp_user = config["rtsp"]["user"]
            rtsp_password = config["rtsp"]["password"]
            rtsp_feed = config["rtsp"]["feed"]
            cap = cv2.VideoCapture(
                f"rtsp://{rtsp_user}:{rtsp_password}" f"@{rtsp_feed}",
                cv2.CAP_FFMPEG,
            )

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # reduce resolution before decoding
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)   # or 320
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)  # or 240

        if cap.isOpened():
            print("Succesfully connected to stream", flush=True)
            return cap

        eprint("Failed to reconnect.")
        eprint(f"Wait for {wait_sec} seconds and try again")
        cap.release()
        time.sleep(wait_sec)


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
    s = f"\n\nDESCRIPTION" f"\n\t{description}"
    return s


def usage_header(header):
    s = f"\n\n{header.upper()}"
    return s


def add_option(option, description):
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
        ("-v/--video VIDEO", "test with video", True),
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
    global SHOW_DISPLAY, SAVE_VIDEO, SEND_EMAIL, CONFIGURATION_FILE, PLAY_VIDEO

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
        elif passed_args[0] == "-v" or passed_args[0] == "--video":
            passed_args.pop(0)
            PLAY_VIDEO = passed_args[0]
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
    person_detected = False
    # pylint: disable=invalid-name
    email_sent = False
    email_future = None
    start_timeout = 0

    if CONFIGURATION_FILE is None:
        eprint("Configuration not specified.")
        usage(sys.argv)
        sys.exit(1)

    configuration = load_json_file(CONFIGURATION_FILE)
    TIMEOUT = int(configuration["timeout"])  # Secs

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = configuration["rtsp"][
        "opencv_ffmpeg_capture_options"
    ]

    # Frame and properties
    video_capture = try_to_connect_stream(configuration)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if SAVE_VIDEO:
        return_value, video_frame = video_capture.read()

        # Restart the stream
        if return_value is False or video_frame is None:
            eprint("Lost connection. Trying to reconnect")
            video_capture.release()
            video_capture = try_to_connect_stream(configuration)

        output_video_path = configuration["rtsp"]["save_video"]["path"]
        output_video_name = configuration["rtsp"]["save_video"]["name"]
        output_video_path = (
            f"{output_video_path}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
        )
        output_video_format = configuration["rtsp"]["save_video"]["format"]
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
        print(f"Saving to {output_video}...", flush=True)

        try:
            os.makedirs(output_video_path)
        except FileExistsError:
            # directory already exists
            pass

        out = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"FFV1"),
            video_fps,
            (video_width, video_height),
        )

    # Create executor
    executor = ThreadPoolExecutor(max_workers=1)

    while True:
        return_value, video_frame = video_capture.read()

        # Restart the stream
        if return_value is False or video_frame is None:
            eprint("Lost connection. Trying to reconnect")
            video_capture.release()
            video_capture = try_to_connect_stream(configuration)
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
        results = model(video_frame, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                if model.names[cls] == "person" and confidence > CONFIDENCE_MIN:
                    person_detected = True
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
                    print("Email sent.", flush=True)
                    email_sent = False
                    email_future = None

            if person_detected and (time.time() - start_timeout) > TIMEOUT:
                email_future = executor.submit(
                    send_email_report,
                    video_frame.copy(),
                    SAVE_IMAGE_TYPE,
                    configuration,
                )
                email_sent = True
                start_timeout = time.time()
                person_detected = False

        # Show display
        if SHOW_DISPLAY:
            cv2.imshow(configuration["rtsp"]["feed"], video_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Save video
        if SAVE_VIDEO and out is not None:
            now = datetime.now()

            # Change every hour
            if now.hour != hour:
                # Release before reconstructing
                out.release()

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
                print(f"Saving to {output_video}...", flush=True)

                try:
                    os.makedirs(output_video_path)
                except FileExistsError:
                    # directory already exists
                    pass

                # the output will be written to output.avi
                out = cv2.VideoWriter(
                    output_video,
                    cv2.VideoWriter_fourcc(*"H264"),
                    video_fps,
                    (video_width, video_height),
                )
            out.write(video_frame)

    # Release and close threading
    executor.shutdown(wait=True)
    video_capture.release()
    if SAVE_VIDEO:
        out.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
