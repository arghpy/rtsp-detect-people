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
import numpy as np

# MobileNet
MOBILENET_RESIZED_DIMENSIONS = (300, 300)  # Dimensions that SSD was trained on.
MOBILE_NET_IMG_NORM_RATIO = 0.007843  # In grayscale a pixel can range between 0 and 255

MOBILENET_CATEGORIES = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}

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
    msg["Subject"] = (
        config["email"]["subject"] + f": {time.strftime("%Y-%m-%d_%H:%M:%S")}"
    )
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


# pylint: disable=too-many-locals
def process_frame_mobilenet(net, frame):
    """Process frame using MobileNet-SSD"""
    (original_height, original_width) = frame.shape[:2]

    # Preprocess input
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, MOBILENET_RESIZED_DIMENSIONS),
        MOBILE_NET_IMG_NORM_RATIO,
        MOBILENET_RESIZED_DIMENSIONS,
        127.5,
    )
    net.setInput(blob)
    detections = net.forward()

    person_found = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_MIN:
            class_id = int(detections[0, 0, i, 1])
            label = MOBILENET_CATEGORIES[class_id]

            if label == "person":
                person_found = True

                box = detections[0, 0, i, 3:7] * np.array(
                    [original_width, original_height, original_width, original_height]
                )
                (start_x, start_y, end_x, end_y) = box.astype("int")

                cv2.rectangle(
                    frame, (start_x, start_y), (end_x, end_y), BOX_COLOR, FONT_THICKNESS
                )
                cv2.putText(
                    frame,
                    f"Person: {confidence*100:.2f}%",
                    (start_x, start_y - 5),
                    FONT_NAME,
                    FONT_SCALE,
                    BOX_COLOR,
                    FONT_THICKNESS,
                )
    return frame, person_found


def load_mobilenet_model(config):
    """Load MobileNet-SSD model"""
    # Load the pre-trained neural network
    net = cv2.dnn.readNetFromCaffe(
        config["mobilenet"]["prototxt"], config["mobilenet"]["caffemodel"]
    )
    return net


def try_to_connect_stream(config):
    """Try to reconnect the stream"""
    wait_sec = 5

    while True:
        cap = cv2.VideoCapture(
            f"rtsp://{config["rtsp"]["user"]}:{config["rtsp"]["password"]}"
            f"@{config["rtsp"]["camera"]}",
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
    )

    program_usage = f"{argv[0]} {" ".join([f"{opt[0]}" if opt[2] is True else f"[{opt[0]}]" for opt in options])}"
    program_description = usage_description(
        "Detect people from RTSP stream."
    )
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

    video_capture = try_to_connect_stream(configuration)

    if SAVE_VIDEO:
        return_value, video_frame = video_capture.read()

        # Restart the stream
        if return_value is False or video_frame is None:
            eprint("Lost connection. Trying to reconnect")
            video_capture.release()
            video_capture = try_to_connect_stream(configuration)

        video_frame_height, video_frame_width, _ = video_frame.shape

        video_path = configuration["rtsp"]["save_video"]["path"]
        video_name = configuration["rtsp"]["save_video"]["name"]
        video_path = (
            f"{video_path}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
        )
        output_video_format = configuration["rtsp"]["save_video"]["format"]
        output_video_fps = float(configuration["rtsp"]["save_video"]["fps"])
        output_video_name = (
            f"{video_name}_{year}"
            f"-{month}"
            f"-{day}"
            f"_{hour}"
            f"-{minute}"
            f"-{second}"
            f".{output_video_format}"
        )

        output_video = f"{video_path}/{output_video_name}"
        print(f"Saving to {output_video}...", flush=True)

        try:
            os.makedirs(video_path)
        except FileExistsError:
            # directory already exists
            pass

        out = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"H264"),
            output_video_fps,
            (video_frame_width, video_frame_height),
        )

    # Create executor
    executor = ThreadPoolExecutor(max_workers=1)

    # Load neural network
    neural_network = load_mobilenet_model(configuration)

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

        video_frame, something_found = process_frame_mobilenet(
            neural_network, video_frame
        )

        # Send email
        if SEND_EMAIL:
            if email_sent and email_future is not None:
                if email_future.done():
                    print("Email sent.", flush=True)
                    email_sent = False
                    email_future = None

            if something_found and (time.time() - start_timeout) > TIMEOUT:
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
            cv2.imshow(configuration["rtsp"]["camera"], video_frame)
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

                video_path = configuration["rtsp"]["save_video"]["path"]
                video_path = (
                    f"{video_path}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
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

                output_video = f"{video_path}/{output_video_name}"
                print(f"Saving to {output_video}...", flush=True)

                try:
                    os.makedirs(video_path)
                except FileExistsError:
                    # directory already exists
                    pass

                # the output will be written to output.avi
                out = cv2.VideoWriter(
                    output_video,
                    cv2.VideoWriter_fourcc(*"H264"),
                    output_video_fps,
                    (video_frame_width, video_frame_height),
                )
            out.write(video_frame)

    # Release and close threading
    executor.shutdown(wait=True)
    video_capture.release()
    if SAVE_VIDEO:
        out.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
