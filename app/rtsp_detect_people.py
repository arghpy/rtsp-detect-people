#!/usr/bin/env python3
"""Detect people from an RTSP stream"""

# pylint: disable=c-extension-no-member

import os
import queue
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import requests
from app.integrations.email import send_email_report
from app.integrations.home_assistant import ha_trigger_boolean
from app.integrations.ntfy import send_ntfy
from app.integrations.webserver import HLS_DIR, hls_writer, start_web_server
from app.utils.config import CONFIG, process_configuration
from app.utils.help import usage
from app.utils.logger import eprint, pprint
from app.utils.video import probe_stream, reader_frames_thread, writer_stream
from app.yolo.detection import process_frame, load_model

# Args
ARGS = {}
ARGS["CONFIGURATION_FILE"] = None
ARGS["ENABLE_WEB"] = False
ARGS["HA_TRIGGER"] = False
ARGS["SAVE_VIDEO"] = False
ARGS["SEND_EMAIL"] = False
ARGS["SEND_NTFY"] = False
ARGS["SHOW_DISPLAY"] = False
ARGS["WEB_PORT"] = None


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
    if ARGS["SAVE_VIDEO"]:
        OUT_VIDEO_WRITER.stdin.close()
        OUT_VIDEO_WRITER.wait()

    # Destroy window if display was set
    if ARGS["SHOW_DISPLAY"]:
        cv2.destroyAllWindows()

    shutil.rmtree(HLS_DIR, ignore_errors=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_signals)
signal.signal(signal.SIGINT, handle_signals)


def parse_arguments(argv):
    """Parse command line arguments"""
    global ARGS

    passed_args = argv[1:]

    while len(passed_args) > 0:
        if passed_args[0] == "-h" or passed_args[0] == "--help":
            usage(argv)
            sys.exit(0)
        elif passed_args[0] == "-d" or passed_args[0] == "--display":
            ARGS["SHOW_DISPLAY"] = True
        elif passed_args[0] == "-s" or passed_args[0] == "--save":
            ARGS["SAVE_VIDEO"] = True
        elif passed_args[0] == "-e" or passed_args[0] == "--email":
            ARGS["SEND_EMAIL"] = True
        elif passed_args[0] == "-n" or passed_args[0] == "--ntfy":
            ARGS["SEND_NTFY"] = True
        elif passed_args[0] == "-c" or passed_args[0] == "--config":
            passed_args.pop(0)
            ARGS["CONFIGURATION_FILE"] = passed_args[0]
        elif passed_args[0] == "-w" or passed_args[0] == "--web":
            ARGS["ENABLE_WEB"] = True
            passed_args.pop(0)
            ARGS["WEB_PORT"] = passed_args[0]
        elif passed_args[0] == "--ha-trigger":
            ARGS["HA_TRIGGER"] = True
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

    OUT_VIDEO_WRITER = None
    # pylint: disable=invalid-name
    email_sent = False
    email_future = None
    start_timeout = 0
    STOP_EVENT = threading.Event()

    # Create executor
    executor = ThreadPoolExecutor(max_workers=1)

    if ARGS["CONFIGURATION_FILE"] is None:
        eprint("Configuration not specified.")
        usage(sys.argv)
        sys.exit(1)

    process_configuration(ARGS["CONFIGURATION_FILE"])
    load_model()

    PERSON_DETECTED = False
    OCCUPANCY_DETECTED = False
    HA_TOGGLE = False

    # Frame and properties
    video_width, video_height, video_fps = probe_stream(CONFIG["RTSP_URL"])
    try:
        video_fps = CONFIG["VIDEO_FPS"]
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
            CONFIG["RTSP_URL"],
            STOP_EVENT,
        ),
        daemon=True,
    )
    stream_reader_thread.start()

    if ARGS["ENABLE_WEB"]:
        HLS_WRITER = hls_writer(HLS_DIR, video_width, video_height, video_fps)
        web_thread = threading.Thread(
            target=start_web_server, args=(ARGS["WEB_PORT"],), daemon=True
        )
        web_thread.start()

    if ARGS["SAVE_VIDEO"]:
        output_video_path = CONFIG["VIDEO_PATH"]
        output_video_path = (
            f"{output_video_path}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
        )
        output_video_format = "mkv"
        output_video_name = (
            f"{CONFIG['VIDEO_NAME']}_{year}"
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
        video_frame, PERSON_DETECTED = process_frame(video_frame)

        # Send email
        if ARGS["SEND_EMAIL"]:
            if email_sent and email_future is not None:
                if email_future.done():
                    pprint("Email sent")
                    email_sent = False
                    email_future = None

        if PERSON_DETECTED and not OCCUPANCY_DETECTED:
            OCCUPANCY_DETECTED = True
            if ARGS["HA_TRIGGER"]:
                HA_TOGGLE = not HA_TOGGLE
                ha_trigger_boolean(HA_TOGGLE)
        elif not PERSON_DETECTED and OCCUPANCY_DETECTED:
            OCCUPANCY_DETECTED = False
            if ARGS["HA_TRIGGER"]:
                HA_TOGGLE = not HA_TOGGLE
                ha_trigger_boolean(HA_TOGGLE)

        if PERSON_DETECTED and (time.time() - start_timeout) > CONFIG["TIMEOUT"]:
            now = datetime.now()
            minute = now.minute
            second = now.second

            SAVE_IMAGE_PATH = f"{output_video_path}/captures"
            SAVE_IMAGE_NAME = (
                f"{CONFIG['VIDEO_NAME']}"
                f"_{minute}"
                f":{second}"
                f".jpeg"
            )
            SAVE_IMAGE = f"{SAVE_IMAGE_PATH}/{SAVE_IMAGE_NAME}"
            rc = cv2.imwrite(SAVE_IMAGE, video_frame)
            if rc:
                pprint(f"Saved image to {SAVE_IMAGE}")
            else:
                eprint(f"Failed to save image to {SAVE_IMAGE}")

            if ARGS["SEND_EMAIL"]:
                email_future = executor.submit(
                    send_email_report,
                    SAVE_IMAGE,
                )
                email_sent = True

            if ARGS["SEND_NTFY"]:
                try:
                    send_ntfy(
                        CONFIG["NTFY_URL"],
                        CONFIG["NTFY_TAG"],
                        "Person detected",
                        "",
                        SAVE_IMAGE,
                        "detection.jpeg",
                    )
                    pprint("Successfully sent ntfy")
                except requests.exceptions.HTTPError:
                    eprint("Failed to send ntfy")

            start_timeout = time.time()

        if ARGS["ENABLE_WEB"]:
            HLS_WRITER.stdin.write(video_frame.tobytes())

        # Show display
        if ARGS["SHOW_DISPLAY"]:
            cv2.imshow(CONFIG["RTSP_FEED"], video_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Save video
        if ARGS["SAVE_VIDEO"] and OUT_VIDEO_WRITER is not None:
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

                output_video_path = CONFIG["VIDEO_PATH"]
                output_video_path = (
                    f"{output_video_path}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
                )

                output_video_name = (
                    f"{CONFIG['VIDEO_NAME']}_{year}"
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
    if ARGS["SAVE_VIDEO"]:
        OUT_VIDEO_WRITER.stdin.close()
        OUT_VIDEO_WRITER.wait()

    # Destroy window if display was set
    if ARGS["SHOW_DISPLAY"]:
        cv2.destroyAllWindows()

    shutil.rmtree(HLS_DIR, ignore_errors=True)
