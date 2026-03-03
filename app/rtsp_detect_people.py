#!/usr/bin/env python3
"""Detect people from an RTSP stream"""

import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime

import cv2
import requests

from help import usage
from config import CONFIG, process_configuration
from core import eprint, pprint
from detection import process_frame
from home_assistant import ha_trigger_boolean
from notifications import send_ntfy
from video_processing import writer_stream, probe_stream
from webserver import HLS_DIR, hls_writer, start_web_server

# Args
ARGS = {}
ARGS["CONFIGURATION_FILE"] = None
ARGS["SHOW_DISPLAY"] = False
ARGS["SAVE_VIDEO"] = False
ARGS["ENABLE_WEB"] = False
ARGS["WEB_PORT"] = None

# Other globals
MAX_FRAME_DROPS = 5


# pylint: disable=unused-argument
def handle_signals(signum, exec_frame):
    """Respond to different signals"""
    global STOP_EVENT

    signame = signal.Signals(signum).name
    pprint(f"Received {signame}({signum})")

    # Stop reader
    STOP_EVENT.set()
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
    # pylint: disable=global-statement,line-too-long
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
        elif passed_args[0] == "--ha-light":
            ARGS["HA_LIGHT"] = True
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
    start_timeout = 0
    STOP_EVENT = threading.Event()
    PERSON_DETECTED = False
    OCCUPANCY_DETECTED = False
    HA_TOGGLE = False

    if ARGS["CONFIGURATION_FILE"] is None:
        eprint("Configuration not specified.")
        usage(sys.argv)
        sys.exit(1)

    process_configuration(ARGS["CONFIGURATION_FILE"])

    # Frame and properties
    video_width, video_height, video_fps = probe_stream(CONFIG["RTSP_URL"])
    # Scale all streams to 1920 width
    target_width = 1920
    target_height = int(video_height * target_width / video_width)

    # Ensure width/height are even for NVENC
    target_width = (target_width // 2) * 2
    target_height = (target_height // 2) * 2

    pprint(f"Original resolution for encoding: {video_width}x{video_height}")
    pprint(f"Target resolution for encoding  : {target_width}x{target_height}")

    if ARGS["ENABLE_WEB"]:
        HLS_WRITER = hls_writer(HLS_DIR, target_width, target_height, video_fps)
        web_thread = threading.Thread(
            target=start_web_server, args=(ARGS["WEB_PORT"],), daemon=True
        )
        web_thread.start()

    if ARGS["SAVE_VIDEO"]:
        output_video_path = (
            f"{CONFIG['VIDEO_PATH']}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
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
        # Run model on frame
        for frame, PERSON_DETECTED in process_frame(CONFIG["RTSP_URL"], video_width, video_height):
            video_frame = cv2.resize(frame, (target_width, target_height))
            if PERSON_DETECTED and not OCCUPANCY_DETECTED:
                OCCUPANCY_DETECTED = True
                if ARGS["HA_LIGHT"]:
                    HA_TOGGLE = not HA_TOGGLE
                    ha_trigger_boolean(HA_TOGGLE)
            elif not PERSON_DETECTED and OCCUPANCY_DETECTED:
                OCCUPANCY_DETECTED = False
                if ARGS["HA_LIGHT"]:
                    HA_TOGGLE = not HA_TOGGLE
                    ha_trigger_boolean(HA_TOGGLE)

            if PERSON_DETECTED and (time.time() - start_timeout) > CONFIG["TIMEOUT"]:
                now = datetime.now()
                year = now.year
                month = now.month
                day = now.day
                hour = now.hour
                minute = now.minute
                second = now.second

                SAVE_IMAGE_PATH = f"{output_video_path}/captures"
                SAVE_IMAGE_NAME = (
                    f"{CONFIG['VIDEO_NAME']}" f"_{minute}" f":{second}" f".{SAVE_IMAGE_TYPE}"
                )
                SAVE_IMAGE = f"{SAVE_IMAGE_PATH}/{SAVE_IMAGE_NAME}"
                rc = cv2.imwrite(SAVE_IMAGE, video_frame)
                if rc:
                    pprint(f"Saved image to {SAVE_IMAGE}")
                else:
                    eprint(f"Failed to save image to {SAVE_IMAGE}")

                if ARGS["SEND_NTFY"]:
                    try:
                        send_ntfy(
                            "Person detected",
                            "",
                            SAVE_IMAGE,
                            f"detection.{SAVE_IMAGE_TYPE}",
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
                if hour != now.hour:
                    # Release before reconstructing
                    OUT_VIDEO_WRITER.stdin.close()
                    OUT_VIDEO_WRITER.wait()

                    year = now.year
                    month = now.month
                    day = now.day
                    hour = now.hour
                    minute = now.minute
                    second = now.second

                    output_video_path = (
                        f"{CONFIG['VIDEO_PATH']}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
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

    # Stop reader
    STOP_EVENT.set()
    web_thread.join(timeout=2)

    # Stop writer
    if ARGS["SAVE_VIDEO"]:
        OUT_VIDEO_WRITER.stdin.close()
        OUT_VIDEO_WRITER.wait()

    # Destroy window if display was set
    if ARGS["SHOW_DISPLAY"]:
        cv2.destroyAllWindows()

    shutil.rmtree(HLS_DIR, ignore_errors=True)
