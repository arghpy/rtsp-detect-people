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

    if ARGS["ENABLE_WEB"]:
        HLS_WRITER = hls_writer(HLS_DIR, video_width, video_height, video_fps)
        web_thread = threading.Thread(
            target=start_web_server, args=(ARGS["WEB_PORT"],), daemon=True
        )
        web_thread.start()

    if ARGS["SAVE_VIDEO"]:
        output_video_path = (
            f"{CONFIG['VIDEO_PATH']}" f"/{now.year}" f"/{now.month}" f"/{now.day}" f"/{now.hour}"
        )
        output_video_format = "mkv"
        output_video_name = (
            f"{CONFIG['VIDEO_NAME']}_{now.year}"
            f"-{now.month}"
            f"-{now.day}"
            f"_{now.hour}"
            f"-{now.minute}"
            f"-{now.second}"
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
        video_frame, PERSON_DETECTED = process_frame(CONFIG["RTSP_URL"])
        print(f"size: {video_frame.size}, {video_frame.shape[0]} {video_frame.shape[1]}")

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

            SAVE_IMAGE_PATH = f"{output_video_path}/captures"
            SAVE_IMAGE_NAME = (
                f"{CONFIG['VIDEO_NAME']}" f"_{now.minute}" f":{now.second}" f".{SAVE_IMAGE_TYPE}"
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
            if now.hour != now.hour:
                # Release before reconstructing
                OUT_VIDEO_WRITER.stdin.close()
                OUT_VIDEO_WRITER.wait()

                now.year = now.year
                now.month = now.month
                now.day = now.day
                now.hour = now.hour
                now.minute = now.minute
                now.second = now.second

                output_video_path = (
                    f"{CONFIG['VIDEO_PATH']}" f"/{now.year}" f"/{now.month}" f"/{now.day}" f"/{now.hour}"
                )

                output_video_name = (
                    f"{CONFIG['VIDEO_NAME']}_{now.year}"
                    f"-{now.month}"
                    f"-{now.day}"
                    f"_{now.hour}"
                    f"-{now.minute}"
                    f"-{now.second}"
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
