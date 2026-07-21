#!/usr/bin/env python3
"""Detect people from an RTSP stream"""

# pylint: disable=c-extension-no-member

import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import requests

import app.utils.config
import app.utils.help
import app.utils.logger
import app.utils.video

# Args
ARGS = {}
ARGS["CONFIGURATION_FILE"] = None
ARGS["CAMERA"] = False
ARGS["HA_TRIGGER"] = False
ARGS["SEND_NTFY"] = False
ARGS["DETECTION"] = False
ARGS["CAMERA_PATH"] = None
cap = None


# pylint: disable=unused-argument
def handle_signals(signum, exec_frame):
    """Respond to different signals"""
    global STOP_EVENT

    signame = signal.Signals(signum).name
    app.utils.logger.pprint(f"Received {signame}({signum})")

    # Stop reader
    STOP_EVENT.set()
    stream_reader_thread.join(timeout=2)
    cap.release()
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_signals)
signal.signal(signal.SIGINT, handle_signals)


def parse_arguments(argv):
    """Parse command line arguments"""
    global ARGS

    passed_args = argv[1:]

    while len(passed_args) > 0:
        if passed_args[0] == "-h" or passed_args[0] == "--help":
            app.utils.help.usage(argv)
            sys.exit(0)
        elif passed_args[0] == "-n" or passed_args[0] == "--ntfy":
            ARGS["SEND_NTFY"] = True
        elif passed_args[0] == "-d" or passed_args[0] == "--detection":
            ARGS["DETECTION"] = True
        elif passed_args[0] == "-c" or passed_args[0] == "--config":
            passed_args.pop(0)
            ARGS["CONFIGURATION_FILE"] = passed_args[0]
        elif passed_args[0] == "--camera":
            ARGS["CAMERA"] = True
            passed_args.pop(0)
            ARGS["CAMERA_PATH"] = passed_args[0]
        elif passed_args[0] == "--ha-trigger":
            ARGS["HA_TRIGGER"] = True
        else:
            app.utils.logger.eprint(f"Invalid option: {passed_args[0]}")
            app.utils.help.usage(argv)
            sys.exit(0)
        passed_args.pop(0)

    if ARGS["CAMERA"] == False:
        app.utils.logger.eprint("--camera option missing.")
        app.utils.help.usage(argv)
        sys.exit(1)



if __name__ == "__main__":
    parse_arguments(sys.argv)

    # pylint: disable=invalid-name
    start_timeout = 0
    STOP_EVENT = threading.Event()

    if ARGS["CONFIGURATION_FILE"] is None:
        app.utils.logger.eprint("Configuration not specified.")
        app.utils.help.usage(sys.argv)
        sys.exit(1)

    app.utils.config.process_configuration(ARGS["CONFIGURATION_FILE"])

    if (
        app.utils.config.CONFIG["NTFY_URL"] is None
        or app.utils.config.CONFIG["NTFY_TAG"] is None
    ):
        ARGS["SEND_NTFY"] = False
    else:
        import app.integrations.ntfy

    if (
        app.utils.config.CONFIG["HA_ENTITY_ID"] is None
        or app.utils.config.CONFIG["HA_ENTITY_TYPE"] is None
    ):
        ARGS["HA_TRIGGER"] = False
    else:
        import app.integrations.home_assistant

    if ARGS["DETECTION"]:
        import app.yolo.detection
        app.yolo.detection.load_model()
        PERSON_DETECTED = False
        OCCUPANCY_DETECTED = False
        OCCUPANCY_DETECTED_TIMEOUT = 10  # secs
        OCCUPANCY_LAST_SEEN = 0  # timestamp of last detection
        HA_TOGGLE = False

    # Frame and properties
    video_width, video_height, video_fps = app.utils.video.probe_stream(
        app.utils.config.CONFIG["RTSP_URL"]
    )
    if video_fps < 10 or video_fps > 50:
        try:
            video_fps = app.utils.config.CONFIG["VIDEO_FPS"]
            app.utils.logger.pprint(f"FPS was overridden by the config to {video_fps}")
        except KeyError:
            pass

    # if it doesn't exist in config, default value will be used
    MAX_BATCH_SIZE = app.utils.config.CONFIG["YOLO_BATCH"]

    QUEUE_SIZE = video_fps
    if 3 * MAX_BATCH_SIZE < video_fps:
        QUEUE_SIZE = 3 * MAX_BATCH_SIZE

    FRAME_QUEUE = queue.Queue(maxsize=int(QUEUE_SIZE))

    # Open the stream
    cap = cv2.VideoCapture(f"rtsp://mediamtx:8554/{ARGS['CAMERA_PATH']}")
    while not cap.isOpened():
        app.utils.logger.eprint(f"Could not read from rtsp://mediamtx:8554/{ARGS['CAMERA_PATH']}")
        cap.open(f"rtsp://mediamtx:8554/{ARGS['CAMERA_PATH']}")

    stream_reader_thread = threading.Thread(
        target=app.utils.video.collect_frames,
        args=(
            cap,
            FRAME_QUEUE,
            STOP_EVENT,
        ),
        daemon=True,
    )
    stream_reader_thread.start()

    # MAIN LOOP
    while True:
        while not cap.isOpened():
            app.utils.logger.eprint(f"Could not read from rtsp://mediamtx:8554/{ARGS['CAMERA_PATH']}")
            cap.open(f"rtsp://mediamtx:8554/{ARGS['CAMERA_PATH']}")
        # Create directory structure
        now = datetime.now()

        output_video_path = app.utils.config.CONFIG["VIDEO_PATH"]
        output_video_path = (
            f"{output_video_path}"
            f"{now.strftime('/%Y/%m/%d/%H')}"
        )

        SAVE_IMAGE_PATH = f"{output_video_path}/captures"
        os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)

        frames = []
        batch_timeout = min(MAX_BATCH_SIZE/video_fps, 0.1)  # calculate the ideal time to wait for MAX_BATCH_SIZE frames
        start_time = time.time()

        while len(frames) < MAX_BATCH_SIZE:
            try:
                frame = FRAME_QUEUE.get(timeout=batch_timeout)
                if frame.size == 0 or frame.shape[0] < 50 or frame.shape[1] < 50:
                    app.utils.logger.eprint(
                        "[WARN] Corrupt frame detected, skipping..."
                    )
                    continue
                frames.append(frame)
            except queue.Empty:
                break

            # stop early if we've waited too long
            if time.time() - start_time >= batch_timeout:
                break

        if len(frames) == 0:
            continue

        if ARGS["DETECTION"]:
            processed_frames = app.yolo.detection.process_frames(frames)
            processed_frames_bytes = b"".join(f.tobytes() for f, _ in processed_frames)
            OCCUPANCY_DETECTED = any(detection for _, detection in processed_frames)

            # Update last seen if detected
            if OCCUPANCY_DETECTED:
                OCCUPANCY_LAST_SEEN = time.time()
                if ARGS["HA_TRIGGER"] and not HA_TOGGLE:
                    HA_TOGGLE = True
                    app.integrations.home_assistant.ha_trigger_boolean(True)

            # If timeout has passed since last detection, turn off
            if ARGS["HA_TRIGGER"] and HA_TOGGLE:
                if time.time() - OCCUPANCY_LAST_SEEN > OCCUPANCY_DETECTED_TIMEOUT:
                    HA_TOGGLE = False
                    app.integrations.home_assistant.ha_trigger_boolean(False)
        else:
            processed_frames = frames
            processed_frames_bytes = b"".join(f.tobytes() for f in processed_frames)


        if ARGS["DETECTION"]:
            # Loop all frames
            for video_frame, PERSON_DETECTED in processed_frames:
                if (
                    PERSON_DETECTED
                    and (time.time() - start_timeout) > app.utils.config.CONFIG["TIMEOUT"]
                ):
                    now = datetime.now()
                    minute = now.minute
                    second = now.second

                    SAVE_IMAGE_NAME = (
                        f"{app.utils.config.CONFIG['VIDEO_NAME']}"
                        f"_{minute}"
                        f":{second}"
                        f".jpeg"
                    )
                    SAVE_IMAGE = f"{SAVE_IMAGE_PATH}/{SAVE_IMAGE_NAME}"
                    rc = cv2.imwrite(SAVE_IMAGE, video_frame)
                    if rc:
                        app.utils.logger.pprint(f"Saved image to {SAVE_IMAGE}")
                    else:
                        app.utils.logger.eprint(f"Failed to save image to {SAVE_IMAGE}")

                    if ARGS["SEND_NTFY"]:
                        try:
                            app.integrations.ntfy.send_ntfy(
                                app.utils.config.CONFIG["NTFY_URL"],
                                app.utils.config.CONFIG["NTFY_TAG"],
                                "Person detected",
                                "",
                                SAVE_IMAGE,
                                "detection.jpeg",
                            )
                            app.utils.logger.pprint("Successfully sent ntfy")
                        except requests.exceptions.HTTPError:
                            app.utils.logger.eprint("Failed to send ntfy")

                    start_timeout = time.time()

    # Stop reader
    STOP_EVENT.set()
    stream_reader_thread.join(timeout=2)
    cap.release()
