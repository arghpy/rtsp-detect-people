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

import app.integrations.email
import app.integrations.home_assistant
import app.integrations.ntfy
import app.utils.config
import app.utils.help
import app.utils.logger
import app.utils.video
import app.yolo.detection

# Args
ARGS = {}
ARGS["CONFIGURATION_FILE"] = None
ARGS["ENABLE_WEB"] = False
ARGS["HA_TRIGGER"] = False
ARGS["SAVE_VIDEO"] = False
ARGS["SEND_EMAIL"] = False
ARGS["SEND_NTFY"] = False
ARGS["SHOW_DISPLAY"] = False
ARGS["STREAM_PATH"] = None


# pylint: disable=unused-argument
def handle_signals(signum, exec_frame):
    """Respond to different signals"""
    global STOP_EVENT

    signame = signal.Signals(signum).name
    app.utils.logger.pprint(f"Received {signame}({signum})")

    # Release and close threading
    executor.shutdown(wait=True)

    # Stop reader
    STOP_EVENT.set()
    stream_reader_thread.join(timeout=2)

    # Stop MediaMTX
    if ARGS["ENABLE_WEB"]:
        MEDIAMTX_WRITER.stdin.close()
        MEDIAMTX_WRITER.wait()

    # Stop writer
    if ARGS["SAVE_VIDEO"]:
        OUT_VIDEO_WRITER.stdin.close()
        OUT_VIDEO_WRITER.wait()

    # Destroy window if display was set
    if ARGS["SHOW_DISPLAY"]:
        cv2.destroyAllWindows()

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
            ARGS["STREAM_PATH"] = passed_args[0]
        elif passed_args[0] == "--ha-trigger":
            ARGS["HA_TRIGGER"] = True
        else:
            app.utils.logger.eprint(f"Invalid option: {passed_args[0]}")
            app.utils.help.usage(argv)
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
        app.utils.logger.eprint("Configuration not specified.")
        app.utils.help.usage(sys.argv)
        sys.exit(1)

    app.utils.config.process_configuration(ARGS["CONFIGURATION_FILE"])

    if (
        app.utils.config.CONFIG["VIDEO_NAME"] is None
        or app.utils.config.CONFIG["VIDEO_PATH"] is None
        or app.utils.config.CONFIG["VIDEO_FPS"] is None
    ):
        ARGS["SAVE_VIDEO"] = False

    if (
        app.utils.config.CONFIG["EMAIL_SUBJECT"] is None
        or app.utils.config.CONFIG["EMAIL_FROM"] is None
        or app.utils.config.CONFIG["EMAIL_TO"] is None
        or app.utils.config.CONFIG["EMAIL_SERVER"] is None
        or app.utils.config.CONFIG["EMAIL_PORT"] is None
        or app.utils.config.CONFIG["EMAIL_PASSWORD"] is None
    ):
        ARGS["SEND_EMAIL"] = False

    if (
        app.utils.config.CONFIG["NTFY_URL"] is None
        or app.utils.config.CONFIG["NTFY_TAG"] is None
    ):
        ARGS["SEND_NTFY"] = False

    if (
        app.utils.config.CONFIG["HA_ENTITY_ID"] is None
        or app.utils.config.CONFIG["HA_ENTITY_TYPE"] is None
    ):
        ARGS["HA_TRIGGER"] = False

    app.yolo.detection.load_model()

    PERSON_DETECTED = False
    OCCUPANCY_DETECTED = False
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

    MAX_BATCH_SIZE = app.utils.config.CONFIG["YOLO_BATCH"]

    QUEUE_SIZE = video_fps
    if 3 * MAX_BATCH_SIZE < video_fps:
        QUEUE_SIZE = 3 * MAX_BATCH_SIZE

    FRAME_QUEUE = queue.Queue(maxsize=int(QUEUE_SIZE))
    stream_reader_thread = threading.Thread(
        target=app.utils.video.reader_frames_thread,
        args=(
            FRAME_QUEUE,
            video_width,
            video_height,
            video_fps,
            app.utils.config.CONFIG["RTSP_URL"],
            STOP_EVENT,
        ),
        daemon=True,
    )
    stream_reader_thread.start()

    if ARGS["ENABLE_WEB"]:
        MEDIAMTX_WRITER = app.utils.video.mediamtx_stream(
            video_width, video_height, video_fps, ARGS['STREAM_PATH']
        )

    if ARGS["SAVE_VIDEO"]:
        output_video_path = app.utils.config.CONFIG["VIDEO_PATH"]
        output_video_path = (
            f"{output_video_path}" f"/{year}" f"/{month}" f"/{day}" f"/{hour}"
        )
        output_video_format = "mkv"
        output_video_name = (
            f"{app.utils.config.CONFIG['VIDEO_NAME']}_{year}"
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

        OUT_VIDEO_WRITER = app.utils.video.writer_stream(
            output_video, video_width, video_height, video_fps
        )

    OCCUPANCY_DETECTED_TIMEOUT = 10  # secs
    OCCUPANCY_LAST_SEEN = 0  # timestamp of last detection

    # MAIN LOOP
    while True:
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

        processed_frames = app.yolo.detection.process_frames(frames)
        processed_frames_bytes = b"".join(f.tobytes() for f, _ in processed_frames)

        # Check if any detection is true
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

        # Send to MediaMTX
        if ARGS["ENABLE_WEB"] and MEDIAMTX_WRITER is not None:
            MEDIAMTX_WRITER.stdin.write(processed_frames_bytes)

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

                output_video_path = app.utils.config.CONFIG["VIDEO_PATH"]
                output_video_path = (
                    f"{output_video_path}"
                    f"/{year}"
                    f"/{month}"
                    f"/{day}"
                    f"/{hour}"
                )

                output_video_name = (
                    f"{app.utils.config.CONFIG['VIDEO_NAME']}_{year}"
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

                OUT_VIDEO_WRITER = app.utils.video.writer_stream(
                    output_video, video_width, video_height, video_fps
                )
            OUT_VIDEO_WRITER.stdin.write(processed_frames_bytes)

        # Loop all frames
        for video_frame, PERSON_DETECTED in processed_frames:
            # Send email
            if ARGS["SEND_EMAIL"]:
                if email_sent and email_future is not None:
                    if email_future.done():
                        app.utils.logger.pprint("Email sent")
                        email_sent = False
                        email_future = None

            if (
                PERSON_DETECTED
                and (time.time() - start_timeout) > app.utils.config.CONFIG["TIMEOUT"]
            ):
                now = datetime.now()
                minute = now.minute
                second = now.second

                SAVE_IMAGE_PATH = f"{output_video_path}/captures"
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

                if ARGS["SEND_EMAIL"]:
                    email_future = executor.submit(
                        app.integrations.email.send_email_report,
                        SAVE_IMAGE,
                    )
                    email_sent = True

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

            # Show display
            if ARGS["SHOW_DISPLAY"]:
                cv2.imshow(app.utils.config.CONFIG["RTSP_FEED"], video_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    # Release and close threading
    executor.shutdown(wait=True)

    # Stop reader
    STOP_EVENT.set()
    stream_reader_thread.join(timeout=2)

    # Stop MediaMTX
    if ARGS["ENABLE_WEB"]:
        MEDIAMTX_WRITER.stdin.close()
        MEDIAMTX_WRITER.wait()

    # Stop writer
    if ARGS["SAVE_VIDEO"]:
        OUT_VIDEO_WRITER.stdin.close()
        OUT_VIDEO_WRITER.wait()

    # Destroy window if display was set
    if ARGS["SHOW_DISPLAY"]:
        cv2.destroyAllWindows()
