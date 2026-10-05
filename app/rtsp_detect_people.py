#!/usr/bin/env python3
"""Detect people from an RTSP stream"""
import app.utils.config
import app.utils.help
import app.utils.logger
import app.utils.video
import app.integrations.ntfy
import app.integrations.home_assistant
import app.integrations.mediamtx
import app.yolo.detection
import cv2
import os
import queue
import requests
import sys
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


# Args
ARGS = {}
ARGS["CAMERA"]= False
ARGS["CAMERA_ARG"]= None
ARGS["CONFIG"]= False
ARGS["CONFIG_ARG"]= None
ARGS["HA_TRIGGER"] = False
ARGS["NTFY_TAG"]= False
ARGS["NTFY_TAG_ARG"]= None


def parse_arguments(argv):
    """Parse command line arguments"""
    global ARGS
    passed_args = argv[1:]

    while len(passed_args) > 0:
        if passed_args[0] == "--help":
            app.utils.help.usage(argv)
            sys.exit(0)
        elif passed_args[0] == "--config":
            ARGS["CONFIG"] = True
            passed_args.pop(0)
            ARGS["CONFIG_ARG"] = str(passed_args[0])
        elif passed_args[0] == "--ntfy-tag":
            ARGS["NTFY_TAG"] = True
            passed_args.pop(0)
            ARGS["NTFY_TAG_ARG"] = str(passed_args[0])
        elif passed_args[0] == "--camera":
            ARGS["CAMERA"] = True
            passed_args.pop(0)
            ARGS["CAMERA_ARG"] = str(passed_args[0])
        elif passed_args[0] == "--ha-trigger":
            ARGS["HA_TRIGGER"] = True
        elif passed_args[0] == "--classes":
            ARGS["CLASSES"] = True
            passed_args.pop(0)
            ARGS["CLASSES_ARG"] = str(passed_args[0])
        else:
            app.utils.logger.eprint(f"Invalid option: {passed_args[0]}")
            app.utils.help.usage(argv)
            sys.exit(0)
        passed_args.pop(0)

    if not ARGS["CONFIG"] or ARGS["CONFIG_ARG"] is None:
        app.utils.logger.eprint("configuration not specified.")
        app.utils.help.usage(sys.argv)
        sys.exit(1)

    if not ARGS["NTFY_TAG"] or ARGS["NTFY_TAG_ARG"] is None:
        app.utils.logger.eprint("ntfy tag not specified")
        app.utils.help.usage(sys.argv)
        sys.exit(1)

    if not ARGS["CAMERA"] or ARGS["CAMERA_ARG"] is None:
        app.utils.logger.eprint("camera not specified")
        app.utils.help.usage(sys.argv)
        sys.exit(1)

    if not ARGS["HA_TRIGGER"]:
        app.utils.logger.eprint("ha-trigger not specified")
        app.utils.help.usage(sys.argv)
        sys.exit(1)


if __name__ == "__main__":
    parse_arguments(sys.argv)
    app.utils.config.process_configuration(ARGS["CONFIG_ARG"])

    # pylint: disable=invalid-name
    start_timeout = 0
    app.yolo.detection.load_model()
    OCCUPANCY_DETECTED_TIMEOUT = 10  # secs
    OCCUPANCY_LAST_SEEN = 0  # timestamp of last detection
    HA_TOGGLE = False

    # Frame and properties
    mediamtx_rtsp_url = f"rtsp://mediamtx:8554/{ARGS['CAMERA_ARG']}"
    video_fps = app.utils.video.probe_stream(mediamtx_rtsp_url)
    if video_fps < 10 or video_fps > 50:
        video_fps = 20

    # if it doesn't exist in config, default value will be used
    MAX_BATCH_SIZE = app.utils.config.CONFIG["YOLO_BATCH"]
    QUEUE_SIZE = max(2, int(MAX_BATCH_SIZE * 1.5))
    FRAME_QUEUE = queue.Queue(maxsize=int(QUEUE_SIZE))
    batch_timeout = min(MAX_BATCH_SIZE/video_fps, 0.2)  # calculate the ideal time to wait for MAX_BATCH_SIZE frames

    # Open the stream
    cap = cv2.VideoCapture(mediamtx_rtsp_url)
    while not cap.isOpened():
        app.utils.logger.eprint(f"Could not read from {mediamtx_rtsp_url}")
        cap.open(mediamtx_rtsp_url)

    stream_reader_thread = threading.Thread(
        target=app.utils.video.collect_frames,
        args=(cap, mediamtx_rtsp_url, FRAME_QUEUE),
        daemon=True,
    )
    stream_reader_thread.start()


    # Create directory structure
    now = datetime.now(ZoneInfo("Europe/Bucharest"))
    next_now = now + timedelta(hours=1)

    base_video_path = app.utils.config.CONFIG["VIDEO_PATH"]
    now_video_path = (
        f"{base_video_path}"
        f"{now.strftime('/%Y/%m/%d/%H')}"
    )
    next_video_path = (
        f"{base_video_path}"
        f"{next_now.strftime('/%Y/%m/%d/%H')}"
    )

    SAVE_IMAGE_PATH = f"{now_video_path}/captures"
    NEXT_SAVE_IMAGE_PATH = f"{next_video_path}/captures"
    os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)
    os.makedirs(NEXT_SAVE_IMAGE_PATH, exist_ok=True)


    # MAIN LOOP
    while True:
        if datetime.now(ZoneInfo("Europe/Bucharest")).hour == next_now.hour:
            # Create directory structure
            now = datetime.now(ZoneInfo("Europe/Bucharest"))
            next_now = now + timedelta(hours=1)
            prev_now = now - timedelta(hours=1)

            now_video_path = (
                f"{base_video_path}"
                f"{prev_now.strftime('/%Y/%m/%d/%H')}"
            )
            next_video_path = (
                f"{base_video_path}"
                f"{next_now.strftime('/%Y/%m/%d/%H')}"
            )

            SAVE_IMAGE_PATH = f"{now_video_path}/captures"
            NEXT_SAVE_IMAGE_PATH = f"{next_video_path}/captures"
            os.makedirs(NEXT_SAVE_IMAGE_PATH, exist_ok=True)

            # A date object is immutable; all operations produce a new object
            start = prev_now.replace(minute=0, second=0, microsecond=0).isoformat()
            end = now.replace(minute=0, second=0, microsecond=0).isoformat()
            download_hour_recording = threading.Thread(
                target=app.integrations.mediamtx.download_recording,
                args=(ARGS['CAMERA_ARG'], start, end, f"{now_video_path}/{ARGS['CAMERA_ARG']}.mp4"),
                daemon=True,
            )
            download_hour_recording.start()


        # Possible busy loop?
        frames = []
        while len(frames) < MAX_BATCH_SIZE:
            try:
                frame = FRAME_QUEUE.get_nowait()
                frames.append(frame)
            except queue.Empty:
                if len(frames) > 2:
                    break
                continue

        if (time.time() - start_timeout) > app.utils.config.CONFIG["TIMEOUT"]:
            classes = ARGS['CLASSES_ARG'].split(',')
            processed_frames = app.yolo.detection.process_frames(frames, classes)
            if len(processed_frames) > 0:
                start_timeout = time.time()
                video_frame = processed_frames[-1]

                # Update last seen if detected
                OCCUPANCY_LAST_SEEN = time.time()
                if not HA_TOGGLE:
                    HA_TOGGLE = True
                    app.integrations.home_assistant.ha_trigger_boolean(True)
                    app.utils.logger.iprint("Home assistant trigger: ON")

                now = datetime.now(ZoneInfo("Europe/Bucharest"))
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
                    app.utils.logger.iprint(f"Saved image to {SAVE_IMAGE}")
                    try:
                        # Sent on the docker network to container
                        app.integrations.ntfy.send_ntfy(
                            "http://ntfy", ARGS["NTFY_TAG_ARG"],
                            "Object detected", "",
                            SAVE_IMAGE, "detection.jpeg",
                        )
                        app.utils.logger.iprint("Successfully sent ntfy")
                    except requests.exceptions.HTTPError:
                        app.utils.logger.eprint("Failed to send ntfy")
                else:
                    app.utils.logger.eprint(f"Failed to save image to {SAVE_IMAGE}")
            else:
                if HA_TOGGLE and (time.time() - OCCUPANCY_LAST_SEEN > OCCUPANCY_DETECTED_TIMEOUT):
                    HA_TOGGLE = False
                    app.integrations.home_assistant.ha_trigger_boolean(False)
                    app.utils.logger.iprint("Home assistant trigger: OFF")
