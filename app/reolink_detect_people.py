#!/usr/bin/env python3
"""Detect people from an RTSP stream"""
import app.utils.config
import app.utils.help
import app.utils.logger
import app.utils.video
import cv2
import os
import queue
import requests
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta


# Args
ARGS = {}
ARGS["CONFIGURATION_FILE"] = None
ARGS["CAMERA"] = False
ARGS["HA_TRIGGER"] = False
ARGS["SEND_NTFY"] = False
ARGS["DETECTION"] = False
ARGS["CAMERA_PATH"] = None
ARGS["WEBHOOK"] = False
ARGS["WEBHOOK_PORT"] = None


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
        elif passed_args[0] == "--webhook-port":
            ARGS["WEBHOOK"] = True
            passed_args.pop(0)
            ARGS["WEBHOOK_PORT"] = int(passed_args[0])
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

    if ARGS["CONFIGURATION_FILE"] is None:
        app.utils.logger.eprint("Configuration not specified.")
        app.utils.help.usage(sys.argv)
        sys.exit(1)

    app.utils.config.process_configuration(ARGS["CONFIGURATION_FILE"])

    if (app.utils.config.CONFIG["NTFY_URL"] is None
        or app.utils.config.CONFIG["NTFY_TAG"] is None):
        ARGS["SEND_NTFY"] = False
    else:
        import app.integrations.ntfy

    if (app.utils.config.CONFIG["HA_ENTITY_ID"] is None
        or app.utils.config.CONFIG["HA_ENTITY_TYPE"] is None):
        ARGS["HA_TRIGGER"] = False
    else:
        import app.integrations.home_assistant

    if ARGS["DETECTION"] and ARGS["WEBHOOK"] and ARGS["WEBHOOK_PORT"] is not None:
        import app.integrations.webhook
        import app.integrations.reolink
        from http.server import HTTPServer

        webhook_reader_thread = threading.Thread(
            target=lambda: HTTPServer(("0.0.0.0", ARGS["WEBHOOK_PORT"]), app.integrations.webhook.Handler).serve_forever(),
            daemon=True
        )
        webhook_reader_thread.start()

        reolink_token, reolink_token_expiration = app.integrations.reolink.login(ARGS["CAMERA_PATH"], app.utils.config.CONFIG["RTSP_USER"], app.utils.config.CONFIG["RTSP_PASS"])
        if not reolink_token:
            app.utils.logger.eprint(f"Could not login to Reolink camera: {ARGS['CAMERA_PATH']}")
            sys.exit(1)

    # Create directory structure
    now = datetime.now()
    next_now = now + timedelta(hours=1)

    base_video_path = app.utils.config.CONFIG["VIDEO_PATH"]
    output_video_path = (
        f"{base_video_path}"
        f"{now.strftime('/%Y/%m/%d/%H')}"
    )
    next_output_video_path = (
        f"{base_video_path}"
        f"{next_now.strftime('/%Y/%m/%d/%H')}"
    )

    SAVE_IMAGE_PATH = f"{output_video_path}/captures"
    NEXT_SAVE_IMAGE_PATH = f"{next_output_video_path}/captures"
    os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)
    os.makedirs(NEXT_SAVE_IMAGE_PATH, exist_ok=True)


    # TODO: initialize token

    # MAIN LOOP
    while True:
        if datetime.now().hour == next_now.hour:
            SAVE_IMAGE_PATH = NEXT_SAVE_IMAGE_PATH
            # Create directory structure
            now = datetime.now()
            next_now = now + timedelta(hours=1)

            next_output_video_path = (
                f"{base_video_path}"
                f"{next_now.strftime('/%Y/%m/%d/%H')}"
            )


            NEXT_SAVE_IMAGE_PATH = f"{next_output_video_path}/captures"
            os.makedirs(NEXT_SAVE_IMAGE_PATH, exist_ok=True)

        if ARGS["DETECTION"] and ARGS["WEBHOOK"] and ARGS["WEBHOOK_PORT"] is not None:
            if (reolink_token_expiration - time.time()) < 100:
                reolink_token, reolink_token_expiration = app.integrations.reolink.login(ARGS["CAMERA_PATH"], app.utils.config.CONFIG["RTSP_USER"], app.utils.config.CONFIG["RTSP_PASS"])
                if not reolink_token:
                    app.utils.logger.eprint(f"Could not log in to Reolink camera: {ARGS['CAMERA_PATH']}")
                    sys.exit(1)

            if app.integrations.webhook.camera_alert():
                # Update last seen if detected
                OCCUPANCY_LAST_SEEN = time.time()
                if ARGS["HA_TRIGGER"] and not HA_TOGGLE:
                    HA_TOGGLE = True
                    app.integrations.home_assistant.ha_trigger_boolean(True)

                # If timeout has passed since last detection, turn off
                if ARGS["HA_TRIGGER"] and HA_TOGGLE:
                    if time.time() - OCCUPANCY_LAST_SEEN > OCCUPANCY_DETECTED_TIMEOUT:
                        HA_TOGGLE = False
                        app.integrations.home_assistant.ha_trigger_boolean(False)

                now = datetime.now()
                minute = now.minute
                second = now.second

                SAVE_IMAGE_NAME = (
                    f"{app.utils.config.CONFIG['VIDEO_NAME']}"
                    f"_{minute}"
                    f"-{second}"
                    f".jpeg"
                )
                SAVE_IMAGE = f"{SAVE_IMAGE_PATH}/{SAVE_IMAGE_NAME}"

                img = app.integrations.reolink.get_snapshot(ARGS['CAMERA_PATH'], reolink_token)
                if img is not None:
                    compressed_img = app.integrations.ntfy.compress_for_ntfy(img)
                    with open(SAVE_IMAGE, "wb") as f:
                        f.write(compressed_img)
                    app.utils.logger.pprint(f"Saved snapshot to {SAVE_IMAGE}")
                else:
                    app.utils.logger.eprint(f"Failed to save snapshot to {SAVE_IMAGE}")

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
