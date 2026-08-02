import queue
import subprocess
import time

import cv2
import numpy as np
import app.utils.logger
import app.yolo.cuda


def collect_frames(cap, rtsp_url, frame_queue, STOP_EVENT):
    failed_frames = 0
    while not STOP_EVENT.is_set():
        ret, frame = cap.read()
        if failed_frames > 20:
            app.utils.logger.eprint("Restarting cap")
            while not cap.isOpened():
                app.utils.logger.eprint(f"Could not read from {rtsp_url}")
                cap.open(rtsp_url)
            failed_frames = 0

        if not ret:
            app.utils.logger.eprint("Could not read frame.")
            failed_frames += 1
            time.sleep(1)
            continue
        else:
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                # drop frame if queue full
                pass


def probe_stream(rtsp_url) -> tuple[int, int, int]:
    """Probe the stream to get data"""
    while True:
        app.utils.logger.pprint("Probing stream info")
        # Open stream once to get video properties
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            app.utils.logger.eprint("Could not open RTSP stream")
            time.sleep(1)
            cap.release()
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()
        break

    app.utils.logger.pprint(f"Stream resolution: {width}x{height}, FPS: {fps}")
    return width, height, fps
