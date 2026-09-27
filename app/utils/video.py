import queue
import time

import cv2
import app.utils.logger
import app.yolo.cuda


def collect_frames(cap, rtsp_url, frame_queue):
    failed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            failed_frames += 1
            app.utils.logger.eprint(
                f"Could not read frame ({failed_frames}/20)"
            )

            if failed_frames >= 20:
                app.utils.logger.iprint("Reconnecting to camera...")
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                if cap.isOpened():
                    app.utils.logger.iprint("Camera reconnected")
                    failed_frames = 0
                else:
                    app.utils.logger.eprint(
                        "Could not reconnect to camera"
                    )
                time.sleep(1)

            continue

        failed_frames = 0
        try:
            frame_queue.put(frame, timeout=1)
        except queue.Full:
            pass


def probe_stream(rtsp_url: str) -> int:
    """Probe the stream to get data"""
    while True:
        app.utils.logger.iprint("Probing stream info")
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

    app.utils.logger.iprint(f"Stream resolution: {width}x{height}, FPS: {fps}")
    return fps
