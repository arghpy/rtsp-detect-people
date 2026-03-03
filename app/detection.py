import cv2
import os

# pylint: disable=import-error
from ultralytics import YOLO

from config import CONFIG
from core import eprint

CUDA_ENABLED = False

model = YOLO("yolo26m.pt")
try:
    model.to("cuda")  # Enable GPU
    CUDA_ENABLED = True
except Exception as e:
    eprint(f"[ERROR] Failed to initialize YOLO model with nvidia: {e}")
    eprint("Continuing with cpu detection.")


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def process_frame(rtsp_url):
    """Process frame with yolo model"""
    person_detected = False

    # half=True - Enable FP16 for faster inference
    results = model(
        source=rtsp_url,
        show=False,
        conf=CONFIG["CONFIDENCE_MIN"],
        verbose=False,
        half=True,
        stream=True,
        rect=True,
        classes=[0],  # Persons
    )
    frame = None

    for result in results:
        frame = result.orig_img
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person: {confidence*100:.2f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        yield frame, person_detected
