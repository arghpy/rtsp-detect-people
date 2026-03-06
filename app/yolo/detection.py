# pylint: disable=import-error
import cv2
from ultralytics import YOLO

import app.utils.config
import app.utils.logger

CUDA_ENABLED = False
model = None


def load_model():
    global model, CUDA_ENABLED

    model = YOLO(app.utils.config.CONFIG["YOLO_MODEL"])
    try:
        model.to("cuda")  # Enable GPU
        CUDA_ENABLED = True
        app.utils.logger.pprint("CUDA found. Running on GPU")
    except Exception as e:
        app.utils.logger.eprint(f"[ERROR] Failed to initialize YOLO model with nvidia: {e}")
        app.utils.logger.eprint("Continuing with cpu detection.")


def process_frames(frames):
    """Process frame with yolo model"""
    # half=True - Enable FP16 for faster inference
    results = model(
        source=frames, conf=app.utils.config.CONFIG["CONFIDENCE_MIN"], verbose=False, half=True, imgsz=app.utils.config.CONFIG['YOLO_IMGSZ']
    )
    output = []

    for frame, result in zip(frames, results):
        person_detected = False
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            if model.names[cls] == "person":
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
        output.append((frame, person_detected))
    return output
