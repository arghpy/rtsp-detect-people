# pylint: disable=import-error
import cv2
from ultralytics import YOLO
from app.utils.config import CONFIG
from app.utils.logger import eprint, pprint

CUDA_ENABLED = False
model = None


def load_model():
    global model, CUDA_ENABLED

    model = YOLO(CONFIG["MODEL"])
    try:
        model.to("cuda")  # Enable GPU
        CUDA_ENABLED = True
        pprint("CUDA found. Running on GPU")
    except Exception as e:
        eprint(f"[ERROR] Failed to initialize YOLO model with nvidia: {e}")
        eprint("Continuing with cpu detection.")


def process_frame(frame):
    """Process frame with yolo model"""
    person_detected = False

    # half=True - Enable FP16 for faster inference
    results = model(
        source=frame, conf=CONFIG["CONFIDENCE_MIN"], verbose=False, half=True
    )

    for result in results:
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
    return frame, person_detected
