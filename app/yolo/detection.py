# pylint: disable=import-error
import cv2
from ultralytics import settings

import app.utils.config
import app.utils.logger
import app.yolo.cuda

# Disable analytics and crash reporting
settings.update({"sync": False})

from ultralytics import YOLO

model = None

def load_model():
    global model
    yolo_model = app.utils.config.CONFIG['YOLO_MODEL']
    model = YOLO(yolo_model)

    try:
        model.to("cuda")  # Enable GPU
        app.yolo.cuda.CUDA_ENABLED = True
        app.utils.logger.pprint("CUDA found. Running on GPU")
    except Exception as e:
        app.utils.logger.eprint(f"Failed to initialize YOLO model with nvidia: {e}")
        app.utils.logger.eprint("Continuing with cpu detection.")


def process_frames(frames):
    """Process frame with yolo model only for people"""
    yolo_imgsz = app.utils.config.CONFIG['YOLO_IMGSZ']
    yolo_conf  = app.utils.config.CONFIG['CONFIDENCE_MIN']

    results = model(source=frames, conf=yolo_conf, verbose=False, classes=[0], half=True, imgsz=yolo_imgsz)
    output = []

    for frame, result in zip(frames, results):
        boxes = result.boxes
        if len(boxes) == 0:
            continue
        for box in boxes:
            confidence = float(box.conf[0])
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
        output.append(frame)
    return output
