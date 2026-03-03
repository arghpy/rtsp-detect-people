import select
import cv2
import numpy as np
import time
import queue
import subprocess
from detection import CUDA_ENABLED
from core import pprint, eprint


def writer_stream(video_path, width, height, fps) -> subprocess.Popen:
    """Write stream to file"""
    pprint(f"Saving video to {video_path}")

    writer_cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",  # no audio
    ]
    if CUDA_ENABLED:
        writer_cmd.extend(["-c:v", "h264_nvenc", "-preset", "llhp"])
    else:
        writer_cmd.extend(["-c:v", "libx264", "-preset", "veryfast"])

    writer_cmd.extend(
        [
            "-g",
            f"{fps*2}",  # keyframe every 2 seconds
            "-x264-params",
            f"keyint={fps*2}:min-keyint={fps}",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
            video_path,
        ]
    )
    # pylint: disable=consider-using-with
    writer = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE)
    return writer


def probe_stream(rtsp_url) -> tuple[int, int, int]:
    """Probe the stream to get data"""

    while True:
        pprint("Probing stream info")
        # Open stream once to get video properties
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            eprint("Could not open RTSP stream")
            time.sleep(1)
            cap.release()
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()
        break

    # Scale all streams to 1920 width
    target_width = 1920
    target_height = int(height * target_width / width)

    # Ensure width/height are even for NVENC
    target_width = (target_width // 2) * 2
    target_height = (target_height // 2) * 2

    pprint(f"Target resolution for encoding: {target_width}x{target_height}")

    return target_width, target_height, fps
