import queue
import subprocess
import time

import cv2
import numpy as np
import app.utils.logger
import app.yolo.detection


def collect_frames(cap, frame_queue, STOP_EVENT):
    while not STOP_EVENT.is_set():
        ret, frame = cap.read()
        if not ret:
            app.utils.logger.eprint("Could not read frame.")
            time.sleep(1)
            continue
        else:
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                # drop frame if queue full
                pass


def mediamtx_stream(width, height, fps, path="live") -> subprocess.Popen:
    """Stream to MediaMTX for WebRTC"""
    stream_cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",
    ]
    if app.yolo.detection.CUDA_ENABLED:
        stream_cmd.extend(["-c:v", "h264_nvenc", "-preset", "llhq"])
    else:
        stream_cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",  # reduces latency for live streaming
            "-x264-params", f"keyint={fps*2}:min-keyint={fps}",
        ])

    stream_cmd.extend([
        "-g", f"{fps*2}",
        "-pix_fmt", "yuv420p",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        "rtsp://mediamtx:8554/" + path,
    ])
    # pylint: disable=consider-using-with
    mediamtx_streamer = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)
    return mediamtx_streamer


def writer_stream(video_path, width, height, fps) -> subprocess.Popen:
    """Write stream to file"""
    app.utils.logger.pprint(f"Saving video to {video_path}")

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
    if app.yolo.detection.CUDA_ENABLED:
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


def read_exact(pipe, size):
    buf = bytearray(size)
    view = memoryview(buf)

    n = 0
    while n < size:
        chunk = pipe.read(size - n)
        if not chunk:
            return None
        view[n:n+len(chunk)] = chunk
        n += len(chunk)

    return buf


def read_frame(pipe: subprocess.Popen, width, height) -> np.ndarray | None:
    """Read frame from reader"""
    size = width * height * 3

    raw = read_exact(pipe, size)
    if raw is None or len(raw) != size:
        return None

    frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))

    # Ensure writable for OpenCV without always copying
    if not frame.flags.writeable:
        frame = np.array(frame, copy=True)  # copy only if necessary

    return frame


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
