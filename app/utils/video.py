import queue
import select
import subprocess
import time

import cv2
import numpy as np
from app.utils.logger import eprint, pprint
from app.yolo.detection import CUDA_ENABLED


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


def read_frame(pipe: subprocess.Popen, width, height) -> np.ndarray | None:
    """Read frame from reader"""
    size = width * height * 3

    # Wait until data is available or timeout expires
    rlist, _, _ = select.select([pipe], [], [], 2)  # wait 2 seconds for data
    if not rlist:
        return None  # no data

    raw = pipe.read(size)
    if raw is None or len(raw) != size:
        return None

    frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
    return frame.copy()


def reader_stream(rtsp_url, fps) -> subprocess.Popen:
    """Continuously get frames from stream"""
    pprint("Starting ffmpeg reader")

    reader_cmd = [
        "ffmpeg",
    ]
    if CUDA_ENABLED:
        reader_cmd.extend(["-hwaccel", "cuda"])
    reader_cmd.extend(
        [
            "-rtsp_transport",
            "tcp",
            "-i",
            rtsp_url,
            "-loglevel",
            "error",
            "-vf",
            f"fps={fps}",
            "-an",
            "-sn",  # disable audio and subs
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
    )

    # pylint: disable=consider-using-with
    reader = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE)
    return reader


def terminate_pipe_process(pipe: subprocess.Popen):
    """Safely terminate the pipe"""
    wait_timeout = 5
    pipe.stdout.close()
    pipe.terminate()

    try:
        pipe.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        eprint(f"Waited for {wait_timeout}. Killing process")
        pipe.kill()


def reconnect_pipe_process(pipe: subprocess.Popen, rtsp_url, fps):
    """Safely reconnect to stream, retry until ffmpeg is alive."""
    terminate_pipe_process(pipe)

    attempt = 0
    while True:
        attempt += 1
        eprint(f"Reconnecting attempt #{attempt}...")

        new_pipe = reader_stream(rtsp_url, fps)
        time.sleep(1.0)  # give ffmpeg a moment to start

        if new_pipe and new_pipe.poll() is None and new_pipe.stdout:
            pprint("Successfully reconnected")
            return new_pipe

        eprint("ffmpeg failed to start or exited immediately")
        terminate_pipe_process(new_pipe)
        time.sleep(2)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def reader_frames_thread(frame_queue, width, height, fps, rtsp_url, stop_event):
    """Continuously add frames in queue to be processed"""
    pprint("Reader thread started")

    pipe = reader_stream(rtsp_url, fps)
    if pipe is None or pipe.returncode is not None:
        stop_event.set()

    dropped_frames = 0

    while not stop_event.is_set():
        frame = None
        try:
            frame = read_frame(pipe.stdout, width, height)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            eprint(f"Exception reading frame: {e}")
            dropped_frames += 1
            continue

        if frame is None:
            dropped_frames += 1

            if dropped_frames >= fps * 2:
                eprint(f"{dropped_frames} consecutive frames missing. Reconnecting")
                pipe = reconnect_pipe_process(pipe, rtsp_url, fps)
                dropped_frames = 0
        else:
            dropped_frames = 0

            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                # Queue full → drop frame to avoid blocking
                pass

    terminate_pipe_process(pipe)


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

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()
        break

    pprint(f"Stream resolution: {width}x{height}, FPS: {fps}")
    return width, height, fps
