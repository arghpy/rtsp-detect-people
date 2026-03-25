import queue
import subprocess
import time

import cv2
import numpy as np
import app.utils.logger
import app.yolo.detection


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


def reader_stream(rtsp_url, fps) -> subprocess.Popen:
    """Continuously get frames from stream"""
    app.utils.logger.pprint("Starting ffmpeg reader")

    reader_cmd = [
        "ffmpeg",
    ]
    if app.yolo.detection.CUDA_ENABLED:
        reader_cmd.extend(["-hwaccel", "cuda"])
    reader_cmd.extend(
        [
            "-rtsp_transport",
            "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
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
    reader = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE, bufsize=0)
    try:
        import fcntl
        fcntl.fcntl(reader.stdout.fileno(), fcntl.F_SETPIPE_SZ, 1_000_000)
    except Exception as e:
        app.utils.logger.eprint(f"Could not increase read buffer size: {e}")
        pass

    return reader


def terminate_pipe_process(pipe: subprocess.Popen):
    """Safely terminate the pipe"""
    wait_timeout = 5
    pipe.stdout.close()
    pipe.terminate()

    try:
        pipe.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        app.utils.logger.eprint(f"Waited for {wait_timeout}. Killing process")
        pipe.kill()


def reconnect_pipe_process(pipe: subprocess.Popen, rtsp_url, fps):
    """Safely reconnect to stream, retry until ffmpeg is alive."""
    terminate_pipe_process(pipe)

    attempt = 0
    while True:
        attempt += 1
        app.utils.logger.eprint(f"Reconnecting attempt #{attempt}...")

        new_pipe = reader_stream(rtsp_url, fps)
        time.sleep(1.0)  # give ffmpeg a moment to start

        if new_pipe and new_pipe.poll() is None and new_pipe.stdout:
            app.utils.logger.pprint("Successfully reconnected")
            return new_pipe

        app.utils.logger.eprint("ffmpeg failed to start or exited immediately")
        terminate_pipe_process(new_pipe)
        time.sleep(2)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def reader_frames_thread(frame_queue, width, height, fps, rtsp_url, stop_event):
    """Continuously add frames in queue to be processed"""
    app.utils.logger.pprint("Reader thread started")

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
            app.utils.logger.eprint(f"Exception reading frame: {e}")
            dropped_frames += 1
            continue

        if frame is None:
            dropped_frames += 1

            if dropped_frames >= fps * 2:
                app.utils.logger.eprint(f"{dropped_frames} consecutive frames missing. Reconnecting")
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
