import time
import os
import subprocess
import requests
import app.utils.logger


def download_file(url: str, params: dict, output_path: str) -> None:
    app.utils.logger.iprint(f"Downloading recording from {url}")

    # Timeout with stream=True means that it should wait
    # only 300s to receive data
    response = requests.get(url, params=params, stream=True, timeout=300)
    response.raise_for_status()

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                file.write(chunk)

    app.utils.logger.iprint(f"Finished downloading recording: {output_path}")


def download_recording(path: str, start: str, end: str, video_path: str) -> None:
    time.sleep(90) # This is done to ensure termination of the 1m video segment
    app.utils.logger.iprint(f"Downloading recording from {start} till {end}")
    response = requests.get("http://mediamtx:9996/list",
                            params={
                                "path": path,
                                "start": start,
                                "end": end,
                            },
                            timeout=300)
    response.raise_for_status()

    recordings = response.json()
    if len(recordings) == 1:
        for index, recording in enumerate(recordings):
            app.utils.logger.iprint(f"Downloading {index+1}/{len(recordings)} recordings")
            download_file(url=recordings[0]["url"], params={"format": "mp4"}, output_path=video_path)
    else:
        temp_dir = "/tmp/mediamtx"
        os.makedirs(temp_dir, exist_ok=True)
        input_files = []
        for index, recording in enumerate(recordings):
            app.utils.logger.iprint(f"Downloading {index+1}/{len(recordings)} recordings")
            temp_recording = f"{temp_dir}/{index+1:04d}.mp4"
            download_file(url=recording["url"], params={"format": "mp4"}, output_path=temp_recording)
            input_files.append(temp_recording)

        # Create FFmpeg concat file
        concat_file = f"{temp_dir}/concat.txt"

        with open(concat_file, "w", encoding="utf-8") as file:
            for input_file in input_files:
                file.write(f"file '{input_file}'\n")

        app.utils.logger.iprint(f"Concatenating {len(input_files)} recordings")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                "-movflags", "+faststart",
                "-y",
                str(video_path),
            ],
            check=True,
        )
        app.utils.logger.iprint(f"Finished downloading recording: {video_path}")
    return recordings
