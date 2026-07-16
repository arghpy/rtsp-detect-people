# rtsp-detect-people

Detect people from an RTSP stream using YOLOv8n model.

## Requirements

Use the docker image to run the program, without the need to install the libraries.

## Configuration

An example configuration file [can be found here](config.json).

## Running

```bash
docker compose up -d
```

The program by itself contains the following options:

```bash
python3 -m app.rtsp_detect_people -c/--config FILE [-h/--help] --camera CAMERA

DESCRIPTION
       Detect people from RTSP stream.

OPTIONS

-h/--help,
        print this help message

-n/--ntfy,
        send notification through ntfy

-d/--detection,
        detect people on stream

-c/--config FILE,
        specify configuration file

--camera CAMERA,
        Camera configured in mediaMTX

--ha-trigger,
        Home Assistant: trigger while person detected
```

Options:
- **-h/--help**: print the help message
- **-c/--config FILE**: mandatory

## Viewing

The stream can be seen on **http://IP:8889/PATH**.
It uses the capabilities of [MediaMTX](https://mediamtx.org/) for display the stream with the performant WebRTC.

## Notes

If you have a CUDA capable gpu, use this docker compose file:
```yaml
services:
  mediamtx:
    image: bluenviron/mediamtx:latest
    restart: unless-stopped
    ports:
      - 8554:8554   # RTSP (ffmpeg pushes here)
      - 8889:8889   # WebRTC (browsers connect here)
      - 8189:8189/udp
    volumes:
      - ./mediamtx.yml:/mediamtx.yml
  camera1:
    build: .
    user: 1000:1000
    restart: unless-stopped
    volumes:
      - ./:/usr/src/app
    working_dir: /usr/src/app
    command:
      [
        "python3", "-m", "app.rtsp_detect_people",
        "--config", "configuration-camera1.json",
        "--ntfy",
        "--detection",
        "--camera", "front"
      ]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
  camera2:
    build: .
    user: 1000:1000
    restart: unless-stopped
    volumes:
      - ./:/usr/src/app
    working_dir: /usr/src/app
    command:
      [
        "python3", "-m", "app.rtsp_detect_people",
        "--config", "configuration-camera2.json",
        "--ntfy",
        "--detection",
        "--camera", "back"
      ]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
```

In case the connection to the camera is lost, it will try to reconnect indefinitely.

The timeout set in the configuration file represents the timeout in seconds between notifications sent,
in case there is a person detected continuously for a long period of time.
