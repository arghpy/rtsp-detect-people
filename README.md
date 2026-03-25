# rtsp-detect-people

Detect people from an RTSP stream using YOLOv8n model.

## Requirements

Use the docker image to run the program, without the need to install the libraries.

## Configuration

An example configuration file [can be found here](config.json).

For people who would like to send an email via Gmail, the following is required:
- create an [App Password](https://myaccount.google.com/apppasswords) (in my tests, leaving the spaces in between worked also)
- server: "smpt.gmail.com"
- port: 465

## Running

```bash
docker compose up -d
```

The program by itself contains the following options:

```bash
python3 -m app.rtsp_detect_people -c/--config FILE [-h/--help] [-s/--save] [-e/--email] [-w/--web PORT]

DESCRIPTION
       Detect people from RTSP stream.

OPTIONS

-c/--config FILE,
       specify configuration file

-h/--help,
       print this help message

-s/--save,
       save live footage

-e/--email,
       send email

-w/--web PATH,
       Start web server on path
```

Options:
- **-h/--help**: print the help message
- **-c/--config FILE**: mandatory
- **-s/--save**: save captured video with the name and path specified in the configuration file,
in the form *path/year/month/day/hour/video_name_year-month-day-hour-minute-second.mkv*
- **-e/--email**: send email

## Viewing

The stream can be seen on **http://<IP>:8889/<PATH>**.

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
        "--save",
        "--email",
        "--ntfy",
        "--web", "camera1"
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
        "--save",
        "--email",
        "--ntfy",
        "--web", "camera2"
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

The timeout set in the configuration file represents the timeout in seconds between emails sent,
in case there is a person detected continuously for a long period of time.
