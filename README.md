# rtsp-detect-people

Detect people from an RTSP stream using YOLOv8n model.

## Requirements

A standard installation of `python3` should contain almost all libraries used in the program.
The following need to be additionally installed (as packages or via pip):
- `opencv` for `cv2` python library
- `ultralytics`

Or, you can use the docker image to run the program, without the need to install the libraries.

## Installation

After satisfying the requirements, simply run `rtsp_detect_people.py` with a configuration file.

## Configuration

An example configuration file [can be found here](config.json).

For people who would like to send an email via Gmail, the following is required:
- create an [App Password](https://myaccount.google.com/apppasswords) (in my tests, leaving the spaces in between worked also)
- server: "smpt.gmail.com"
- port: 465

## Running

```bash
rtsp_detect_people.py -c/--config FILE [-h/--help] [-d/--display] [-s/--save] [-e/--email] [-w/--web PORT]

DESCRIPTION
       Detect people from RTSP stream.

OPTIONS

-c/--config FILE,
       specify configuration file

-h/--help,
       print this help message

-d/--display,
       view footage live

-s/--save,
       save live footage

-e/--email,
       send email

-w/--web PORT,
       Start web server on port
```

Options:
- **-h/--help**: print the help message
- **-c/--config FILE**: mandatory
- **-d/--display**: requires a display to run on
- **-s/--save**: save captured video with the name and path specified in the configuration file,
in the form *path/year/month/day/hour/video_name_year-month-day-hour-minute-second.mkv*
- **-e/--email**: send email

## Notes

If using docker, don't forget to pass the ports between the container and host, in order to be able
to view the live stream.

If you have a CUDA capable gpu, use this docker compose file:
```yaml
services:
  rtsp_detect:
    build: .
    user: 1000:1000
    restart: unless-stopped
    volumes:
      - ./:/usr/src/app
    working_dir: /usr/src/app
    ports:
      - 5000:5000
    command:
      [
        "python3",
        "rtsp_detect_people.py",
        "--config",
        "config.json",
        "--save",
        "--email",
        "--detect",
        "--web", "5000"
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

If you wish to run this as a systemd service:

```ini
[Unit]
Description=Save and detect people on security camera
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
EnvironmentFile=/etc/sysconfig/surveillance/camera_front/service_args.conf
ExecStart=/usr/local/bin/rtsp_detect_people.py $ARGS
Restart=on-failure
RestartSec=5
KillSignal=SIGINT

[Install]
WantedBy=multi-user.target
```

Where the EnvironmentFile contains:

```bash
#######################################################
# Define ARGS for camera_front_save_and_detect.service
#######################################################

ARGS="--config /etc/sysconfig/surveillance/camera_front/rtsp_detect_people/config.json --save --email"
```
