# rtsp-detect-people

[![Super-Linter](https://github.com/arghpy/rtsp-detect-people/actions/workflows/manage_pull_requests.yaml/badge.svg)](https://github.com/marketplace/actions/super-linter)

Detect people from an RTSP stream.

## Requirements

A standard installation of `python3` should contain almost all libraries used in the program.
The following need to be additionally installed (as packages or via pip):
- `opencv` for `cv2` python library
- `numpy`

## Installation

After satisfying the requirements, simply run `rtsp_detect_people.py` with a configuration file.

## Configuration

An example configuration file [can be found here](config.json).

The MobileNetSSD model and weights [can be found here](https://automaticaddison.com/how-to-detect-objects-in-video-using-mobilenet-ssd-in-opencv/).

If a more updated or reliable source is found, please create an issue about this.

For people who would like to send an email via Gmail, the following is required:
- create an [App Password](https://myaccount.google.com/apppasswords) (in my tests, leaving the spaces in between worked also)
- server: "smpt.gmail.com"
- port: 465

## Running

```bash
./rtsp_detect_people.py -c/--config FILE [-h/--help] [-d/--display] [-s/--save] [-e/--email]

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
```

Options:
- **-h/--help**: print the help message
- **-c/--config FILE**: mandatory
- **-d/--display**: requires a display to run on
- **-s/--save**: save captured video with the name and path specified in the configuration file,
in the form *path/year/month/day/hour/video_name_year-month-day-hour-minute-second.mkv*
- **-e/--email**: send email

## Notes

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
