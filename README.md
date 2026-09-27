# rtsp-detect-people

[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

View live stream and save audio/video from camera using [MediaMTX](https://mediamtx.org/).
Send a rich push notification via [ntfy](https://ntfy.sh/) with a snapshot when a person is detected by the Reolink camera.

## Requirements

Docker.

## Configuration

### Services

In order to not manually remove video files, the following services and script are provided:
- [timer](services/clean_files.timer)
- [service](services/clean_files.service)
- [script](bin/clean_files.sh)

You need to manually install them:

```bash
sudo cp services/clean_files* /etc/systemd/system/
sudo cp bin/clean_files.sh /usr/local/bin/

sudo systemctl enable clean_files.timer
```

The service will run every 15m.

### Detection

It is meant to store user/password for the camera and information about where to save the snapshots.
An example configuration file [can be found here](app/configuration_camera-front.json).

### MediaMTX

An example configuration file [can be found here](mediamtx/mediamtx.yml).

### ntfy

An example configuration file [can be found here](ntfy/server.yml).
These values were changed:

```yaml
base-url: <URL>
attachment-cache-dir: "/var/cache/ntfy/attachments"
attachment-total-size-limit: "5G"
attachment-file-size-limit: "150M"
attachment-expiry-duration: "24h"
visitor-attachment-total-size-limit: "1G"
visitor-attachment-daily-bandwidth-limit: "5G"
```

## Running

```bash
# Initially
docker compose up -d --build

# Afterwards
docker compose up -d
```

The program by itself contains the following options:

```bash
--help,
	print this help message

--ntfy-tag TAG,
	TAG to which to send notification through ntfy

--config FILE,
	configuration FILE to use

--camera CAMERA,
	CAMERA configured as path in MediaMTX configuration file

--webhook-port PORT,
	PORT to receive push notifications from Reolink camera
```

## Live stream

The stream can be seen on **http://IP:8889/PATH**.
It uses the capabilities of [MediaMTX](https://mediamtx.org/) to display the stream with the performant WebRTC.

## Notifications

Snapshots are saved in a docker volume inside ntfy container.
After configuring the <URL> in its configuration file, they can be viewed at **<URL>/<TAG>** by subscribing.

## Saving audio/video

This is done by mediamtx inside the container.
Videos are saved in the format: **/path/to/recordings/%Y/%m/%d/%H/${MTX_PATH}_%M:%S.mkv**.
