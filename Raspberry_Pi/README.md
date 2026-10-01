# Raspberry Pi

This directory contains the Raspberry Pi BLE display app and a `systemd` service unit.

- App script: `ble_text_display.py`
- Icon font: `fa-solid-900.ttf` (Font Awesome 5 Free Solid, used for SFX chip icons)
- Service file: `ble_text_display.service`

## Python App:

- Runs a fullscreen `pygame` overlay for captions.
- Exposes a BLE peripheral named `CGPI`.
- Uses one BLE service and two write characteristics:
  - Caption text UUID: `6E400002-B5A3-F393-E0A9-E50E24DCCA9E`
  - Sound effect UUID: `6E400003-B5A3-F393-E0A9-E50E24DCCA9E`
- Sound effect behavior:
  - Sound effects are rendered as rounded chips with a category icon and label.
  - The payload can be the server's JSON shape (`{"labels": [{"label": ..., "category": ...}, ...]}` or a bare JSON list), or plain comma-separated text.
  - `Silence` and `Speech` labels (case-insensitive) are ignored and clear any visible chips.
  - Chips disappear automatically after 3.5 seconds.
- Displayed text is capped to 150 characters.

## Pi Setup

1. Update packages and install system dependencies:
```bash
sudo apt update
sudo apt install -y python3-pygame python3-gi python3-dbus python3-venv
```

2. Create and prepare virtual environment:
```bash
cd /home/monkey
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install bluezero pygame
```

3. Copy the app script and icon font to the expected runtime path:
```bash
cp /path/to/repo/Raspberry_Pi/ble_text_display.py /home/monkey/ble_text_display.py
cp /path/to/repo/Raspberry_Pi/fa-solid-900.ttf /home/monkey/fa-solid-900.ttf
```

## Run Manually (Dev/Test)

```bash
cd /home/monkey
source .venv/bin/activate
python ble_text_display.py
```

## Install As a systemd Service

The included unit file expects:
- User: `monkey`
- Working directory: `/home/monkey`
- Python: `/home/monkey/.venv/bin/python`
- Script: `/home/monkey/ble_text_display.py`

1. Install the unit file:
```bash
sudo cp /path/to/repo/Raspberry_Pi/ble_text_display.service /etc/systemd/system/ble_text_display.service
```

2. Reload systemd and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ble_text_display.service
sudo systemctl restart ble_text_display.service
```

3. Check service status/logs:
```bash
sudo systemctl status ble_text_display.service
journalctl -u ble_text_display.service -f
```

If your Pi username/home path differs from `monkey`/`/home/monkey`, edit `User=`, `WorkingDirectory=`, and `ExecStart=` in `ble_text_display.service` before installing.

## Local Pi Development With Pi Connect

Enable:
```bash
rpi-connect on
rpi-connect signin
```

Disable:
```bash
rpi-connect off
```
