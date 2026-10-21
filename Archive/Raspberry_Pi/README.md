# Raspberry Pi

Raspberry Pi BLE display app and Cage startup files.

- App script: `ble_text_display.py`
- Launcher: `start.sh` activates the Python virtual environment and runs the app inside Cage.
- Service template: `cage@.service` starts the launcher in a local virtual-terminal session.

## Python App:

- Fullscreen `pygame` display with a black background (transparent in glasses) and white captions.
- Initially displays `Waiting for BLE text...`. New captions replace the previous text.
- Exposes a BLE peripheral named `CGPI`.
- Uses the first available Bluetooth adapter and advertises from a background thread.
- Uses BLE service `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` and three UTF-8 write characteristics (each supports write and write without response):
  - Caption text UUID: `6E400002-B5A3-F393-E0A9-E50E24DCCA9E`
  - Sound effect UUID: `6E400003-B5A3-F393-E0A9-E50E24DCCA9E`
  - Position config UUID: `6E400004-B5A3-F393-E0A9-E50E24DCCA9E`
- Sound effect behavior:
  - Blank text, `Silence`, or `Speech` (case-insensitive) clears the sound-effect label.
  - Any other text is shown in blue, in square brackets, above the captions.
- Captions and sound-effect labels are each trimmed and capped to 150 characters before rendering. Text wraps at word boundaries using the configured margin.

## Caption Position BLE Config

Caption placement can be adjusted by writing UTF-8 text to the position config characteristic:

- UUID: `6E400004-B5A3-F393-E0A9-E50E24DCCA9E`
- `x`: horizontal center as a screen ratio. Default: `0.5`
- `y`: vertical center as a screen ratio. Default: `0.83`
- `margin`: left/right wrapping margin in pixels. Default: `40`
- `gap`: sound-effect gap above captions in line-height units. Default: `0.5`

UTF-8 Format:

```text
x=0.5,y=0.75,margin=40,gap=0.5
```

Values can be partial, so `y=0.7` updates only the vertical position.
`x` and `y` are clamped to `0.0`–`1.0`; `margin` is converted to a nonnegative integer and `gap` to a nonnegative float. Unknown keys are ignored and invalid numeric values are logged. Settings are held in memory and reset when the app restarts.

## Pi Setup

These instructions use the Pi user `cg-pi` with home directory `/home/cg-pi`. Run the environment and app-copy commands as that user. The supplied startup files already use these paths.

1. Update packages and install system dependencies:

```bash
sudo apt update
sudo apt install -y cage bluez python3-pygame python3-gi python3-dbus python3-venv libpam-systemd
sudo systemctl enable --now bluetooth.service
```

2. Create a virtual environment that can use the system-installed Pygame, GI, and D-Bus bindings:

```bash
cd /home/cg-pi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install bluezero
```

3. Copy the app and launcher to their runtime paths (replace `/path/to/repo` with your checkout):

```bash
cp /path/to/repo/Raspberry_Pi/ble_text_display.py /home/cg-pi/ble_text_display.py
cp /path/to/repo/Raspberry_Pi/start.sh /home/cg-pi/start.sh
chmod +x /home/cg-pi/start.sh
```

## Run Manually (Dev/Test)

From a local console logged in as `cg-pi`, launch the same Cage session used by the service:

```bash
/home/cg-pi/start.sh
```

Stop `cage@tty1.service` first if it is already running. For testing within an existing graphical session, run `/home/cg-pi/.venv/bin/python /home/cg-pi/ble_text_display.py` directly.

## Install As a systemd Service

With the configuration below, the template runs as `cg-pi`, uses `PAMName=cage` to open a login session, and launches `/home/cg-pi/start.sh`. The launcher activates `/home/cg-pi/.venv` and runs `cage python /home/cg-pi/ble_text_display.py`. The service restarts automatically whenever the launcher exits.

1. If migrating from the old service, disable it before starting Cage:

```bash
sudo systemctl disable --now ble_text_display.service
```

2. Install the template:

```bash
sudo cp /path/to/repo/Raspberry_Pi/cage@.service /etc/systemd/system/cage@.service
```

3. Create `/etc/pam.d/cage` using `sudoedit /etc/pam.d/cage` with the following minimal configuration from the [Cage systemd setup guide](https://github.com/cage-kiosk/cage/wiki/Starting-Cage-on-boot-with-systemd). Cage requires wlroots with systemd-logind support for this setup.

```text
auth           required        pam_unix.so nullok
account        required        pam_unix.so
session        required        pam_unix.so
session        required        pam_systemd.so
```

4. Reload systemd, enable the graphical boot target, and start the instance on `tty1`:

```bash
sudo systemctl daemon-reload
sudo systemctl set-default graphical.target
sudo systemctl enable --now cage@tty1.service
```

The explicit instance name `cage@tty1.service` selects `tty1`, overriding the template's `DefaultInstance=tty7`. The service replaces the getty on that terminal and switches the local display to it on startup. Use a dedicated kiosk session without a competing desktop/display manager on that terminal.

5. Check status and logs, or restart after updating the installed app/launcher:

```bash
sudo systemctl status cage@tty1.service
journalctl -u cage@tty1.service -f
sudo systemctl restart cage@tty1.service
```

If startup fails, check that `start.sh` is executable, its paths match the installed files, and `/etc/pam.d/cage` exists. For missing Python modules, confirm the virtual environment uses system site packages. If the display opens but BLE does not advertise, check the journal and `systemctl status bluetooth.service`; the app requires an available Bluetooth adapter.

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
