# 🏥 CareGesture

Touchless patient assistance, running entirely on the Arduino UNO Q.

A patient holds a hand gesture in front of the USB camera for about 2 seconds
and a request appears on the nurse station. The Modulino Thermo adds room
temperature and humidity. Nothing leaves the board.

| Gesture | Request |
| --- | --- |
| Open palm | Assistance |
| Two fingers | Water |
| Closed fist | Room comfort (with the room conditions at that moment) |

## Pages

With the app running, open on any device on the same network:

- `http://<board-ip>:7000/patient.html` — patient screen
- `http://<board-ip>:7000/nurse.html` — nurse station (Acknowledge / Resolve)
- `http://<board-ip>:7000/` — launcher, including a browser-only demo mode

## How it is put together

- `python/main.py` — the whole backend: the Video Object Detection brick runs
  the gesture model on the camera feed, the ported state machine turns held
  gestures into requests, and the WebUI brick serves the pages and the
  `/api/...` routes they call.
- `sketch/sketch.ino` — reads the Modulino Thermo once a second and sends it
  over the Bridge.
- `assets/` — the original CareGesture frontend, polling the API on the board.

If the log says a model label "is not in LABEL_ALIASES", add that class name to
`LABEL_ALIASES` in `python/main.py`.

Not a medical or nurse-call device. The Thermo reading is room air, never body
temperature.
