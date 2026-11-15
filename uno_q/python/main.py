"""CareGesture, running entirely on the UNO Q.

What used to be spread over a laptop (MediaPipe + Keras, FastAPI backend,
static frontend) and the board (Modulino Thermo) now lives in this one app:

  USB camera -> HOG + linear SVM detector trained ON THIS BOARD
                (python/gesture_hog.py, model from training/train_hog_svm.py)
             -> GestureStateMachine + DetectionTracker + RequestStore
                (ported from backend/; pydantic and uvicorn dropped)
             -> REST routes on the WebUI brick, which also serves assets/
  Modulino Thermo (sketch) -> Bridge "record_sensor_samples" -> Environment

Open http://<board-ip>:7000/patient.html and http://<board-ip>:7000/nurse.html.
The rules and thresholds are the original backend's; see the constants below.

CareGesture is an accessibility demo. It is NOT a medical or nurse-call device,
and the Thermo reading is room air only, never body temperature.
"""

import base64
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import Camera

# "ondevice": the HOG + SVM model trained on the UNO Q (python/ondevice_model/).
# "edgeimpulse": the previous cloud-trained model. To go back, set this and
# re-add the brick: arduino:video_object_detection with model ei-model-1164664-1.
DETECTOR_BACKEND = "ondevice"

# Where frames come from. "usb": the room camera on the board. "browser": the
# webcam of the device that opens patient.html (frames are POSTed to the board,
# which decodes them and runs the model). "auto": USB if one is found, else browser.
# Browser mode serves the UI over HTTPS (self-signed), since browsers only allow
# camera access on secure pages.
CAMERA_SOURCE = "auto"

# On-device pipeline only: a label counts once it is seen in CONFIRM_FRAMES of
# the last CONFIRM_WINDOW frames, and after a request fires, that request
# cannot fire again for REQUEST_COOLDOWN_MS.
CONFIRM_WINDOW = 4
CONFIRM_FRAMES = 3
REQUEST_COOLDOWN_MS = 5000

# --- Tunable thresholds (same values as backend/) ----------------------------
MIN_CONFIDENCE = 0.6    # predictions below this are ignored (laptop model used 0.75)
HOLD_MS = 1500          # how long a gesture must be held to fire
NEUTRAL_MS = 1000       # how long "none" must be seen before re-arming
# Longest silence between two predictions inside one hold. The Edge Impulse
# model on the board reports 1-7 detections/s while a hand is up, so this is
# looser than the laptop's 500 ms.
MAX_GAP_MS = 1000
MIN_ATTEMPT_MS = 300    # shorter runs are jitter, not an attempt
SENT_SHOW_MS = 3000     # how long "sent" stays on screen
REJECTED_SHOW_MS = 3000 # how long "rejected" stays on screen
RECENT_WINDOW_MS = 300_000  # failed_attempts_recent looks back 5 minutes
ENVIRONMENT_STALE_MS = 10000

# The brick only calls back when the model sees a gesture; the laptop pipeline
# sent "none" for every empty frame. After this much silence a "none" is fed in
# so a lowered hand ends the hold and re-arms. Keep it >= MAX_GAP_MS.
NO_HAND_MS = 1000
# The brick gives no heartbeat, so the detector counts as up after its first
# detection, or after this warm-up (model loading) has passed.
DETECTOR_WARMUP_S = 20

ROOM_ID = "room-101"  # matches the frontend's default ?room= for a real backend

GESTURE_TO_REQUEST = {
    "paper": "assistance",      # open palm
    "scissors": "water",        # two fingers
    "rock": "room_comfort",     # closed fist
}

# Preset nurse replies. The backend stores the code; the frontend owns the wording.
# A reply that does not fit the request type is rejected, so a patient is never told something untrue.
REPLIES_FOR_TYPE = {
    "assistance": {"on_my_way", "helping_another_patient", "use_call_button"},
    "water": {"bringing_water", "helping_another_patient"},
    "room_comfort": {"adjusting_room", "helping_another_patient"},
}

# Edge Impulse class name -> the label the frontend knows. Names are lower-cased
# with spaces and dashes turned into underscores before lookup. If the log says
# a label "is not in LABEL_ALIASES", add it here.
LABEL_ALIASES = {
    # madhacks26-gestures classes
    "assistance": "paper", "water_request": "scissors", "comfort": "rock",
    # other common names
    "paper": "paper", "open": "paper", "open_hand": "paper", "open_palm": "paper",
    "palm": "paper", "five": "paper",
    "scissors": "scissors", "water": "scissors", "peace": "scissors", "two": "scissors",
    "two_fingers": "scissors", "v": "scissors", "v_sign": "scissors", "victory": "scissors",
    "rock": "rock", "room_comfort": "rock", "fist": "rock", "closed": "rock", "closed_fist": "rock",
    "none": "none", "neutral": "none", "background": "none", "nothing": "none",
}

# --- Clocks ------------------------------------------------------------------


def now_ms() -> int:
    """Backend clock: milliseconds of uptime. Only differences are ever used."""
    return int(time.monotonic() * 1000)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Data shapes (the team contract from backend/schemas.py) ------------------


@dataclass
class Prediction:
    label: str          # paper, scissors, rock, none
    confidence: float
    timestamp_ms: int


@dataclass
class Environment:
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    timestamp_ms: Optional[int] = None  # wall clock: the browser compares it to Date.now()
    sensor_ok: bool = False
    stale: bool = False


@dataclass
class Request:
    id: str
    room_id: str
    type: str
    status: str
    created_at: str
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    # A preset reply the nurse sent to the patient, and when (ISO-8601, UTC).
    reply: Optional[str] = None
    reply_at: Optional[str] = None
    # Milliseconds since created_at by the board's clock, filled in on every read. The UI times
    # waiting and escalation with it, so a device with a wrong clock cannot hide or fake an alert.
    age_ms: Optional[int] = None
    environment: Optional[Environment] = None


@dataclass
class Status:
    camera_ok: bool
    inference_ok: bool
    sensor_ok: bool
    backend_ok: bool


@dataclass
class Detection:
    state: str  # ready, recognizing, sent, rejected, unavailable
    label: Optional[str] = None
    progress: Optional[float] = None
    failed_attempts_recent: int = 0
    # Raw on-device model output for the latest frame (before hold/cooldown rules).
    model: Optional[str] = None
    live_label: Optional[str] = None
    live_confidence: Optional[float] = None
    live_ms: Optional[int] = None


# --- Gesture state machine (backend/gesture_state_machine.py) ----------------

IDLE = "idle"
HOLDING = "holding"
LOCKED = "locked"


class GestureStateMachine:
    """Turns a stream of predictions into confirmed requests. The false-alert
    protection: a gesture must be held for HOLD_MS, fires once, then nothing
    fires again until "none" has been seen for NEUTRAL_MS."""

    def __init__(self) -> None:
        self.state = IDLE
        self._label: Optional[str] = None
        self._hold_start: Optional[int] = None
        self._neutral_start: Optional[int] = None
        self._last_ts: Optional[int] = None
        self._fired_at: Optional[int] = None

    def _cooled(self, ts: int) -> bool:
        return self._fired_at is None or ts - self._fired_at >= REQUEST_COOLDOWN_MS

    def update(self, pred: Prediction) -> Optional[str]:
        ts = pred.timestamp_ms
        is_none = pred.label == "none"
        if not is_none and pred.confidence < MIN_CONFIDENCE:
            # Unsure frame. The Edge Impulse model jitters around the threshold
            # (0.84, 0.43, 0.78...) so, unlike the laptop version, this is neither
            # evidence of a gesture nor of a lowered hand: ignore it. A real stall
            # still ends the hold through the gap rule and the synthesized "none".
            return None
        elapsed = 0 if self._last_ts is None else ts - self._last_ts
        self._last_ts = ts
        gap = elapsed > MAX_GAP_MS or elapsed < 0
        is_valid = not is_none

        if self.state == LOCKED and elapsed >= NEUTRAL_MS and self._cooled(ts):
            self._reset()  # silence for the whole neutral period counts as neutral

        if self.state == LOCKED:
            if gap or not is_none:
                self._neutral_start = None
            if is_none:
                if self._neutral_start is None:
                    self._neutral_start = ts
                if ts - self._neutral_start >= NEUTRAL_MS and self._cooled(ts):
                    self._reset()
            return None

        if not is_valid:
            self._reset()
            return None

        same_hold = self.state == HOLDING and pred.label == self._label and not gap
        if not same_hold:
            self.state = HOLDING
            self._label = pred.label
            self._hold_start = ts
            return None

        if ts - self._hold_start >= HOLD_MS:
            self.state = LOCKED
            self._neutral_start = None
            self._fired_at = ts
            return GESTURE_TO_REQUEST[pred.label]

        return None

    @property
    def label(self) -> Optional[str]:
        return self._label if self.state == HOLDING else None

    @property
    def progress(self) -> float:
        if self.state != HOLDING or self._hold_start is None:
            return 0.0
        return max(0.0, min(1.0, (self._last_ts - self._hold_start) / HOLD_MS))

    def _reset(self) -> None:
        self.state = IDLE
        self._label = None
        self._hold_start = None
        self._neutral_start = None


# --- Detection tracker (backend/detection.py) --------------------------------


class DetectionTracker:
    """Works out what the patient screen should show. Watches the same
    predictions as the state machine but never creates requests."""

    def __init__(self) -> None:
        self._run_start: Optional[int] = None
        self._run_last: Optional[int] = None
        self._run_fired = False
        self._last_sent_at: Optional[int] = None
        self._last_failed_at: Optional[int] = None
        self._failures: deque = deque()

    def update(self, pred: Prediction, fired: bool, now: int) -> None:
        if pred.label == "none":
            self._end_run(now)
            return
        if self._run_start is not None and pred.timestamp_ms < self._run_last:
            self._end_run(now)
        if self._run_start is None:
            self._run_start = pred.timestamp_ms
            self._run_fired = False
        self._run_last = pred.timestamp_ms
        if fired:
            self._run_fired = True
            self._last_sent_at = now

    def _end_run(self, now: int) -> None:
        if self._run_start is not None and not self._run_fired:
            if self._run_last - self._run_start >= MIN_ATTEMPT_MS:
                self._last_failed_at = now
                self._failures.append(now)
        self._run_start = self._run_last = None
        self._run_fired = False

    def detection(self, machine: GestureStateMachine, predictions_fresh: bool, now: int) -> Detection:
        while self._failures and now - self._failures[0] > RECENT_WINDOW_MS:
            self._failures.popleft()
        recent = len(self._failures)

        if not predictions_fresh:
            return Detection(state="unavailable", failed_attempts_recent=recent)
        if self._last_sent_at is not None and now - self._last_sent_at <= SENT_SHOW_MS:
            return Detection(state="sent", failed_attempts_recent=recent)
        if machine.state == HOLDING:
            return Detection(
                state="recognizing",
                label=machine.label,
                progress=round(machine.progress, 2),
                failed_attempts_recent=recent,
            )
        ongoing_unsure = (
            self._run_start is not None
            and not self._run_fired
            and self._run_last - self._run_start >= HOLD_MS
        )
        recently_failed = (
            self._last_failed_at is not None and now - self._last_failed_at <= REJECTED_SHOW_MS
        )
        if ongoing_unsure or recently_failed:
            return Detection(state="rejected", failed_attempts_recent=recent)
        return Detection(state="ready", failed_attempts_recent=recent)


# --- Request store (backend/request_store.py) --------------------------------

ALLOWED_TRANSITIONS = {"new": "acknowledged", "acknowledged": "resolved"}


class RequestNotFound(Exception):
    pass


class InvalidTransition(Exception):
    pass


class ReplyNotAllowed(Exception):
    pass


def with_age(request: "Request") -> "Request":
    """A copy of the request with age_ms set from the board's own clock."""
    created = datetime.fromisoformat(request.created_at)
    age = int((datetime.now(timezone.utc) - created).total_seconds() * 1000)
    return replace(request, age_ms=max(0, age))


class RequestStore:
    """In-memory. A request only ever moves forward: new -> acknowledged -> resolved."""

    def __init__(self) -> None:
        self._requests: dict = {}
        self._counter = 0
        self._lock = threading.Lock()

    def create(self, request_type: str, environment: Optional[Environment] = None) -> Request:
        with self._lock:
            self._counter += 1
            request = Request(
                id=f"req-{self._counter:04d}",
                room_id=ROOM_ID,
                type=request_type,
                status="new",
                created_at=now_iso(),
                environment=environment,
            )
            self._requests[request.id] = request
            return with_age(request)

    def list(self) -> list:
        with self._lock:
            return [with_age(r) for r in reversed(self._requests.values())]

    def transition(self, request_id: str, new_status: str) -> Request:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise RequestNotFound(request_id)
            if ALLOWED_TRANSITIONS.get(request.status) != new_status:
                raise InvalidTransition(f"cannot change {request.status} -> {new_status}")
            stamp = "acknowledged_at" if new_status == "acknowledged" else "resolved_at"
            updated = replace(request, status=new_status, **{stamp: now_iso()})
            self._requests[request_id] = updated
            return with_age(updated)

    def reply(self, request_id: str, code: str) -> Request:
        """Attach a preset reply for the patient. The latest reply wins.

        Replying to a request nobody has looked at yet also acknowledges it. A resolved
        request is closed, and the reply must fit the request type."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise RequestNotFound(request_id)
            if request.status == "resolved":
                raise InvalidTransition("request is already resolved")
            if code not in REPLIES_FOR_TYPE.get(request.type, set()):
                raise ReplyNotAllowed(f"reply {code} does not fit a {request.type} request")
            now = now_iso()
            changes = {"reply": code, "reply_at": now}
            if request.status == "new":
                changes.update(status="acknowledged", acknowledged_at=now)
            updated = replace(request, **changes)
            self._requests[request_id] = updated
            return with_age(updated)


# --- Backend (backend/main.py) -----------------------------------------------


class Backend:
    def __init__(self) -> None:
        self.store = RequestStore()
        self._machine = GestureStateMachine()
        self._tracker = DetectionTracker()
        self._lock = threading.Lock()
        self._started_at = now_ms()
        self._last_prediction_at: Optional[int] = None  # real or synthesized "none"
        self._last_detection_at: Optional[int] = None   # model output only
        self._environment: Optional[Environment] = None
        self._environment_at: Optional[int] = None

    def handle_prediction(self, pred: Prediction, from_model: bool = True) -> Optional[Request]:
        with self._lock:
            now = now_ms()
            self._last_prediction_at = now
            if from_model:
                self._last_detection_at = now
            request_type = self._machine.update(pred)
            self._tracker.update(pred, request_type is not None, now)
        if request_type is None:
            return None
        snapshot = None
        if request_type == "room_comfort":
            current = self.get_environment()
            if current.temperature_c is not None:
                snapshot = current
        return self.store.create(request_type, environment=snapshot)

    def hold_progress(self) -> float:
        with self._lock:
            return self._machine.progress

    def ms_since_prediction(self) -> Optional[int]:
        with self._lock:
            if self._last_prediction_at is None:
                return None
            return now_ms() - self._last_prediction_at

    def handle_environment(self, reading: Environment) -> None:
        with self._lock:
            self._environment = reading
            self._environment_at = now_ms()

    def get_environment(self) -> Environment:
        with self._lock:
            if self._environment is None:
                return Environment(sensor_ok=False, stale=True)
            stale = now_ms() - self._environment_at > ENVIRONMENT_STALE_MS
            return replace(
                self._environment,
                stale=stale,
                sensor_ok=self._environment.sensor_ok and not stale,
            )

    def _detector_ok(self) -> bool:
        return (
            self._last_detection_at is not None
            or now_ms() - self._started_at >= DETECTOR_WARMUP_S * 1000
        )

    def get_detection(self) -> Detection:
        with self._lock:
            detection = self._tracker.detection(self._machine, self._detector_ok(), now_ms())
        if DETECTOR_BACKEND == "ondevice":
            detection = replace(detection, model="HOG+SVM trained on UNO Q", **live_result)
        return detection

    def get_status(self) -> Status:
        ok = self._detector_ok()
        return Status(
            camera_ok=ok,
            inference_ok=ok,
            sensor_ok=self.get_environment().sensor_ok,
            backend_ok=True,
        )


backend = Backend()

# --- Camera + model (replaces gesture_model/webcam_feed.py) -------------------

_unmapped_labels: set = set()
_detections_in_window = 0
_last_seen_log = 0.0


def normalize_label(raw: str) -> str:
    return str(raw).strip().lower().replace(" ", "_").replace("-", "_")


def on_detections(detections: dict) -> None:
    """One callback per analysed frame that contains at least one detection.
    detections: {label: {"confidence", "bounding_box_xyxy"}} or {label: [...]}"""
    global _detections_in_window, _last_seen_log
    _detections_in_window += 1

    best_label, best_conf = "none", 1.0
    found = -1.0
    for raw, d in detections.items():
        label = LABEL_ALIASES.get(normalize_label(raw))
        if label is None:
            if raw not in _unmapped_labels:
                _unmapped_labels.add(raw)
                print(f"Model label {raw!r} is not in LABEL_ALIASES; ignoring it")
            continue
        for item in d if isinstance(d, list) else [d]:
            conf = float(item.get("confidence", 0.0))
            if conf > found:
                found = conf
                best_label, best_conf = label, conf

    # At most one line per second, so the log shows what the model sees and how sure it is.
    if found >= 0 and time.monotonic() - _last_seen_log >= 1.0:
        _last_seen_log = time.monotonic()
        print(f"Seen: {best_label} {best_conf:.2f} (hold {backend.hold_progress():.0%})")

    request = backend.handle_prediction(Prediction(best_label, best_conf, now_ms()))
    if request is not None:
        print(f"Request {request.id}: {request.type} ({best_label} held)")


def background_loop() -> None:
    """Feeds "none" while the model is quiet, and logs the model rate every 10 s."""
    global _detections_in_window
    last_report = time.monotonic()
    while True:
        time.sleep(0.1)
        since = backend.ms_since_prediction()
        if since is None or since >= NO_HAND_MS:
            backend.handle_prediction(Prediction("none", 1.0, now_ms()), from_model=False)
        if time.monotonic() - last_report >= 10:
            print(f"Model: {_detections_in_window} detections in the last 10 s")
            _detections_in_window = 0
            last_report = time.monotonic()


# Latest raw on-device prediction, shown on the patient page via /api/detection.
live_result: dict = {"live_label": None, "live_confidence": None, "live_ms": None}


def ondevice_loop() -> None:
    """Grabs frames, runs the UNO Q-trained HOG + SVM detector, feeds the state machine.
    Everything here runs on the board: OpenCV HOG + one matrix product, no network."""
    from gesture_hog import DEFAULT_MODEL_PATH, GestureDetector

    model = GestureDetector.load()
    print(f"On-device model loaded: {DEFAULT_MODEL_PATH} (threshold {model.threshold}, T {model.T:.2f})")
    if not frame_source.is_started():
        frame_source.start()
    recent: deque = deque(maxlen=CONFIRM_WINDOW)
    while True:
        frame = frame_source.capture()
        if frame is None:
            time.sleep(0.02)
            continue
        t0 = time.perf_counter()
        # 0.7 had the fewest false alarms on held-out images (training/evidence/train_log.txt).
        dets = model.detect_frame(frame, threshold=0.7)
        ms = int((time.perf_counter() - t0) * 1000)
        raw, conf = (dets[0][0], dets[0][1]) if dets else ("none", 1.0)
        live_result.update(live_label=None if raw == "none" else raw,
                           live_confidence=None if raw == "none" else round(conf, 2), live_ms=ms)
        recent.append(raw)
        # Consecutive-frame confirmation: unconfirmed frames are skipped, not fed.
        if sum(1 for r in recent if r == raw) < CONFIRM_FRAMES:
            continue
        if raw == "none":
            backend.handle_prediction(Prediction("none", 1.0, now_ms()))
        else:
            on_detections({raw: {"confidence": conf}})


class BrowserCamera:
    """Frames the patient page sends from its own webcam (POST /api/frame).
    Only the capture happens in the browser: decoding, HOG and the SVM run on the board."""

    def __init__(self) -> None:
        self._frame = None
        self._new = threading.Event()
        self._lock = threading.Lock()

    def push(self, frame) -> None:
        with self._lock:
            self._frame = frame
        self._new.set()

    def capture(self):
        if not self._new.wait(timeout=1.0):
            return None
        self._new.clear()
        with self._lock:
            return self._frame

    def is_started(self) -> bool:
        return True

    def start(self) -> None:
        pass


# One USB camera, shared: the detector reads frames from it and the
# WebUI streams it to the patient page as MJPEG at /camera (see below).
camera = None
if CAMERA_SOURCE != "browser":
    try:
        camera = Camera(resolution=(640, 480), fps=15)
    except Exception as error:
        if CAMERA_SOURCE == "usb":
            raise
        print(f"No USB camera ({error}); using the browser's webcam instead")
browser_camera = BrowserCamera() if camera is None else None
frame_source = camera if camera is not None else browser_camera
print(f"Camera source: {'usb' if camera is not None else 'browser'}")

if DETECTOR_BACKEND == "ondevice":
    threading.Thread(target=ondevice_loop, daemon=True).start()
else:
    from arduino.app_bricks.video_objectdetection import VideoObjectDetection

    # The brick runs the model selected in app.yaml on the camera above.
    # Its threshold is a little under MIN_CONFIDENCE: frames in between still reach
    # the tracker (so an unsure hold shows "not recognized") but never the hold timer.
    detector = VideoObjectDetection(camera=camera, confidence=0.5, debounce_sec=0.0)
    detector.on_detect_all(on_detections)

# --- Modulino Thermo (sketch/sketch.ino, once a second) ----------------------


def record_sensor_samples(celsius, humidity) -> None:
    def valid(v):
        return isinstance(v, (int, float)) and not math.isnan(v)

    ok = valid(celsius)
    backend.handle_environment(
        Environment(
            temperature_c=round(float(celsius), 1) if ok else None,
            humidity_pct=round(float(humidity), 1) if ok and valid(humidity) else None,
            timestamp_ms=int(time.time() * 1000),
            sensor_ok=ok,
        )
    )


Bridge.provide("record_sensor_samples", record_sensor_samples)

# --- HTTP API (backend/api.py) on the WebUI brick ----------------------------

ui = WebUI(use_tls=browser_camera is not None)
if camera is not None:
    ui.expose_camera("/camera", camera)  # <img src="/camera"> on patient.html


def get_camera_source():
    return {"source": "usb" if camera is not None else "browser"}


def post_frame(body: dict):
    # A JPEG frame from the browser's webcam, base64 in JSON. Decoded and analysed on the board.
    if browser_camera is None:
        raise HTTPException(status_code=409, detail="the board is using its USB camera")
    try:
        data = base64.b64decode((body or {}).get("jpeg", ""))
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        frame = None
    if frame is None:
        raise HTTPException(status_code=422, detail="jpeg must be a base64 JPEG image")
    browser_camera.push(frame)
    return {"ok": True}


def list_requests():
    return [asdict(r) for r in backend.store.list()]


def update_request(request_id: str, body: dict):
    # The ONLY place a request status changes: a nurse clicked a button.
    status = (body or {}).get("status")
    if status not in ("acknowledged", "resolved"):
        raise HTTPException(status_code=422, detail="status must be acknowledged or resolved")
    try:
        return asdict(backend.store.transition(request_id, status))
    except RequestNotFound:
        raise HTTPException(status_code=404, detail="request not found")
    except InvalidTransition as error:
        raise HTTPException(status_code=409, detail=str(error))


def reply_to_request(request_id: str, body: dict):
    # A nurse tapped a preset reply. Presets only, never free text.
    code = (body or {}).get("reply")
    if code not in {c for codes in REPLIES_FOR_TYPE.values() for c in codes}:
        raise HTTPException(status_code=422, detail="unknown reply")
    try:
        return asdict(backend.store.reply(request_id, code))
    except RequestNotFound:
        raise HTTPException(status_code=404, detail="request not found")
    except InvalidTransition as error:
        raise HTTPException(status_code=409, detail=str(error))
    except ReplyNotAllowed as error:
        raise HTTPException(status_code=422, detail=str(error))


def get_environment():
    return asdict(backend.get_environment())


def get_status():
    return asdict(backend.get_status())


def get_detection(room_id: str = ""):
    # One camera, one room: room_id is accepted so the frontend can send it.
    return asdict(backend.get_detection())


ui.expose_api("GET", "/api/requests", list_requests)
ui.expose_api("PATCH", "/api/requests/{request_id}", update_request)
ui.expose_api("POST", "/api/requests/{request_id}/reply", reply_to_request)
ui.expose_api("GET", "/api/environment", get_environment)
ui.expose_api("GET", "/api/status", get_status)
ui.expose_api("GET", "/api/detection", get_detection)
ui.expose_api("GET", "/api/camera_source", get_camera_source)
ui.expose_api("POST", "/api/frame", post_frame)

threading.Thread(target=background_loop, daemon=True).start()

App.run()
