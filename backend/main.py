"""Wires everything together and starts the server.

    python -m backend.main --mock     laptop, no hardware, fake data
    python -m backend.main            real mode: Member 1 and Member 4 push
                                      data to the /api/ingest/* routes (or
                                      call Backend.handle_* in-process)
"""

import argparse
import random
import threading
import time
from typing import Callable, Optional

import uvicorn

from .api import create_app
from .gesture_state_machine import GestureStateMachine
from .detection import DetectionTracker
from .request_store import RequestStore
from .schemas import Detection, Environment, GesturePrediction, Request, Status

# If no prediction arrives for this long, the camera/inference is not working.
PREDICTION_TIMEOUT_MS = 3000
# If no sensor reading arrives for this long, the reading is stale.
ENVIRONMENT_STALE_MS = 10000


def wall_clock_ms() -> int:
    return int(time.time() * 1000)


class Backend:
    """Holds all backend state. One instance per server."""

    def __init__(self, now_ms: Callable[[], int] = wall_clock_ms) -> None:
        self.store = RequestStore()
        self._machine = GestureStateMachine()
        self._tracker = DetectionTracker()
        self._now_ms = now_ms  # injectable so tests can control time
        self._lock = threading.Lock()
        self._last_prediction_at: Optional[int] = None   # backend clock
        self._environment: Optional[Environment] = None
        self._environment_at: Optional[int] = None       # backend clock

    def handle_prediction(self, prediction: GesturePrediction) -> Optional[Request]:
        """Member 1 calls this for every frame. Returns the new request if
        this prediction confirmed a hold. Never looks at the sensor."""
        with self._lock:
            now = self._now_ms()
            self._last_prediction_at = now
            request_type = self._machine.update(prediction)
            self._tracker.update(prediction, request_type is not None, now)
        if request_type is None:
            return None
        # Room-comfort requests carry the room conditions at this moment.
        snapshot = None
        if request_type == "room_comfort":
            current = self.get_environment()
            if current.temperature_c is not None:
                snapshot = current
        return self.store.create(request_type, environment=snapshot)

    def handle_environment(self, reading: Environment) -> None:
        """Member 4 calls this for every Thermo reading."""
        with self._lock:
            self._environment = reading
            # Staleness uses OUR clock at arrival. The board's timestamp may
            # be uptime milliseconds, which cannot be compared to wall time.
            self._environment_at = self._now_ms()

    def get_environment(self) -> Environment:
        with self._lock:
            if self._environment is None:
                return Environment(sensor_ok=False, stale=True)
            stale = self._now_ms() - self._environment_at > ENVIRONMENT_STALE_MS
            return self._environment.model_copy(
                update={
                    "stale": stale,
                    "sensor_ok": self._environment.sensor_ok and not stale,
                }
            )

    def _predictions_fresh(self) -> bool:
        """Caller must hold self._lock."""
        return (
            self._last_prediction_at is not None
            and self._now_ms() - self._last_prediction_at <= PREDICTION_TIMEOUT_MS
        )

    def get_detection(self) -> Detection:
        with self._lock:
            return self._tracker.detection(
                self._machine, self._predictions_fresh(), self._now_ms()
            )

    def get_status(self) -> Status:
        with self._lock:
            predictions_fresh = self._predictions_fresh()
        # With one prediction stream we cannot tell a dead camera from dead
        # inference, so both follow prediction freshness.
        return Status(
            camera_ok=predictions_fresh,
            inference_ok=predictions_fresh,
            sensor_ok=self.get_environment().sensor_ok,
            backend_ok=True,
        )


# --- Mock mode ---------------------------------------------------------------
# (label, confidence, seconds). Loops forever. Shows every rule in action.
MOCK_SCRIPT = [
    ("none", 0.95, 1.0),
    ("paper", 0.92, 2.2),     # real hold        -> assistance
    ("none", 0.95, 2.0),
    ("paper", 0.90, 0.4),     # brief wave       -> nothing
    ("none", 0.95, 1.5),
    ("scissors", 0.60, 2.0),  # low confidence   -> nothing
    ("none", 0.95, 1.5),
    ("scissors", 0.88, 2.2),  # real hold        -> water
    ("none", 0.95, 2.0),
    ("rock", 0.85, 2.2),      # real hold        -> room_comfort
    ("none", 0.95, 20.0),     # quiet period before the loop repeats
]
MOCK_FRAME_INTERVAL_S = 0.1
MOCK_ENVIRONMENT_INTERVAL_S = 2.0


def run_mock_predictions(backend: Backend) -> None:
    while True:
        for label, confidence, seconds in MOCK_SCRIPT:
            end = time.time() + seconds
            while time.time() < end:
                backend.handle_prediction(
                    GesturePrediction(
                        label=label, confidence=confidence, timestamp_ms=wall_clock_ms()
                    )
                )
                time.sleep(MOCK_FRAME_INTERVAL_S)


def run_mock_environment(backend: Backend) -> None:
    while True:
        backend.handle_environment(
            Environment(
                temperature_c=round(22.0 + random.uniform(-0.5, 0.5), 1),
                humidity_pct=round(45.0 + random.uniform(-2.0, 2.0), 1),
                timestamp_ms=wall_clock_ms(),
                sensor_ok=True,
            )
        )
        time.sleep(MOCK_ENVIRONMENT_INTERVAL_S)


def main() -> None:
    parser = argparse.ArgumentParser(description="CareGesture backend")
    parser.add_argument("--mock", action="store_true", help="fake gestures and sensor")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    backend = Backend()
    if args.mock:
        print("MOCK MODE: predictions and sensor readings are fake.")
        for target in (run_mock_predictions, run_mock_environment):
            threading.Thread(target=target, args=(backend,), daemon=True).start()

    uvicorn.run(create_app(backend, mock=args.mock), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
