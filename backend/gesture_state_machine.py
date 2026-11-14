"""Turns a stream of gesture predictions into confirmed requests.

This is the false-alert protection. The rules:
  1. "none" and anything below MIN_CONFIDENCE never count as a gesture.
  2. The same gesture must stay stable for HOLD_MS before it fires.
  3. One hold fires exactly one request.
  4. After firing, nothing can fire again until the camera has seen
     "none" continuously for NEUTRAL_MS (the patient lowered their hand).
  5. If predictions stop arriving for more than MAX_GAP_MS (camera stall),
     or the timestamp goes backwards, any hold in progress is thrown away.
  6. Complete silence for NEUTRAL_MS also counts as the neutral period.

Time comes only from prediction.timestamp_ms, never the wall clock, so the
behaviour is deterministic and easy to test.
"""

from typing import Optional

from .schemas import GESTURE_TO_REQUEST, GesturePrediction

# --- Tunable thresholds ------------------------------------------------------
MIN_CONFIDENCE = 0.75   # predictions below this are ignored
HOLD_MS = 1500          # how long a gesture must be held to fire
NEUTRAL_MS = 1000       # how long "none" must be seen before re-arming
# Longest allowed silence between two predictions inside one hold.
# Raise this if the model runs slower than about 3 frames per second.
MAX_GAP_MS = 500

# --- States ------------------------------------------------------------------
IDLE = "idle"          # waiting for a gesture
HOLDING = "holding"    # a valid gesture is being held, timer running
LOCKED = "locked"      # a request just fired, waiting for a neutral period


class GestureStateMachine:
    def __init__(self) -> None:
        self.state = IDLE
        self._label: Optional[str] = None          # gesture being held
        self._hold_start: Optional[int] = None     # when the hold began
        self._neutral_start: Optional[int] = None  # when "none" began (LOCKED)
        self._last_ts: Optional[int] = None        # previous prediction time

    def update(self, pred: GesturePrediction) -> Optional[str]:
        """Feed one prediction. Returns a request type ("assistance",
        "water", "room_comfort") when a hold is confirmed, otherwise None."""
        ts = pred.timestamp_ms
        # Time since the previous prediction (0 for the very first one).
        elapsed = 0 if self._last_ts is None else ts - self._last_ts
        self._last_ts = ts
        # A stall, or a clock that jumped backwards (producer restarted):
        # either way we cannot trust any timer that was running.
        gap = elapsed > MAX_GAP_MS or elapsed < 0

        is_none = pred.label == "none"
        is_valid = not is_none and pred.confidence >= MIN_CONFIDENCE

        if self.state == LOCKED and elapsed >= NEUTRAL_MS:
            # No predictions at all for the whole neutral period. Some
            # producers send nothing when no hand is visible, so silence
            # counts as neutral. Unlock and handle this prediction normally;
            # it still needs its own full hold to fire.
            self._reset()

        if self.state == LOCKED:
            # Only an unbroken run of "none" unlocks. A low-confidence gesture
            # is not proof the hand went down, so it restarts the count.
            if gap or not is_none:
                self._neutral_start = None
            if is_none:
                if self._neutral_start is None:
                    self._neutral_start = ts
                if ts - self._neutral_start >= NEUTRAL_MS:
                    self._reset()
            return None

        if not is_valid:
            # Rule 1: breaks any hold in progress.
            self._reset()
            return None

        same_hold = self.state == HOLDING and pred.label == self._label and not gap
        if not same_hold:
            # New gesture, changed gesture, or a camera stall: restart the timer.
            self.state = HOLDING
            self._label = pred.label
            self._hold_start = ts
            return None

        if ts - self._hold_start >= HOLD_MS:
            # Rules 2-4: fire once, then lock.
            self.state = LOCKED
            self._neutral_start = None
            return GESTURE_TO_REQUEST[pred.label]

        return None

    @property
    def label(self) -> Optional[str]:
        """The gesture currently being held, or None."""
        return self._label if self.state == HOLDING else None

    @property
    def progress(self) -> float:
        """How far through the hold we are, 0.0 to 1.0 (0.0 when not holding)."""
        if self.state != HOLDING or self._hold_start is None:
            return 0.0
        return max(0.0, min(1.0, (self._last_ts - self._hold_start) / HOLD_MS))

    def _reset(self) -> None:
        self.state = IDLE
        self._label = None
        self._hold_start = None
        self._neutral_start = None
