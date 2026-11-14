"""Works out what the patient screen should show (GET /api/detection).

It watches the same predictions as the state machine, but only to report
progress and failed attempts. It never creates requests.

An "attempt" is one unbroken run of non-"none" predictions (a hand in view).
If the run lasts at least MIN_ATTEMPT_MS and never produced a request, it
counts as a failed attempt, so a quick wave or a gesture the model is
unsure about shows "not recognized". Times for the on-screen messages use
the backend clock; run lengths use the prediction timestamps.
"""

from collections import deque
from typing import Optional

from .gesture_state_machine import HOLD_MS, HOLDING, GestureStateMachine
from .schemas import Detection, GesturePrediction

MIN_ATTEMPT_MS = 300        # shorter runs are jitter, not an attempt
SENT_SHOW_MS = 3000         # how long "sent" stays on screen
REJECTED_SHOW_MS = 3000     # how long "rejected" stays on screen
RECENT_WINDOW_MS = 300_000  # failed_attempts_recent looks back 5 minutes


class DetectionTracker:
    def __init__(self) -> None:
        self._run_start: Optional[int] = None   # prediction time the run began
        self._run_last: Optional[int] = None    # prediction time of latest frame
        self._run_fired = False
        self._last_sent_at: Optional[int] = None    # backend clock
        self._last_failed_at: Optional[int] = None  # backend clock
        self._failures: deque[int] = deque()        # backend clock

    def update(self, pred: GesturePrediction, fired: bool, now_ms: int) -> None:
        """Call for every prediction, after the state machine has seen it."""
        if pred.label == "none":
            self._end_run(now_ms)
            return
        if self._run_start is not None and pred.timestamp_ms < self._run_last:
            self._end_run(now_ms)  # clock went backwards: producer restarted
        if self._run_start is None:
            self._run_start = pred.timestamp_ms
            self._run_fired = False
        self._run_last = pred.timestamp_ms
        if fired:
            self._run_fired = True
            self._last_sent_at = now_ms

    def _end_run(self, now_ms: int) -> None:
        if self._run_start is not None and not self._run_fired:
            if self._run_last - self._run_start >= MIN_ATTEMPT_MS:
                self._last_failed_at = now_ms
                self._failures.append(now_ms)
        self._run_start = self._run_last = None
        self._run_fired = False

    def detection(
        self,
        machine: GestureStateMachine,
        predictions_fresh: bool,
        now_ms: int,
    ) -> Detection:
        while self._failures and now_ms - self._failures[0] > RECENT_WINDOW_MS:
            self._failures.popleft()
        recent = len(self._failures)

        if not predictions_fresh:
            return Detection(state="unavailable", failed_attempts_recent=recent)
        if self._last_sent_at is not None and now_ms - self._last_sent_at <= SENT_SHOW_MS:
            return Detection(state="sent", failed_attempts_recent=recent)
        if machine.state == HOLDING:
            return Detection(
                state="recognizing",
                label=machine.label,
                progress=round(machine.progress, 2),
                failed_attempts_recent=recent,
            )
        # A gesture held for a full hold time that the model was never sure
        # about: tell the patient now instead of waiting for them to lower it.
        ongoing_unsure = (
            self._run_start is not None
            and not self._run_fired
            and self._run_last - self._run_start >= HOLD_MS
        )
        recently_failed = (
            self._last_failed_at is not None
            and now_ms - self._last_failed_at <= REJECTED_SHOW_MS
        )
        if ongoing_unsure or recently_failed:
            return Detection(state="rejected", failed_attempts_recent=recent)
        return Detection(state="ready", failed_attempts_recent=recent)
