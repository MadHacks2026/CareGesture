"""One test per false-alert rule in gesture_state_machine.py."""

import pytest

from backend.gesture_state_machine import GestureStateMachine
from backend.schemas import GesturePrediction


def feed(machine, label, confidence, start_ms, duration_ms, step_ms=100):
    """Send predictions every step_ms from start_ms to start_ms + duration_ms
    (inclusive). Returns the list of request types that fired."""
    events = []
    for ts in range(start_ms, start_ms + duration_ms + 1, step_ms):
        result = machine.update(
            GesturePrediction(label=label, confidence=confidence, timestamp_ms=ts)
        )
        if result is not None:
            events.append(result)
    return events


def test_brief_wave_fires_nothing():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.95, 0, 800) == []
    assert feed(machine, "none", 0.95, 900, 1000) == []


def test_low_confidence_hold_fires_nothing():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.74, 0, 5000) == []


def test_none_never_fires():
    machine = GestureStateMachine()
    assert feed(machine, "none", 0.99, 0, 5000) == []


def test_hold_fires_exactly_once():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 1500) == ["assistance"]


def test_just_under_hold_time_fires_nothing():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 1400) == []


def test_continued_hold_still_fires_once():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 10000) == ["assistance"]


def test_repeat_without_neutral_gap_is_blocked():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 1500) == ["assistance"]
    # Only 0.5 s of "none": not long enough to re-arm.
    assert feed(machine, "none", 0.9, 1600, 500) == []
    assert feed(machine, "paper", 0.9, 2200, 3000) == []


def test_repeat_after_neutral_gap_fires_again():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 1500) == ["assistance"]
    assert feed(machine, "none", 0.9, 1600, 1000) == []
    assert feed(machine, "paper", 0.9, 2700, 1500) == ["assistance"]


def test_different_gesture_while_locked_is_blocked():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 1500) == ["assistance"]
    assert feed(machine, "rock", 0.9, 1600, 3000) == []


def test_low_confidence_does_not_count_as_neutral():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 1500) == ["assistance"]
    # Hand still up but the model is unsure: stays locked.
    assert feed(machine, "paper", 0.5, 1600, 2000) == []
    assert feed(machine, "paper", 0.9, 3700, 3000) == []


def test_gesture_switch_restarts_the_timer():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 1000) == []
    # 1.0 s of paper + 1.0 s of rock must not add up to a request.
    assert feed(machine, "rock", 0.9, 1100, 1000) == []
    # Rock fires only after its own full 1.5 s.
    assert feed(machine, "rock", 0.9, 2200, 400) == ["room_comfort"]


def test_low_confidence_frame_breaks_the_hold():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 1000) == []
    assert feed(machine, "paper", 0.5, 1100, 0) == []
    assert feed(machine, "paper", 0.9, 1200, 1400) == []


def test_prediction_gap_resets_the_hold():
    machine = GestureStateMachine()
    assert feed(machine, "paper", 0.9, 0, 1000) == []
    # Camera stalls for 2 s. The hand may have moved, so start over.
    assert feed(machine, "paper", 0.9, 3000, 1400) == []
    assert feed(machine, "paper", 0.9, 4500, 0) == ["assistance"]


@pytest.mark.parametrize(
    "label, expected",
    [("paper", "assistance"), ("scissors", "water"), ("rock", "room_comfort")],
)
def test_gesture_mapping(label, expected):
    machine = GestureStateMachine()
    assert feed(machine, label, 0.9, 0, 1500) == [expected]
