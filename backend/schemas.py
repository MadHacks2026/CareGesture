"""Shared data contracts for CareGesture.

These four shapes are the agreement between the team members:
  - GesturePrediction: Member 1 (camera + inference) -> backend
  - Environment:       Member 4 (Thermo firmware)    -> backend -> frontend
  - Request, Status:   backend -> Member 3 (frontend)

Do not change a field without telling the whole team.

CareGesture is an accessibility demo. It is NOT a medical or nurse-call device,
and the Thermo reading is room air only, never body temperature.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# The labels the gesture model can output. "none" means no gesture / unsure.
GestureLabel = Literal["paper", "scissors", "rock", "none"]
RequestType = Literal["assistance", "water", "room_comfort"]
RequestStatus = Literal["new", "acknowledged", "resolved"]
# Preset nurse replies. The backend stores the code; the frontend owns the wording.
ReplyCode = Literal[
    "on_my_way",
    "helping_another_patient",
    "bringing_water",
    "adjusting_room",
    "use_call_button",
]

# Which gesture creates which request.
GESTURE_TO_REQUEST: dict[str, str] = {
    "paper": "assistance",      # open palm
    "scissors": "water",        # two fingers
    "rock": "room_comfort",     # closed fist
}


class GesturePrediction(BaseModel):
    """One model output for one camera frame."""

    label: GestureLabel
    confidence: float = Field(ge=0.0, le=1.0)
    # Any millisecond clock works (epoch or uptime). Only differences are used.
    timestamp_ms: int


class Request(BaseModel):
    """One patient request shown on the nurse dashboard."""

    id: str
    room_id: str
    type: RequestType
    status: RequestStatus
    created_at: str  # ISO-8601, UTC
    # Set when a nurse moves the request forward (ISO-8601, UTC). null until then.
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    # A preset reply the nurse sent to the patient, and when (ISO-8601, UTC). null until then.
    reply: Optional[ReplyCode] = None
    reply_at: Optional[str] = None
    # Milliseconds since created_at by the BACKEND's clock, filled in on every read. The UI uses it
    # to time waiting and escalation, so a device with a wrong clock cannot hide or fake an alert.
    age_ms: Optional[int] = None
    # Room conditions at the moment a room_comfort request was created.
    # null for other types, or if no sensor reading existed at that time.
    environment: Optional["Environment"] = None


class Environment(BaseModel):
    """Room air conditions from the Modulino Thermo.

    When sent TO the backend, only the first four fields matter.
    When read FROM the backend:
      - temperature_c / humidity_pct / timestamp_ms are null if no reading
        has ever arrived.
      - stale is true if the last reading is too old to show as current.
      - sensor_ok is false if the sensor reported a failure OR the reading
        is stale or missing.
    """

    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    timestamp_ms: Optional[int] = None
    sensor_ok: bool = False
    stale: bool = False  # set by the backend, ignored on input


# Request refers to Environment, which is defined just above.
Request.model_rebuild()


class Status(BaseModel):
    """Device health. The UI must not imply monitoring if camera_ok or
    inference_ok is false."""

    camera_ok: bool
    inference_ok: bool
    sensor_ok: bool
    backend_ok: bool


DetectionState = Literal["ready", "recognizing", "sent", "rejected", "unavailable"]


class Detection(BaseModel):
    """What the patient screen should show right now.

    ready        waiting for a gesture
    recognizing  a gesture is being held; progress goes 0.0 -> 1.0
    sent         a request was just created
    rejected     a gesture was tried but not accepted (too brief, too unsure)
    unavailable  predictions have stopped arriving (camera or inference down)
    """

    state: DetectionState
    label: Optional[GestureLabel] = None  # only while recognizing
    progress: Optional[float] = None      # only while recognizing, 0.0 to 1.0
    failed_attempts_recent: int = 0       # rejected attempts in the last 5 minutes
