"""In-memory store for patient requests.

A request only ever moves forward:  new -> acknowledged -> resolved
Skipping a step, repeating a step, or going backwards is rejected.
Requests are lost on restart, which is fine for the demo.
"""

import threading
from datetime import datetime, timezone
from typing import Optional

from .schemas import REPLIES_FOR_TYPE, Environment, ReplyCode, Request

DEFAULT_ROOM_ID = "room-101"

# The only allowed status changes: current status -> next status.
ALLOWED_TRANSITIONS = {
    "new": "acknowledged",
    "acknowledged": "resolved",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _with_age(request: Request) -> Request:
    """A copy of the request with age_ms set from the backend's own clock."""
    created = datetime.fromisoformat(request.created_at)
    age = int((datetime.now(timezone.utc) - created).total_seconds() * 1000)
    return request.model_copy(update={"age_ms": max(0, age)})


class RequestNotFound(Exception):
    pass


class InvalidTransition(Exception):
    pass


class ReplyNotAllowed(Exception):
    pass


class RequestStore:
    def __init__(self) -> None:
        self._requests: dict[str, Request] = {}
        self._counter = 0
        self._lock = threading.Lock()  # API threads and the mock thread share this

    def create(
        self,
        request_type: str,
        room_id: str = DEFAULT_ROOM_ID,
        environment: Optional[Environment] = None,
    ) -> Request:
        with self._lock:
            self._counter += 1
            request = Request(
                id=f"req-{self._counter:04d}",
                room_id=room_id,
                type=request_type,
                status="new",
                created_at=_now_iso(),
                environment=environment,
            )
            self._requests[request.id] = request
            return _with_age(request)

    def list(self) -> list[Request]:
        """All requests, newest first."""
        with self._lock:
            return [_with_age(r) for r in reversed(self._requests.values())]

    def get(self, request_id: str) -> Optional[Request]:
        with self._lock:
            request = self._requests.get(request_id)
            return _with_age(request) if request else None

    def transition(self, request_id: str, new_status: str) -> Request:
        """Move a request to new_status. Called only from the nurse's PATCH."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise RequestNotFound(request_id)
            if ALLOWED_TRANSITIONS.get(request.status) != new_status:
                raise InvalidTransition(
                    f"cannot change {request.status} -> {new_status}"
                )
            # Record when the nurse made each step (used for response times).
            stamp = "acknowledged_at" if new_status == "acknowledged" else "resolved_at"
            updated = request.model_copy(
                update={"status": new_status, stamp: _now_iso()}
            )
            self._requests[request_id] = updated
            return _with_age(updated)

    def reply(self, request_id: str, code: ReplyCode) -> Request:
        """Attach a preset reply for the patient. The latest reply wins.

        Replying to a request nobody has looked at yet also acknowledges it, so one
        tap on "On my way" never leaves a card that is replied-to but still "new".
        A resolved request is closed and cannot be replied to, and the reply must fit the request type.
        """
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise RequestNotFound(request_id)
            if request.status == "resolved":
                raise InvalidTransition("request is already resolved")
            if code not in REPLIES_FOR_TYPE.get(request.type, set()):
                raise ReplyNotAllowed(f"reply {code} does not fit a {request.type} request")
            now = _now_iso()
            update = {"reply": code, "reply_at": now}
            if request.status == "new":
                update.update({"status": "acknowledged", "acknowledged_at": now})
            updated = request.model_copy(update=update)
            self._requests[request_id] = updated
            return _with_age(updated)
