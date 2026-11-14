"""In-memory store for patient requests.

A request only ever moves forward:  new -> acknowledged -> resolved
Skipping a step, repeating a step, or going backwards is rejected.
Requests are lost on restart, which is fine for the demo.
"""

import threading
from datetime import datetime, timezone
from typing import Optional

from .schemas import Request

DEFAULT_ROOM_ID = "room-101"

# The only allowed status changes: current status -> next status.
ALLOWED_TRANSITIONS = {
    "new": "acknowledged",
    "acknowledged": "resolved",
}


class RequestNotFound(Exception):
    pass


class InvalidTransition(Exception):
    pass


class RequestStore:
    def __init__(self) -> None:
        self._requests: dict[str, Request] = {}
        self._counter = 0
        self._lock = threading.Lock()  # API threads and the mock thread share this

    def create(self, request_type: str, room_id: str = DEFAULT_ROOM_ID) -> Request:
        with self._lock:
            self._counter += 1
            request = Request(
                id=f"req-{self._counter:04d}",
                room_id=room_id,
                type=request_type,
                status="new",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._requests[request.id] = request
            return request

    def list(self) -> list[Request]:
        """All requests, newest first."""
        with self._lock:
            return list(reversed(self._requests.values()))

    def get(self, request_id: str) -> Optional[Request]:
        with self._lock:
            return self._requests.get(request_id)

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
            updated = request.model_copy(update={"status": new_status})
            self._requests[request_id] = updated
            return updated
