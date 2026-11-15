"""HTTP API for CareGesture.

For the frontend (Member 3):
  GET   /api/requests            list of requests, newest first
  PATCH /api/requests/{id}       body {"status": "acknowledged" | "resolved"}
  POST  /api/requests/{id}/reply body {"reply": "<ReplyCode>"} preset message to the patient;
                                 also acknowledges a new request; 409 once resolved
  GET   /api/environment         latest room reading (check "stale")
  GET   /api/status              device health
  GET   /api/detection           what the patient screen should show right now
  GET   /api/stream              Server-Sent Events, one snapshot per second
                                 (or just poll the GET routes every second)

For the other producers:
  POST  /api/ingest/prediction   Member 1 sends a GesturePrediction
  POST  /api/ingest/environment  Member 4 sends an Environment reading

Mock mode only:
  POST  /api/demo/trigger        DEMO-ONLY manual request, no camera involved
"""

import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .request_store import InvalidTransition, RequestNotFound
from .schemas import (
    Detection,
    Environment,
    GesturePrediction,
    ReplyCode,
    Request,
    RequestStatus,
    RequestType,
    Status,
)

STREAM_INTERVAL_S = 1.0


class StatusUpdate(BaseModel):
    status: RequestStatus


class ReplyBody(BaseModel):
    reply: ReplyCode


class DemoTrigger(BaseModel):
    type: RequestType


def create_app(backend, mock: bool = False) -> FastAPI:
    """backend is a main.Backend. mock=True adds the demo-only trigger route."""
    app = FastAPI(
        title="CareGesture backend",
        description="Accessibility demo. Not a medical or nurse-call device.",
    )
    # The frontend is served from a different port during development.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/api/requests", response_model=list[Request])
    def list_requests():
        return backend.store.list()

    @app.patch("/api/requests/{request_id}", response_model=Request)
    def update_request(request_id: str, body: StatusUpdate):
        # The ONLY place a request status changes: a nurse clicked a button.
        try:
            return backend.store.transition(request_id, body.status)
        except RequestNotFound:
            raise HTTPException(status_code=404, detail="request not found")
        except InvalidTransition as error:
            raise HTTPException(status_code=409, detail=str(error))

    @app.post("/api/requests/{request_id}/reply", response_model=Request)
    def reply_to_request(request_id: str, body: ReplyBody):
        # A nurse tapped a preset reply. Presets only, never free text.
        try:
            return backend.store.reply(request_id, body.reply)
        except RequestNotFound:
            raise HTTPException(status_code=404, detail="request not found")
        except InvalidTransition as error:
            raise HTTPException(status_code=409, detail=str(error))

    @app.get("/api/environment", response_model=Environment)
    def get_environment():
        return backend.get_environment()

    @app.get("/api/status", response_model=Status)
    def get_status():
        return backend.get_status()

    @app.get("/api/detection", response_model=Detection)
    def get_detection(room_id: str = ""):
        # One camera, one room: room_id is accepted so the frontend can send it.
        return backend.get_detection()

    @app.get("/api/stream")
    async def stream():
        async def events():
            while True:
                snapshot = {
                    "requests": [r.model_dump() for r in backend.store.list()],
                    "environment": backend.get_environment().model_dump(),
                    "status": backend.get_status().model_dump(),
                }
                yield f"data: {json.dumps(snapshot)}\n\n"
                await asyncio.sleep(STREAM_INTERVAL_S)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/api/ingest/prediction")
    def ingest_prediction(prediction: GesturePrediction):
        request = backend.handle_prediction(prediction)
        return {"request": request}  # null unless this prediction confirmed a hold

    @app.post("/api/ingest/environment")
    def ingest_environment(reading: Environment):
        backend.handle_environment(reading)
        return {"ok": True}

    if mock:

        @app.post("/api/demo/trigger")
        def demo_trigger(body: DemoTrigger):
            """DEMO-ONLY. Creates a request without any gesture."""
            request = backend.store.create(body.type)
            return {"demo_only": True, "request": request}

    return app
