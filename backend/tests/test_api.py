"""API tests. Uses a fake clock so staleness can be tested without waiting."""

import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.main import ENVIRONMENT_STALE_MS, PREDICTION_TIMEOUT_MS, Backend


class FakeClock:
    def __init__(self):
        self.now = 1_000_000

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def client(clock):
    return TestClient(create_app(Backend(now_ms=clock), mock=False))


def hold(client, label="paper", confidence=0.9, start_ms=0, duration_ms=1500):
    """Post predictions every 100 ms for duration_ms."""
    for ts in range(start_ms, start_ms + duration_ms + 1, 100):
        response = client.post(
            "/api/ingest/prediction",
            json={"label": label, "confidence": confidence, "timestamp_ms": ts},
        )
        assert response.status_code == 200


def send_environment(client, sensor_ok=True):
    response = client.post(
        "/api/ingest/environment",
        json={
            "temperature_c": 22.5,
            "humidity_pct": 44.0,
            "timestamp_ms": 123,
            "sensor_ok": sensor_ok,
        },
    )
    assert response.status_code == 200


def test_starts_empty(client):
    assert client.get("/api/requests").json() == []


def test_held_gesture_creates_one_request(client):
    hold(client, "paper", duration_ms=4000)
    requests = client.get("/api/requests").json()
    assert len(requests) == 1
    assert requests[0]["type"] == "assistance"
    assert requests[0]["status"] == "new"
    assert set(requests[0]) == {
        "id", "room_id", "type", "status", "created_at",
        "acknowledged_at", "resolved_at", "environment",
    }
    assert requests[0]["acknowledged_at"] is None
    assert requests[0]["resolved_at"] is None
    assert requests[0]["environment"] is None  # only room_comfort carries one


def test_brief_gesture_creates_nothing(client):
    hold(client, "paper", duration_ms=500)
    assert client.get("/api/requests").json() == []


def test_nurse_patch_flow(client):
    hold(client, "scissors")
    request_id = client.get("/api/requests").json()[0]["id"]
    url = f"/api/requests/{request_id}"

    # Skipping straight to resolved is rejected.
    assert client.patch(url, json={"status": "resolved"}).status_code == 409

    response = client.patch(url, json={"status": "acknowledged"})
    assert response.status_code == 200
    assert response.json()["status"] == "acknowledged"

    assert client.patch(url, json={"status": "resolved"}).json()["status"] == "resolved"
    # Resolved is final.
    assert client.patch(url, json={"status": "acknowledged"}).status_code == 409


def test_patch_unknown_id_and_bad_status(client):
    assert (
        client.patch("/api/requests/req-9999", json={"status": "acknowledged"}).status_code
        == 404
    )
    hold(client)
    request_id = client.get("/api/requests").json()[0]["id"]
    assert (
        client.patch(f"/api/requests/{request_id}", json={"status": "done"}).status_code
        == 422
    )


def test_environment_before_any_reading(client):
    body = client.get("/api/environment").json()
    assert body["temperature_c"] is None
    assert body["sensor_ok"] is False
    assert body["stale"] is True


def test_environment_fresh_then_stale(client, clock):
    send_environment(client)
    body = client.get("/api/environment").json()
    assert body["temperature_c"] == 22.5
    assert body["sensor_ok"] is True
    assert body["stale"] is False

    clock.now += ENVIRONMENT_STALE_MS + 1
    body = client.get("/api/environment").json()
    assert body["stale"] is True
    assert body["sensor_ok"] is False
    assert client.get("/api/status").json()["sensor_ok"] is False


def test_sensor_reporting_failure_is_not_ok(client):
    send_environment(client, sensor_ok=False)
    assert client.get("/api/environment").json()["sensor_ok"] is False
    assert client.get("/api/status").json()["sensor_ok"] is False


def test_missing_sensor_does_not_block_requests(client):
    # No environment reading was ever sent.
    hold(client, "rock")
    requests = client.get("/api/requests").json()
    assert [r["type"] for r in requests] == ["room_comfort"]
    assert client.get("/api/status").json()["sensor_ok"] is False


def test_status_camera_goes_offline_when_predictions_stop(client, clock):
    status = client.get("/api/status").json()
    assert status["camera_ok"] is False  # nothing received yet
    assert status["backend_ok"] is True

    hold(client, "none", duration_ms=200)
    status = client.get("/api/status").json()
    assert status["camera_ok"] is True
    assert status["inference_ok"] is True

    clock.now += PREDICTION_TIMEOUT_MS + 1
    status = client.get("/api/status").json()
    assert status["camera_ok"] is False
    assert status["inference_ok"] is False


def test_demo_trigger_only_exists_in_mock_mode(client):
    assert client.post("/api/demo/trigger", json={"type": "water"}).status_code == 404

    mock_client = TestClient(create_app(Backend(), mock=True))
    response = mock_client.post("/api/demo/trigger", json={"type": "water"})
    assert response.status_code == 200
    assert response.json()["demo_only"] is True
    assert mock_client.get("/api/requests").json()[0]["type"] == "water"


# --- Timestamps, environment snapshot, detection ------------------------------

def test_nurse_steps_are_timestamped(client):
    hold(client, "scissors")
    request_id = client.get("/api/requests").json()[0]["id"]
    acked = client.patch(f"/api/requests/{request_id}", json={"status": "acknowledged"}).json()
    assert acked["acknowledged_at"] is not None and acked["resolved_at"] is None
    done = client.patch(f"/api/requests/{request_id}", json={"status": "resolved"}).json()
    assert done["acknowledged_at"] == acked["acknowledged_at"]
    assert done["resolved_at"] is not None


def test_room_comfort_carries_environment_snapshot(client):
    send_environment(client)
    hold(client, "rock")
    request = client.get("/api/requests").json()[0]
    assert request["type"] == "room_comfort"
    assert request["environment"]["temperature_c"] == 22.5
    assert request["environment"]["humidity_pct"] == 44.0


def test_room_comfort_without_a_reading_has_no_snapshot(client):
    hold(client, "rock")
    request = client.get("/api/requests").json()[0]
    assert request["type"] == "room_comfort"
    assert request["environment"] is None


def test_other_types_have_no_snapshot(client):
    send_environment(client)
    hold(client, "paper")
    assert client.get("/api/requests").json()[0]["environment"] is None


def detection(client):
    return client.get("/api/detection", params={"room_id": "01"}).json()


def post_prediction(client, label, ts, confidence=0.9):
    client.post(
        "/api/ingest/prediction",
        json={"label": label, "confidence": confidence, "timestamp_ms": ts},
    )


def test_detection_unavailable_before_any_prediction(client):
    assert detection(client)["state"] == "unavailable"


def test_detection_ready_when_idle(client):
    post_prediction(client, "none", 0)
    assert detection(client)["state"] == "ready"


def test_detection_shows_hold_progress(client):
    for ts in range(0, 801, 100):
        post_prediction(client, "paper", ts)
    result = detection(client)
    assert result["state"] == "recognizing"
    assert result["label"] == "paper"
    assert result["progress"] == pytest.approx(0.53, abs=0.01)


def test_detection_sent_then_back_to_ready(client, clock):
    hold(client, "paper")
    assert detection(client)["state"] == "sent"
    clock.now += 3500
    post_prediction(client, "none", 2000)
    assert detection(client)["state"] == "ready"


def test_detection_brief_wave_is_rejected_and_counted(client):
    for ts in range(0, 501, 100):
        post_prediction(client, "paper", ts)
    post_prediction(client, "none", 600)
    result = detection(client)
    assert result["state"] == "rejected"
    assert result["failed_attempts_recent"] == 1


def test_detection_jitter_is_not_an_attempt(client):
    post_prediction(client, "paper", 0)
    post_prediction(client, "paper", 100)
    post_prediction(client, "none", 200)
    result = detection(client)
    assert result["state"] == "ready"
    assert result["failed_attempts_recent"] == 0


def test_detection_unsure_gesture_is_rejected(client):
    for ts in range(0, 2001, 100):
        post_prediction(client, "scissors", ts, confidence=0.5)
    assert detection(client)["state"] == "rejected"
    assert client.get("/api/requests").json() == []


def test_detection_successful_hold_is_not_a_failure(client):
    hold(client, "paper", duration_ms=3000)
    post_prediction(client, "none", 3100)
    assert detection(client)["failed_attempts_recent"] == 0


def test_detection_failures_expire(client, clock):
    for ts in range(0, 501, 100):
        post_prediction(client, "paper", ts)
    post_prediction(client, "none", 600)
    clock.now += 301_000
    post_prediction(client, "none", 700)
    assert detection(client)["failed_attempts_recent"] == 0


def test_detection_unavailable_when_predictions_stop(client, clock):
    post_prediction(client, "none", 0)
    clock.now += PREDICTION_TIMEOUT_MS + 1
    assert detection(client)["state"] == "unavailable"
