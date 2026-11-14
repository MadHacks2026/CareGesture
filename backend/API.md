# CareGesture backend API

CareGesture is an accessibility demo. It is **not** a medical or nurse-call
device, and the Thermo reading is room air only, never body temperature.

## Running

```bash
pip install -r backend/requirements.txt
python -m backend.main --mock      # laptop, fake gestures and sensor data
python -m backend.main             # real mode, producers push data in
```

Options: `--host` (default `0.0.0.0`), `--port` (default `8000`).

Base URL: `http://localhost:8000`. CORS allows all origins.
Interactive docs (Swagger) are served at `/docs`, the raw spec at `/openapi.json`.

## Who calls what

| Caller | Routes |
| --- | --- |
| Frontend (Member 3) | `GET /api/requests`, `PATCH /api/requests/{id}`, `GET /api/environment`, `GET /api/status`, `GET /api/stream` |
| Camera + inference (Member 1) | `POST /api/ingest/prediction` |
| Thermo / device health (Member 4) | `POST /api/ingest/environment` |
| Mock mode only | `POST /api/demo/trigger` |

## Data shapes

Do not change a field without telling the whole team.

### Request

```json
{
  "id": "req-0001",
  "room_id": "room-101",
  "type": "assistance",
  "status": "new",
  "created_at": "2026-11-14T18:30:00.123456+00:00"
}
```

| Field | Values |
| --- | --- |
| `type` | `assistance`, `water`, `room_comfort` |
| `status` | `new`, `acknowledged`, `resolved` |
| `created_at` | ISO-8601, UTC |

### GesturePrediction

```json
{ "label": "paper", "confidence": 0.92, "timestamp_ms": 1731609000000 }
```

| Field | Notes |
| --- | --- |
| `label` | `paper`, `scissors`, `rock`, `none` |
| `confidence` | float, 0.0 to 1.0 |
| `timestamp_ms` | integer. Any millisecond clock works (epoch or uptime), only differences are used |

### Environment

```json
{
  "temperature_c": 22.1,
  "humidity_pct": 45.3,
  "timestamp_ms": 1731609000000,
  "sensor_ok": true,
  "stale": false
}
```

When **sending** a reading, only the first four fields matter; `stale` is
ignored. When **reading**:

- `temperature_c`, `humidity_pct`, `timestamp_ms` are `null` if no reading has
  ever arrived.
- `stale` is `true` when the last reading is older than 10 s (measured on the
  backend's clock at arrival, so the board's timestamp can be uptime).
- `sensor_ok` is `false` if the sensor reported a failure, or the reading is
  stale or missing.

Show stale values as stale, never as current.

### Status

```json
{ "camera_ok": true, "inference_ok": true, "sensor_ok": true, "backend_ok": true }
```

`camera_ok` and `inference_ok` are `false` if no prediction has arrived for
more than 3 s (or none ever has). The backend has one prediction stream, so it
cannot tell a dead camera from dead inference and both follow the same
freshness. The UI must not imply monitoring when either is `false`.

## Frontend routes

### `GET /api/requests`

All requests, newest first. Returns `Request[]`.

### `PATCH /api/requests/{id}`

The only place a status changes: a nurse clicked a button. Never call it
automatically.

Body:

```json
{ "status": "acknowledged" }
```

Allowed moves are `new` to `acknowledged`, then `acknowledged` to `resolved`.
Skipping a step, repeating one, or going backwards is rejected.

| Code | Meaning |
| --- | --- |
| 200 | Updated `Request` |
| 404 | `request not found` |
| 409 | Invalid transition, for example `cannot change new -> resolved` |
| 422 | Body is not a valid status value |

### `GET /api/environment`

Returns `Environment`. Always succeeds. Before any reading arrives it returns
`sensor_ok: false`, `stale: true` and null values.

### `GET /api/status`

Returns `Status`.

### `GET /api/stream`

Server-Sent Events. One message per second, each a full snapshot:

```
data: {"requests": [...], "environment": {...}, "status": {...}}
```

```js
const source = new EventSource("http://localhost:8000/api/stream");
source.onmessage = (e) => {
  const { requests, environment, status } = JSON.parse(e.data);
};
```

Polling the three GET routes once a second works just as well.

## Producer routes

### `POST /api/ingest/prediction`

Member 1 sends one `GesturePrediction` per camera frame. Response:

```json
{ "request": null }
```

`request` is a `Request` only on the call that confirms a hold, otherwise
`null`. Sending predictions never looks at the sensor, so a missing sensor
cannot block a request.

#### How a prediction becomes a request

1. `none` and anything below confidence **0.75** is ignored.
2. The same gesture must be held for **1.5 s**.
3. One hold creates exactly one request.
4. After firing, nothing fires again until `none` has been seen continuously
   for **1 s** (the patient lowered their hand).
5. A gap of more than **500 ms** between predictions, or a timestamp going
   backwards, discards any hold in progress. Raise `MAX_GAP_MS` if the model
   runs below about 3 frames per second.
6. Complete silence for 1 s also counts as the neutral period.

| Gesture | Label | Request type |
| --- | --- | --- |
| Open palm | `paper` | `assistance` |
| Two fingers | `scissors` | `water` |
| Closed fist | `rock` | `room_comfort` |

The thresholds are constants at the top of
[gesture_state_machine.py](gesture_state_machine.py).

### `POST /api/ingest/environment`

Member 4 sends one `Environment` per Thermo reading. Response: `{ "ok": true }`.
Send `sensor_ok: false` when the sensor fails, rather than going silent.

## Mock mode only

### `POST /api/demo/trigger`

**Demo-only.** Creates a request without any camera. The route does not exist
unless the server was started with `--mock`. Label anything that uses it as a
demo trigger in the UI.

Body: `{ "type": "water" }`. Response:

```json
{ "demo_only": true, "request": { "id": "req-0002", "...": "..." } }
```

### Mock script

`--mock` also starts two background threads that feed fake data in a loop:

- **Predictions** every 0.1 s, running through a real hold (assistance), a
  brief wave (nothing), a low-confidence hold (nothing), a real hold (water),
  a real hold (room_comfort), then 20 s quiet before repeating.
- **Environment** every 2 s, around 22 C and 45 % humidity.

## Notes

- Requests live in memory and are lost on restart.
- The room id is fixed at `room-101`.
- Errors use FastAPI's default shape: `{ "detail": "..." }`.
