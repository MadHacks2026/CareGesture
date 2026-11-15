# CareGesture frontend (Member 3)

Patient screen and nurse station for CareGesture, built for Member 2's FastAPI backend (`backend/api.py`). Plain HTML, CSS and JavaScript: no build step, no dependencies. Both screens also run in a **browser-only demo mode** with no backend at all.

> Prototype for a hackathon. Not a certified nurse-call or emergency system. Fictional rooms only.

## Files

| File | What it does |
| --- | --- |
| `patient.html` | Patient screen: ready / recognizing / sent / not recognized / monitoring unavailable, last request with Sent → Seen by nurse → Resolved steps, gesture guide, room conditions |
| `nurse.html` | Nurse station: live request cards, Acknowledge and Resolve, overdue escalation, sound alerts, health chips, room conditions, response-time insights, recently resolved |
| `styles.css` | Shared high-contrast styles, large touch targets, reduced-motion support |
| `api_client.js` | Talks to the backend (event stream with polling fallback) and contains the browser-only mock |
| `index.html` | Launcher with links to both screens |

## Run it with the real backend

1. Start the backend (Member 2), for example on port 8000.
2. Serve this folder from a second terminal:
   ```bash
   cd frontend
   python3 -m http.server 8080
   ```
3. Open, on any device on the same network (replace the address with the UNO Q's or laptop's IP):
   - Patient: `http://<frontend-ip>:8080/patient.html?api=http://<backend-ip>:8000`
   - Nurse: `http://<frontend-ip>:8080/nurse.html?api=http://<backend-ip>:8000`

CORS is already open in `api.py` (`allow_origins=["*"]`), so the different ports work.

**Demo triggers without a camera:** when the backend runs in mock mode (`create_app(backend, mock=True)`), add `&demo=1` to the patient URL. Three "Demo trigger" buttons appear and call `POST /api/demo/trigger`. If the backend isn't in mock mode, the page says so instead of failing silently.

## Run it with no backend (browser-only demo)

Open both in the same browser, side by side:

- `http://localhost:8080/patient.html?mock=1`
- `http://localhost:8080/nurse.html?mock=1`

Everything is simulated in the browser (localStorage, shared between tabs): held gestures with a progress bar, a quick wave that sends nothing, unplugging the camera or sensor, and sample history for the insights panel. Two different devices do not share this mock data. Use the real backend for a two-device demo.

### URL options

| Option | Default | Meaning |
| --- | --- | --- |
| `api=` | same origin | Backend base URL, e.g. `http://192.168.1.50:8000` |
| `demo=1` | off | Show demo-only trigger buttons (calls `/api/demo/trigger`) |
| `mock=1` | off | Browser-only mock backend, no server needed |
| `poll=1` | off | Poll every second instead of using `/api/stream` |
| `room=` | `01` | Room shown on the patient screen |
| `overdue_s=` | `30` | Seconds before a new request is marked overdue (try `10` in a demo) |
| `escalate_s=` | `60` | Seconds before an unanswered request is escalated to the charge nurse station (try `20` in a demo) |
| `station=charge` | off | Nurse page only: show just the escalated requests, as a second simulated station |

## How it uses `api.py`

| Route | Used for |
| --- | --- |
| `GET /api/stream` | Main live feed. One snapshot per second with `requests`, `environment` and `status`. If the stream fails 3 times, is missing (404), or goes quiet for 3.5 s, the page reports a connection problem and falls back to polling. |
| `GET /api/requests`, `/api/environment`, `/api/status` | Polling fallback, first paint, and an immediate refresh after each nurse action |
| `POST /api/requests/{id}/reply` with `{"reply": "on_my_way"}` (a preset code) | Nurse reply buttons. Also acknowledges a new request. 409 once resolved. If the route is missing, the card says replies need a backend update. |
| `PATCH /api/requests/{id}` with `{"status": "acknowledged"}` or `{"status": "resolved"}` | Nurse buttons. The screen then re-reads the backend and never shows a status the backend hasn't confirmed. 409 shows "already updated elsewhere". 404 shows "no longer exists". |
| `POST /api/demo/trigger` with `{"type": "assistance" \| "water" \| "room_comfort"}` | Demo-only buttons (`?demo=1`) |

The `stale` flag from `/api/environment` is trusted over the browser's own clock, because the UNO Q and the viewing device may disagree on the time.

### Field names assumed (check against `backend/schemas.py`)

`schemas.py` wasn't available when this was built, so the frontend reads the names from the team's shared contract. If any differ, change only the `normalize...` functions at the top of `api_client.js`.

- **Request:** `id`, `room_id`, `type`, `status`, `created_at`, `acknowledged_at`, `resolved_at`, `reply`, `reply_at`, `age_ms`. Also accepts `request_id` and `request_type`.
- **Environment:** `temperature_c`, `humidity_pct`, `timestamp_ms`, `sensor_ok`, `stale`.
- **Status:** `camera_ok`, `inference_ok`, `sensor_ok`, `backend_ok`. Missing or `null` shows as "Unknown", never as OK.

### Small backend asks for Member 2 (all optional)

1. **`acknowledged_at` and `resolved_at`** (done in the backend) on each Request. They power the response-time insights and the "took 4 m" times. Without them, the insights show counts only.
2. **Timezone-aware `created_at`** (e.g. `datetime.now(timezone.utc)`). A naive timestamp from a UNO Q set to UTC would look 6 hours off in Chicago.
3. **`environment` snapshot on `room_comfort` requests.** The nurse card then shows conditions at request time. Without it, the card shows the current reading.
4. **`GET /api/detection?room_id=01`** returning `{"state": "ready|recognizing|sent|rejected|unavailable", "label": "paper", "progress": 0.6, "failed_attempts_recent": 2}`. This enables the patient's live hold progress bar, the "Not recognized, try again" message, and the nurse's unrecognized-attempts chip. Without it, the patient screen still shows ready, sent and unavailable.

## Nurse replies

A nurse can send the patient one of a few preset messages from the request card. Presets only, never free text, and none promises a time.

| Code | Offered for | Patient reads |
| --- | --- | --- |
| `on_my_way` | Assistance | A nurse is on the way. |
| `helping_another_patient` | All types | A nurse is helping another patient. Please wait. |
| `use_call_button` | Assistance | Please use your call button. |
| `bringing_water` | Water | A nurse is bringing your water. |
| `adjusting_room` | Room comfort | A nurse is adjusting the room. |

The backend stores the code; the wording lives in `api_client.js` (`REPLIES`, `REPLY_OPTIONS`). Replying to a `new` request also acknowledges it, so there is never a replied-to card still marked new. The latest reply wins. The patient sees it in a banner above the status panel until the request is resolved, with a chime if patient sound is on.

## Escalation (demo thresholds, not clinical rules)

| Level | Default age | What happens |
| --- | --- | --- |
| 1 New | 0 s | Normal card, chime every 60 s |
| 2 Overdue | `overdue_s` (30 s) | Red card, urgent tone every 20 s |
| 3 Escalated | `escalate_s` (60 s) | Dark red pulsing card, siren tone every 10 s, red banner, and the request appears on the charge nurse station (`nurse.html?station=charge`) |

Levels use the backend's `age_ms` (time since creation by the backend's own clock), so a device with a wrong clock cannot hide or fake an escalation and every station agrees. Without `age_ms` (the browser mock) it falls back to `created_at`. Acknowledging or replying at any level removes it from both stations. For a demo, open the nurse page and the charge page with `overdue_s=10&escalate_s=20` (the launcher has links), then make a gesture and wait.

## Rules the UI follows

- **Unknown is never shown as working.** The patient screen says "Monitoring unavailable" unless `camera_ok` and `inference_ok` are both `true`.
- **No optimistic updates.** Statuses come only from the backend.
- **Stale sensor data is labelled stale,** and sensor problems never block requests.
- **No video** reaches either screen.
- **Lost connection** shows a red banner on both screens, and they recover on their own when the backend returns.

## Nurse list order

1. Overdue new requests, oldest first (escalation)
2. Other new requests, newest first
3. Acknowledged requests, oldest first

## Tested

Automated browser tests (Chrome via Playwright):

- **Against a stand-in that reproduces `api.py`'s routes** (24 checks): the event stream is used instead of polling; a demo trigger reaches the nurse; acknowledge reaches the patient; a 409 race shows a notice; overdue escalation; the `stale` flag; camera down; backend down, then recovery after restart; demo trigger on a non-mock backend; polling mode; no JavaScript errors.
- **Browser-only mock** (21 checks): a wave sends nothing; hold progress; the full acknowledge and resolve loop; a refresh keeps the state; sensor and camera unplugged; no backend.
