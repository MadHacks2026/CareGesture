/* CareGesture frontend API client (Member 3).
 *
 * Real mode (default) talks to Member 2's FastAPI backend (backend/api.py):
 *   GET   /api/stream               Server-Sent Events, one snapshot per second (preferred)
 *   GET   /api/requests             list of requests (polling fallback + refresh)
 *   PATCH /api/requests/{id}        body {"status":"acknowledged"|"resolved"}; 409 = invalid transition
 *   GET   /api/environment          latest room reading, with a "stale" flag
 *   GET   /api/status               device health
 *   POST  /api/demo/trigger         DEMO-ONLY, body {"type": ...}; exists only when the backend runs in mock mode
 *   GET   /api/detection?room_id=   optional, not in api.py yet; the UI works without it
 *
 * Browser mock mode (?mock=1): an in-browser fake backend stored in localStorage. Every tab
 * in the same browser shares it, so patient.html and nurse.html work with no backend at all.
 *
 * URL options: ?mock=1  ?api=http://192.168.1.50:8000  ?room=01  ?overdue_s=180  ?demo=1  ?poll=1
 */
(function (global) {
  'use strict';

  const params = new URLSearchParams(global.location.search);
  const MOCK = params.get('mock') === '1';
  const DEMO = params.get('demo') === '1';          // show demo-only trigger buttons against a real backend
  const FORCE_POLL = params.get('poll') === '1';     // skip the event stream, poll instead
  const STREAM_SILENCE_MS = 3500;                    // no snapshot for this long = connection problem
  const API_BASE = (params.get('api') || '').replace(/\/+$/, '');
  const ROOM_ID = params.get('room') || '01';
  const POLL_MS = 1000;
  const DETECTION_POLL_MS = 300;
  const STALE_MS = 10000;
  const OVERDUE_MS = Math.max(10, Number(params.get('overdue_s') || 180)) * 1000;
  const FETCH_TIMEOUT_MS = 3000;
  const HOLD_MS = 1500;

  const REQUEST_TYPES = {
    assistance: { label: 'Assistance', patientText: 'Assistance requested', gesture: 'paper', gestureName: 'Open palm', icon: '✋' },
    water: { label: 'Water', patientText: 'Water requested', gesture: 'scissors', gestureName: 'Two fingers', icon: '✌️' },
    room_comfort: { label: 'Room comfort', patientText: 'Room comfort requested', gesture: 'rock', gestureName: 'Closed fist', icon: '✊' },
  };
  const GESTURE_TO_TYPE = { paper: 'assistance', scissors: 'water', rock: 'room_comfort' };
  const STATUSES = ['new', 'acknowledged', 'resolved'];

  // ---------------------------------------------------------------- helpers

  function toMs(value) {
    if (value === null || value === undefined || value === '') return null;
    if (typeof value === 'number') return value;
    const t = Date.parse(value);
    return Number.isNaN(t) ? null : t;
  }

  function num(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function triState(value) {
    // true / false / null. Unknown must never be shown as working.
    return value === true ? true : value === false ? false : null;
  }

  function normalizeRequest(r) {
    const created = r.created_at ?? r.timestamp ?? null;
    return {
      id: String(r.id ?? r.request_id),
      room_id: r.room_id === undefined || r.room_id === null ? '' : String(r.room_id),
      type: r.type ?? r.request_type ?? 'assistance',
      status: STATUSES.includes(r.status) ? r.status : 'new',
      created_at: created,
      created_ms: toMs(created),
      acknowledged_ms: toMs(r.acknowledged_at),
      resolved_ms: toMs(r.resolved_at),
      confidence: num(r.confidence),
      environment: r.environment ? normalizeEnvironment(r.environment) : null,
    };
  }

  function normalizeEnvironment(e) {
    if (!e) return null;
    return {
      temperature_c: num(e.temperature_c),
      humidity_pct: num(e.humidity_pct),
      timestamp_ms: toMs(e.timestamp_ms),
      age_ms: num(e.age_ms),
      sensor_ok: e.sensor_ok === true,
      stale: typeof e.stale === 'boolean' ? e.stale : null, // backend's own staleness verdict
    };
  }

  function normalizeStatus(s) {
    s = s || {};
    return {
      camera_ok: triState(s.camera_ok),
      inference_ok: triState(s.inference_ok),
      sensor_ok: triState(s.sensor_ok),
      backend_ok: triState(s.backend_ok),
    };
  }

  function normalizeDetection(d) {
    if (!d) return null;
    const progress = num(d.progress);
    return {
      state: ['ready', 'recognizing', 'sent', 'rejected', 'unavailable'].includes(d.state) ? d.state : 'ready',
      label: d.label || null,
      progress: progress === null ? null : Math.min(1, Math.max(0, progress)),
      failed_attempts_recent: num(d.failed_attempts_recent),
    };
  }

  function readingAgeMs(env, now) {
    if (!env) return null;
    if (env.age_ms !== null && env.age_ms !== undefined) return env.age_ms;
    if (env.timestamp_ms === null) return null;
    const age = now - env.timestamp_ms;
    // A wildly negative or huge age means the UNO Q and this device disagree on the time.
    if (age < -5000 || age > 86400000) return null;
    return Math.max(0, age);
  }

  function isEnvironmentStale(env, now) {
    if (!env || env.temperature_c === null || env.sensor_ok !== true) return true;
    if (env.stale !== null && env.stale !== undefined) return env.stale; // trust the backend, it owns the clock
    const age = readingAgeMs(env, now);
    return age === null || age > STALE_MS;
  }

  function formatClock(ms) {
    if (ms === null || ms === undefined) return '';
    return new Date(ms).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }

  function formatDuration(ms) {
    if (ms === null || ms === undefined || ms < 0) return '';
    const s = Math.round(ms / 1000);
    if (s < 60) return s + ' s';
    const m = Math.floor(s / 60);
    if (m < 60) return m + ' m ' + (s % 60) + ' s';
    return Math.floor(m / 60) + ' h ' + (m % 60) + ' m';
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function typeInfo(type) {
    return REQUEST_TYPES[type] || { label: type, patientText: type, gesture: '', gestureName: '', icon: '' };
  }

  // ---------------------------------------------------------------- real backend

  async function fetchJSON(path, options) {
    options = options || {};
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
    const headers = options.body ? { 'Content-Type': 'application/json' } : {};
    try {
      const res = await fetch(API_BASE + path, { method: options.method || 'GET', body: options.body, headers, signal: ctrl.signal, cache: 'no-store' });
      if (!res.ok) {
        const err = new Error('HTTP ' + res.status + ' on ' + path);
        err.status = res.status;
        throw err;
      }
      if (res.status === 204) return null;
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  let detectionSupported = true;

  const realApi = {
    async getRequests() {
      const data = await fetchJSON('/api/requests');
      const list = Array.isArray(data) ? data : (data && data.requests) || [];
      return list.map(normalizeRequest);
    },
    async updateRequest(id, status) {
      if (!['acknowledged', 'resolved'].includes(status)) throw new Error('Invalid status ' + status);
      const data = await fetchJSON('/api/requests/' + encodeURIComponent(id), { method: 'PATCH', body: JSON.stringify({ status }) });
      return data ? normalizeRequest(data) : null;
    },
    async getEnvironment() {
      return normalizeEnvironment(await fetchJSON('/api/environment'));
    },
    async getStatus() {
      return normalizeStatus(await fetchJSON('/api/status'));
    },
    async demoTrigger(type) {
      try {
        const data = await fetchJSON('/api/demo/trigger', { method: 'POST', body: JSON.stringify({ type }) });
        return data && data.request ? normalizeRequest(data.request) : null;
      } catch (err) {
        if (err.status === 404) err.message = 'The backend is not running in mock mode, so demo triggers are off.';
        throw err;
      }
    },
    async getDetection() {
      if (!detectionSupported) return null;
      try {
        return normalizeDetection(await fetchJSON('/api/detection?room_id=' + encodeURIComponent(ROOM_ID)));
      } catch (err) {
        if (err.status === 404) {
          detectionSupported = false;
          return null;
        }
        throw err;
      }
    },
  };

  // ---------------------------------------------------------------- mock backend

  const MOCK_KEY = 'caregesture_mock_v1';
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function mockDefault() {
    return {
      nextId: 1,
      requests: [],
      detection: { state: 'ready', label: null, started_ms: 0, ended_ms: 0, brief: false },
      failed: [],
      cameraOk: true,
      sensorOk: true,
      frozenEnv: null,
    };
  }

  function mockLoad() {
    try {
      return JSON.parse(global.localStorage.getItem(MOCK_KEY)) || mockDefault();
    } catch (e) {
      return mockDefault();
    }
  }

  function mockSave(state) {
    global.localStorage.setItem(MOCK_KEY, JSON.stringify(state));
  }

  function mockLiveEnv(now) {
    return {
      temperature_c: Math.round((23.4 + 0.6 * Math.sin(now / 60000)) * 10) / 10,
      humidity_pct: Math.round((42 + 2 * Math.sin(now / 90000)) * 10) / 10,
      timestamp_ms: now,
      sensor_ok: true,
    };
  }

  function mockEnv(state, now) {
    if (state.sensorOk) return mockLiveEnv(now);
    const frozen = state.frozenEnv || mockLiveEnv(now - 60000);
    return Object.assign({}, frozen, { sensor_ok: false });
  }

  function mockCreateRequest(state, type, now, extra) {
    const id = 'req_' + String(state.nextId++).padStart(3, '0');
    const req = Object.assign(
      {
        id,
        room_id: ROOM_ID,
        type,
        status: 'new',
        created_at: new Date(now).toISOString(),
        acknowledged_at: null,
        resolved_at: null,
        confidence: Math.round((0.86 + Math.random() * 0.12) * 100) / 100,
        environment: type === 'room_comfort' ? mockEnv(state, now) : null,
      },
      extra || {}
    );
    state.requests.push(req);
    return req;
  }

  // Mirrors Member 2's state machine rules: a steady 1.5 s hold creates exactly one
  // request; a brief or uncertain movement is rejected and creates nothing.
  function mockAdvance(state, now) {
    const d = state.detection;
    if (!state.cameraOk) return;
    if (d.state === 'recognizing') {
      if (d.brief && now - d.started_ms >= 600) {
        d.state = 'rejected';
        d.ended_ms = now;
        state.failed.push(now);
      } else if (!d.brief && now - d.started_ms >= HOLD_MS) {
        mockCreateRequest(state, GESTURE_TO_TYPE[d.label], now);
        d.state = 'sent';
        d.ended_ms = now;
      }
    } else if ((d.state === 'sent' || d.state === 'rejected') && now - d.ended_ms >= 2500) {
      d.state = 'ready';
      d.label = null;
    }
    state.failed = state.failed.filter((t) => now - t < 600000);
  }

  function mockTransaction(fn) {
    const state = mockLoad();
    const now = Date.now();
    mockAdvance(state, now);
    const result = fn(state, now);
    mockSave(state);
    return result;
  }

  const mockApi = {
    async getRequests() {
      await sleep(60);
      return mockTransaction((s) => s.requests.map(normalizeRequest));
    },
    async updateRequest(id, status) {
      await sleep(150);
      return mockTransaction((s, now) => {
        const req = s.requests.find((r) => r.id === id);
        if (!req) {
          const err = new Error('Not found');
          err.status = 404;
          throw err;
        }
        const allowed = (req.status === 'new' && status === 'acknowledged') || (req.status === 'acknowledged' && status === 'resolved');
        if (!allowed) {
          const err = new Error('Invalid transition ' + req.status + ' -> ' + status);
          err.status = 409;
          throw err;
        }
        req.status = status;
        if (status === 'acknowledged') req.acknowledged_at = new Date(now).toISOString();
        if (status === 'resolved') req.resolved_at = new Date(now).toISOString();
        return normalizeRequest(req);
      });
    },
    async getEnvironment() {
      await sleep(40);
      return mockTransaction((s, now) => normalizeEnvironment(mockEnv(s, now)));
    },
    async getStatus() {
      await sleep(40);
      return mockTransaction((s) => normalizeStatus({ camera_ok: s.cameraOk, inference_ok: s.cameraOk, sensor_ok: s.sensorOk, backend_ok: true }));
    },
    async demoTrigger(type) {
      await sleep(60);
      return mockTransaction((s, now) => normalizeRequest(mockCreateRequest(s, type, now)));
    },
    async getDetection() {
      return mockTransaction((s, now) => {
        const d = s.detection;
        const recent = s.failed.length;
        if (!s.cameraOk) return normalizeDetection({ state: 'unavailable', failed_attempts_recent: recent });
        const progress = d.state === 'recognizing' ? (now - d.started_ms) / HOLD_MS : null;
        return normalizeDetection({ state: d.state, label: d.label, progress, failed_attempts_recent: recent });
      });
    },
  };

  const mockControls = {
    holdGesture(gesture) {
      mockTransaction((s, now) => {
        if (!s.cameraOk || s.detection.state === 'recognizing') return;
        s.detection = { state: 'recognizing', label: gesture, started_ms: now, ended_ms: 0, brief: false };
      });
    },
    briefWave() {
      mockTransaction((s, now) => {
        if (!s.cameraOk || s.detection.state === 'recognizing') return;
        s.detection = { state: 'recognizing', label: 'paper', started_ms: now, ended_ms: 0, brief: true };
      });
    },
    setCameraOk(ok) {
      mockTransaction((s) => {
        s.cameraOk = !!ok;
        s.detection = { state: 'ready', label: null, started_ms: 0, ended_ms: 0, brief: false };
      });
    },
    setSensorOk(ok) {
      mockTransaction((s, now) => {
        if (s.sensorOk && !ok) s.frozenEnv = mockLiveEnv(now);
        s.sensorOk = !!ok;
      });
    },
    getFlags() {
      const s = mockLoad();
      return { cameraOk: s.cameraOk, sensorOk: s.sensorOk };
    },
    seedHistory() {
      // Past resolved requests so the insights panel has something to show.
      mockTransaction((s, now) => {
        const plan = [
          ['assistance', 115, 40, 300], ['water', 104, 210, 420], ['room_comfort', 96, 95, 600],
          ['water', 80, 260, 380], ['assistance', 66, 35, 240], ['water', 52, 190, 330],
          ['assistance', 40, 55, 410], ['room_comfort', 25, 120, 500],
        ];
        plan.forEach(([type, minutesAgo, ackSec, resolveSec]) => {
          const created = now - minutesAgo * 60000;
          mockCreateRequest(s, type, created, {
            status: 'resolved',
            acknowledged_at: new Date(created + ackSec * 1000).toISOString(),
            resolved_at: new Date(created + resolveSec * 1000).toISOString(),
          });
        });
        s.requests.sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at));
      });
    },
    reset() {
      mockSave(mockDefault());
    },
  };

  const api = MOCK ? mockApi : realApi;

  // ---------------------------------------------------------------- live updates
  //
  // live(onSnapshot, onFail) delivers {requests, environment, status} about once a second.
  // Real backend: the /api/stream event stream, falling back to polling if the stream keeps
  // failing. Browser mock: polling. environment may be undefined, meaning "no new reading".
  // Returns {refresh, mode}; refresh() fetches a fresh snapshot right away (used after a PATCH).

  async function pollOnce() {
    const [requests, status] = await Promise.all([api.getRequests(), api.getStatus()]);
    let environment;
    try {
      environment = await api.getEnvironment();
    } catch (err) {
      environment = undefined; // keep the last reading; the page shows it as stale
    }
    return { requests, environment, status };
  }

  function live(onSnapshot, onFail) {
    const handle = { mode: 'poll', refresh: null };
    let polling = false;

    async function refresh() {
      try {
        onSnapshot(await pollOnce());
      } catch (err) {
        onFail(err);
      }
    }
    handle.refresh = refresh;

    function startPolling() {
      if (polling) return;
      polling = true;
      handle.mode = 'poll';
      (async function tick() {
        await refresh();
        setTimeout(tick, POLL_MS);
      })();
    }

    if (MOCK || FORCE_POLL || !global.EventSource) {
      startPolling();
      return handle;
    }

    handle.mode = 'stream';
    let lastMessage = Date.now();
    let errorsWithoutMessage = 0;
    const source = new global.EventSource(API_BASE + '/api/stream');

    source.onmessage = (event) => {
      lastMessage = Date.now();
      errorsWithoutMessage = 0;
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      onSnapshot({
        requests: (data.requests || []).map(normalizeRequest),
        environment: data.environment ? normalizeEnvironment(data.environment) : undefined,
        status: normalizeStatus(data.status),
      });
    };

    source.onerror = () => {
      errorsWithoutMessage++;
      // CLOSED means the browser gave up for good (e.g. 404 or wrong content type): it won't retry.
      if (errorsWithoutMessage >= 3 || source.readyState === global.EventSource.CLOSED) {
        source.close();
        clearInterval(watchdog);
        startPolling(); // stream keeps failing: polling also detects when the backend returns
      }
    };

    // A stream can stall without raising an error, so watch for silence too.
    const watchdog = setInterval(() => {
      if (Date.now() - lastMessage > STREAM_SILENCE_MS) onFail(new Error('No update from /api/stream'));
    }, 1000);

    refresh(); // first paint without waiting for the stream
    return handle;
  }

  global.CareAPI = {
    MOCK,
    DEMO,
    live,
    demoTrigger: api.demoTrigger,
    API_BASE,
    ROOM_ID,
    POLL_MS,
    DETECTION_POLL_MS,
    STALE_MS,
    OVERDUE_MS,
    REQUEST_TYPES,
    getRequests: api.getRequests,
    updateRequest: api.updateRequest,
    getEnvironment: api.getEnvironment,
    getStatus: api.getStatus,
    getDetection: api.getDetection,
    mock: MOCK ? mockControls : null,
    util: { formatClock, formatDuration, escapeHtml, typeInfo, readingAgeMs, isEnvironmentStale },
  };
})(window);
