"""Status rules: new -> acknowledged -> resolved, nothing else."""

from datetime import datetime, timedelta, timezone

import pytest

from backend.request_store import InvalidTransition, RequestNotFound, RequestStore


def test_create_gives_unique_ids_and_new_status():
    store = RequestStore()
    first = store.create("water")
    second = store.create("water")
    assert first.id != second.id
    assert first.status == "new"
    assert first.type == "water"
    assert first.room_id


def test_list_is_newest_first():
    store = RequestStore()
    first = store.create("water")
    second = store.create("assistance")
    assert [r.id for r in store.list()] == [second.id, first.id]


def test_valid_transitions():
    store = RequestStore()
    request = store.create("assistance")
    assert store.transition(request.id, "acknowledged").status == "acknowledged"
    assert store.transition(request.id, "resolved").status == "resolved"
    assert store.get(request.id).status == "resolved"


@pytest.mark.parametrize(
    "path, bad",
    [
        ([], "resolved"),                              # skipping a step
        ([], "new"),                                   # repeating
        (["acknowledged"], "acknowledged"),            # repeating
        (["acknowledged"], "new"),                     # going backwards
        (["acknowledged", "resolved"], "acknowledged"),  # resolved is final
        (["acknowledged", "resolved"], "resolved"),
    ],
)
def test_invalid_transitions_rejected(path, bad):
    store = RequestStore()
    request = store.create("assistance")
    for status in path:
        store.transition(request.id, status)
    before = store.get(request.id).status
    with pytest.raises(InvalidTransition):
        store.transition(request.id, bad)
    assert store.get(request.id).status == before


def test_unknown_id():
    with pytest.raises(RequestNotFound):
        RequestStore().transition("req-9999", "acknowledged")


def test_reply_on_new_request_also_acknowledges():
    store = RequestStore()
    request = store.create("water")
    updated = store.reply(request.id, "bringing_water")
    assert updated.status == "acknowledged"
    assert updated.reply == "bringing_water"
    assert updated.reply_at == updated.acknowledged_at


def test_reply_on_acknowledged_request_keeps_the_original_acknowledgement():
    store = RequestStore()
    request = store.create("assistance")
    acked = store.transition(request.id, "acknowledged")
    updated = store.reply(request.id, "on_my_way")
    assert updated.status == "acknowledged"
    assert updated.acknowledged_at == acked.acknowledged_at


def test_reply_is_rejected_once_resolved_or_unknown():
    store = RequestStore()
    request = store.create("assistance")
    store.transition(request.id, "acknowledged")
    store.transition(request.id, "resolved")
    with pytest.raises(InvalidTransition):
        store.reply(request.id, "on_my_way")
    with pytest.raises(RequestNotFound):
        store.reply("req-9999", "on_my_way")


def test_age_ms_counts_from_creation_by_the_backends_clock():
    store = RequestStore()
    request = store.create("water")
    assert 0 <= request.age_ms < 5000
    ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
    store._requests[request.id] = request.model_copy(
        update={"created_at": ten_minutes_ago.isoformat()}
    )
    assert store.get(request.id).age_ms >= 600_000
    assert store.list()[0].age_ms >= 600_000
    assert store.transition(request.id, "acknowledged").age_ms >= 600_000
