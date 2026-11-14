"""Status rules: new -> acknowledged -> resolved, nothing else."""

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
