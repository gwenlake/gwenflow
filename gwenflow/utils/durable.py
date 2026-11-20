"""Ids and clock that stay deterministic when an agent runs inside a durable workflow."""

import weakref
from datetime import datetime, timezone

from gwenflow.utils.uuid7 import build_uuid7, uuid7


def in_workflow() -> bool:
    try:
        from temporalio import workflow
    except ImportError:
        return False
    return workflow.in_workflow()


# Per-workflow-instance state (last timestamp, last counter), so that ids stay
# monotonic within a run and identical on replay. Never shared between runs.
_workflow_uuid7_state: "weakref.WeakKeyDictionary[object, list[int]]" = weakref.WeakKeyDictionary()


def _workflow_uuid7() -> str:
    """UUIDv7 built from the workflow clock and the workflow's seeded random.

    Same layout and monotonicity rule as `uuid7()` (RFC 9562, method 1), but
    every input comes from Temporal, so a replay produces the same ids.
    """
    from temporalio import workflow

    rng = workflow.random()
    timestamp_ms = int(workflow.now().timestamp() * 1000)
    state = _workflow_uuid7_state.setdefault(workflow.instance(), [-1, 0])
    last_ts, last_counter = state
    if timestamp_ms > last_ts:
        counter = rng.getrandbits(41)  # 42-bit counter, MSB at 0
    else:
        timestamp_ms = last_ts
        counter = last_counter + 1
        if counter > 0x3FF_FFFF_FFFF:
            timestamp_ms += 1
            counter = rng.getrandbits(41)
    state[0], state[1] = timestamp_ms, counter
    return str(build_uuid7(timestamp_ms, counter, rng.getrandbits(32)))


def new_id() -> str:
    """A time-sortable UUIDv7, deterministic inside a workflow."""
    if in_workflow():
        return _workflow_uuid7()
    return str(uuid7())


def now_utc() -> datetime:
    if in_workflow():
        from temporalio import workflow

        return workflow.now()
    return datetime.now(tz=timezone.utc)
