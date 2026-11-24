import asyncio
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any


class NodeTimeoutError(TimeoutError):
    def __init__(self, timeout: float | None, still_running: bool = False):
        message = f"timed out after {timeout:g}s" if timeout else "timed out"
        if still_running:
            message += ", still running in the background: not retried so that it does not run twice"
        super().__init__(message)
        self.timeout = timeout
        self.still_running = still_running


class NodeCancelledError(Exception):
    def __init__(self) -> None:
        super().__init__("cancelled")


class NodeControl:
    def __init__(self, timeout: float | None = None):
        self.timeout = timeout
        self.deadline = time.monotonic() + timeout if timeout else None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def stopped(self) -> bool:
        return self._stop.is_set() or self.expired()

    def raise_if_stopped(self) -> None:
        if self.expired():
            raise NodeTimeoutError(self.timeout)
        if self._stop.is_set():
            raise NodeCancelledError()

    def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        token = _current.set(self)
        try:
            return fn(*args)
        finally:
            _current.reset(token)


_current: ContextVar[NodeControl | None] = ContextVar("gwenflow_node_control", default=None)


def _retrieve(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


def start_in_thread(control: NodeControl, fn: Callable[..., Any], *args: Any) -> asyncio.Future:
    future = asyncio.ensure_future(asyncio.to_thread(control.run, fn, *args))
    future.add_done_callback(_retrieve)
    return future


def remaining_time() -> float | None:
    control = _current.get()
    return control.remaining() if control is not None else None


def stop_requested() -> bool:
    control = _current.get()
    return control is not None and control.stopped()


def raise_if_stopped() -> None:
    control = _current.get()
    if control is not None:
        control.raise_if_stopped()


def bounded_timeout(timeout: float | None) -> float | None:
    left = remaining_time()
    if left is None:
        return timeout
    return left if timeout is None else min(timeout, left)
