from typing import Any, Callable, Optional


class Event:
    """A one-shot signal that coroutines can await.

    Triggers exactly once via succeed()/fail(). Awaiting an already-
    triggered event resolves immediately; awaiting one that hasn't
    fired yet suspends until it does (driven by the Simulator's event
    loop).
    """

    __slots__ = ("triggered", "ok", "value", "exception", "_callbacks")

    def __init__(self) -> None:
        self.triggered = False
        self.ok: Optional[bool] = None
        self.value: Any = None
        self.exception: Optional[BaseException] = None
        self._callbacks: list[Callable[["Event"], None]] = []

    def succeed(self, value: Any = None) -> None:
        if self.triggered:
            raise RuntimeError("Event already triggered")
        self.triggered = True
        self.ok = True
        self.value = value
        self._fire_callbacks()

    def fail(self, exception: BaseException) -> None:
        if self.triggered:
            raise RuntimeError("Event already triggered")
        self.triggered = True
        self.ok = False
        self.exception = exception
        self._fire_callbacks()

    def _fire_callbacks(self) -> None:
        callbacks, self._callbacks = self._callbacks, []
        for cb in callbacks:
            cb(self)

    def add_callback(self, cb: Callable[["Event"], None]) -> None:
        """Register a callback(event) to run when the event fires.

        If already triggered, the callback runs immediately.
        """
        if self.triggered:
            cb(self)
        else:
            self._callbacks.append(cb)

    def __await__(self):
        result = yield ("event", self)
        return result
