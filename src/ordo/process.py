from ordo.event import Event
from ordo.exceptions import Interrupt


class Process(Event):
    """An Event that fires when the wrapped coroutine finishes.

    Succeeds with the coroutine's return value, or fails with the
    exception it raised.
    """

    __slots__ = ("coro", "_sim")

    def __init__(self, coro, sim) -> None:
        super().__init__()
        self.coro = coro
        self._sim = sim

    def interrupt(self, cause=None) -> None:
        """Raise Interrupt inside this process's coroutine at its next suspension point."""
        if self.triggered:
            return  # already finished; nothing to interrupt
        self._sim.schedule_call(
            0.0,
            lambda: self._sim._resume(self.coro, sent_exception=Interrupt(cause)),
        )

    def __repr__(self) -> str:
        name = getattr(self.coro, "__qualname__", repr(self.coro))
        return f"Process({name})"
