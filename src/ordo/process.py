from ordo.event import Event
from ordo.exceptions import Interrupt


class Process(Event):
    """An Event that fires when the wrapped coroutine finishes.

    Succeeds with the coroutine's return value, or fails with the
    exception it raised.
    """

    __slots__ = ("coro", "_sim", "generation")

    def __init__(self, coro, sim) -> None:
        super().__init__()
        self.coro = coro
        self._sim = sim
        # Bumped on every out-of-band resumption (currently: interrupt()).
        # Heap entries scheduled for `coro` capture the generation at
        # schedule time; if it no longer matches by the time the entry is
        # popped, the entry is stale (superseded by the out-of-band
        # resumption) and must be dropped instead of driving the coroutine
        # again. See Simulator.schedule()/Simulator.run().
        self.generation = 0

    def interrupt(self, cause=None) -> None:
        """Raise Interrupt inside this process's coroutine at its next suspension point."""
        if self.triggered:
            return  # already finished; nothing to interrupt
        # Invalidate any previously scheduled entries for this coroutine
        # (e.g. the pending wakeup from an in-progress sim.sleep()) before
        # scheduling the interrupt's own resumption.
        self.generation += 1
        self._sim.schedule_call(
            0.0,
            lambda: self._sim._resume(self.coro, sent_exception=Interrupt(cause)),
        )

    def __repr__(self) -> str:
        name = getattr(self.coro, "__qualname__", repr(self.coro))
        return f"Process({name})"
