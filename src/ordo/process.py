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
        # Heap entries capture the generation at schedule time whenever
        # they're scheduled via Simulator.schedule() (used by sim.sleep()
        # and Simulator.process()'s initial kickoff). If the snapshotted
        # generation no longer matches by the time the entry is popped,
        # the entry is stale (superseded by the out-of-band resumption)
        # and must be dropped instead of driving the coroutine again. See
        # Simulator.schedule()/Simulator.run().
        #
        # Resource (resource.py) does NOT go through Simulator.schedule()
        # for grants/reneges/handoffs - as of Task 7 it resolves an Event
        # instead, and Event-triggered resumptions are scheduled via
        # schedule_call() (a plain callback, not a Coroutine heap target),
        # which bypasses run()'s Coroutine-only generation check by
        # construction. Simulator._resume's "event" instruction handling
        # snapshots and checks this same generation counter itself (see
        # `_resume_from_event`) to close that gap, so an interrupt()
        # racing a Resource grant is still caught: a stale grant
        # resumption is dropped rather than crashing on an already-
        # finished coroutine or misdelivering into a new suspension
        # point. Resource._AcquireAwaitable additionally notices when its
        # own Event resolved to a grant that never got delivered (because
        # it was dropped as stale) and releases the slot back, so a
        # dropped grant doesn't leak Resource.in_use either. Interrupting
        # a process still queued in Resource._waiters (never granted at
        # all) is handled separately: the waiter's entry is removed from
        # _waiters as soon as the interrupt propagates through
        # _AcquireAwaitable.__await__, so a later release() never sees it.
        # See tests/test_interrupt_resource_interaction.py for the full
        # set of regression scenarios.
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
