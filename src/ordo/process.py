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
        # they're scheduled via Simulator.schedule() - this isn't specific
        # to sim.sleep(); Resource.request()/_renege()/release() (see
        # resource.py) all go through schedule() too, for immediate grants,
        # reneges, and slot handoffs. If the snapshotted generation no
        # longer matches by the time the entry is popped, the entry is
        # stale (superseded by the out-of-band resumption) and must be
        # dropped instead of driving the coroutine again. See
        # Simulator.schedule()/Simulator.run().
        #
        # KNOWN LIMITATION (pre-Task-7 Resource only): interrupt() has no
        # awareness of Resource internals and never removes the coroutine
        # from Resource._waiters. If a queued waiter is interrupted and
        # then later granted a slot (via release()) or reneges on timeout,
        # the generation check can't reliably save us:
        #   - If the interrupt is caught (or propagates) and the coroutine
        #     finishes as a result, Simulator._resume() pops it from
        #     _process_by_coro entirely, so any later Resource-originated
        #     schedule() call for that same coroutine snapshots generation
        #     None (indistinguishable from "no owning Process") and sails
        #     through the check - resuming an already-finished coroutine,
        #     which raises RuntimeError: cannot reuse already awaited
        #     coroutine.
        #   - If the coroutine survives the interrupt (e.g. it sleeps again
        #     in its except-block), a later Resource-originated grant can
        #     still be delivered into whatever *new* suspension point the
        #     coroutine has since reached, because both were scheduled
        #     under the same post-interrupt generation - a silent
        #     misdelivery, not a drop.
        # This is a real gap in the current (pre-Task-7) Resource, not just
        # a theoretical one - see
        # tests/test_interrupt_resource_interaction.py. It is being
        # deferred rather than patched here because Resource is being
        # rewritten in Task 7 on top of Event, whose design already
        # requires "interrupting a process waiting on a Resource/Event
        # removes it from that thing's waiter list first". Task 7 must
        # implement that removal and include a regression test for this
        # exact interaction.
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
