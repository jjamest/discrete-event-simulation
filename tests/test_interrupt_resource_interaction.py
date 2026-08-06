"""Regression tests for interrupt() + Resource interaction.

A code reviewer flagged that Simulator.schedule()'s generation-check
mechanism (built to fix a stale-heap-entry bug for sim.sleep()) does NOT
protect Resource grants: Resource now (post-Task-7) hands out grants via
Event.succeed(), which the "event" instruction routes through
schedule_call() -> _resume_from_event() -> Simulator._resume() directly.
schedule_call() pushes a plain callback onto the heap (generation=None,
and the dispatch loop's isinstance(target, Coroutine) check never even
looks at generation for callbacks), so the generation mechanism that
protects sim.sleep()'s stale wakeups is bypassed by construction for
Resource grants. A separate fix was required.

Pre-Task-7, this was a real, documented bug: interrupt() had no awareness
of Resource internals, so an interrupted coroutine queued in
Resource._waiters became a permanent "phantom waiter." If later granted a
slot, this either crashed the simulator (RuntimeError: cannot reuse
already awaited coroutine, if the coroutine had already finished) or
silently misdelivered a stale grant into whatever the coroutine was doing
next (if it survived the interrupt) -- and either way leaked a resource
slot, since nothing would ever call .release() for the phantom grant.

Task 7's Event-based Resource rewrite fixes this with two complementary
pieces:

1. _AcquireAwaitable.__await__ wraps its `yield from event.__await__()`
   in a try/except that, on ANY exception propagating through (Interrupt
   or otherwise -- this is general abandonment cleanup, not
   Interrupt-specific): if still queued, removes the waiter's entry from
   Resource._waiters via the same "settled flag + heap removal"
   mechanism already used for timeout-reneges, so a later release() never
   resolves a grant nobody is listening for; if already granted (the
   underlying Event had already fired True) but the coroutine never got
   to act on it (see point 2 below for why that's possible), it calls
   resource.release() itself so the slot isn't orphaned.

2. Simulator._resume's "event" instruction handling now snapshots the
   owning Process's generation when registering the event's fire
   callback, and _resume_from_event checks it before resuming -- mirroring
   the protection run() already gives plain scheduled coroutine
   resumptions. This closes a second, subtler race: release() can grant a
   queued waiter (popping it from _waiters and calling Event.succeed(True)
   on its Event) whose *resumption* is then scheduled via schedule_call
   (bypassing run()'s Coroutine-only generation check by construction) --
   if an interrupt() lands and is processed before that scheduled
   resumption fires, the resumption would otherwise crash (coroutine
   already finished) or misdeliver (coroutine survived and moved on). The
   generation check here drops the stale resumption instead, and piece 1
   above (the result.triggered/ok/value check) is what then notices the
   grant went nowhere and releases the slot back.

These tests assert the FIXED behavior: no crash, no leak, no
misdelivery.
"""
from ordo.exceptions import Interrupt
from ordo.simulator import Simulator


def test_interrupt_injects_correctly_at_acquire_yield_point():
    """Sanity check: coro.throw() injects Interrupt at the Event-based
    acquire suspension point exactly as it does at sim.sleep()'s yield --
    both ultimately suspend at a bare `yield` inside Event.__await__, so
    this isn't Resource-specific machinery.
    """
    sim = Simulator()
    res = sim.resource(capacity=1)
    log = []

    async def holder():
        await res.acquire()

    async def waiter():
        try:
            await res.acquire()
        except Interrupt as e:
            log.append(("interrupted", e.cause, sim.now))

    sim.process(holder())
    waiter_proc = sim.process(waiter())
    sim.run(until=0)  # holder acquires; waiter queues on the exhausted resource

    waiter_proc.interrupt(cause="give up")
    sim.run(until=1)

    assert log == [("interrupted", "give up", 0.0)]
    assert waiter_proc.triggered is True
    assert waiter_proc.ok is True


def test_interrupted_waiter_is_removed_from_resource_waiters():
    """Fixed behavior: interrupting a queued waiter removes its entry from
    Resource._waiters immediately (via the abandonment cleanup in
    _AcquireAwaitable.__await__), so it does not linger as a phantom
    waiter that a later release() might mistakenly grant.
    """
    sim = Simulator()
    res = sim.resource(capacity=1)

    async def holder():
        await res.acquire()

    async def waiter():
        try:
            await res.acquire()
        except Interrupt:
            pass

    sim.process(holder())
    waiter_coro = waiter()
    waiter_proc = sim.process(waiter_coro)
    sim.run(until=0)
    assert len(res._waiters) == 1  # waiter is queued behind the holder

    waiter_proc.interrupt(cause="give up")
    sim.run(until=1)

    assert waiter_proc.triggered is True  # coroutine has already finished
    assert res._waiters == []  # cleaned up, not left as a phantom entry


def test_grant_to_interrupted_finished_waiter_does_not_crash_or_leak():
    """Fixed behavior: a waiter that finishes as a result of being
    interrupted is removed from Resource._waiters before the holder ever
    releases, so release() never attempts to grant it a slot. No crash,
    no slot leak.
    """
    sim = Simulator()
    res = sim.resource(capacity=1)

    async def holder():
        await res.acquire()

    async def waiter():
        try:
            await res.acquire()
        except Interrupt:
            pass  # handler returns -> coroutine finishes here

    sim.process(holder())
    waiter_coro = waiter()
    waiter_proc = sim.process(waiter_coro)
    sim.run(until=0)

    waiter_proc.interrupt(cause="give up")
    sim.run(until=1)
    assert waiter_proc.triggered is True

    # The holder releases its slot. Since the interrupted waiter was
    # already removed from _waiters, there's nobody left to grant it to;
    # the slot count simply drops back to 0.
    res.release()
    sim.run(until=10)  # must not raise

    assert res.in_use == 0  # no leak: the freed slot wasn't handed to a phantom


def test_grant_to_interrupted_surviving_waiter_is_not_misdelivered():
    """Fixed behavior: if the interrupted waiter's coroutine survives
    (keeps running after catching Interrupt) and suspends again elsewhere
    (e.g. sim.sleep), a later release() must not affect it at all -- its
    Resource waiter entry was already removed at interrupt time, so
    release() finds no matching waiter and there's nothing to misdeliver.
    """
    sim = Simulator()
    res = sim.resource(capacity=1)
    log = []

    async def holder():
        await res.acquire()

    async def waiter():
        try:
            await res.acquire()
        except Interrupt:
            log.append(("interrupted", sim.now))
            # Survive the interrupt and suspend again elsewhere.
            got = await sim.sleep(50)
            log.append(("resumed_from_sleep", got, sim.now))

    sim.process(holder())
    waiter_coro = waiter()
    waiter_proc = sim.process(waiter_coro)
    sim.run(until=0)

    # interrupt() only *schedules* its resumption (via schedule_call) --
    # it doesn't run synchronously. So release() must be scheduled to run
    # strictly after the interrupt has actually been processed (otherwise
    # release() would race ahead of the cleanup and grant the still-queued
    # phantom waiter itself, which is a different scenario than the one
    # this test targets). Schedule release() a moment after the interrupt
    # so both are driven by a single run() call in the right order --
    # calling run() a second time with a larger `until` after events have
    # already been popped past their target time is a separate,
    # pre-existing Simulator.run() limitation (heapq.heappop() happens
    # before the `event_time > until` check, discarding the popped-but-
    # deferred event instead of leaving it in the heap) that is out of
    # scope for this Resource fix.
    waiter_proc.interrupt(cause="give up")
    sim.schedule_call(1.0, res.release)
    sim.run(until=60)

    # The sleep(50) resolves at its real, natural wakeup time (t=51: the
    # interrupt happened at t=0, and release() -- which has no waiters to
    # grant to, since the interrupted waiter's entry was already removed
    # -- was scheduled a moment later and has no effect on it), not
    # prematurely at release()'s t=1 -- proving no misdelivery occurred.
    assert log == [("interrupted", 0.0), ("resumed_from_sleep", None, 50.0)]
    assert res._waiters == []  # cleaned up at interrupt time
    assert res.in_use == 0


def test_interrupt_racing_a_just_delivered_grant_does_not_leak_or_crash():
    """A subtler race than the "still queued" scenarios above: the waiter
    has already been granted a slot by release() (its Event fired with
    True, Resource.in_use already reflects the handoff), but the
    *resumption* of its coroutine is a separately scheduled callback
    (Event fire -> schedule_call) that hasn't run yet. If an interrupt()
    is processed first, the grant's resumption becomes stale.

    Fixed behavior: Simulator drops the stale resumption (via the
    generation check on the "event" instruction) instead of crashing on
    an already-finished coroutine, and _AcquireAwaitable notices its
    Event had already resolved to True with nobody left to act on it, so
    it releases the slot itself. No crash, no leak.
    """
    sim = Simulator()
    res = sim.resource(capacity=1)
    log = []

    async def holder():
        await res.acquire()
        await sim.sleep(1)
        res.release()
        log.append(("holder released", sim.now))

    async def waiter():
        try:
            got = await res.acquire()
            log.append(("acquired", got, sim.now))
        except Interrupt:
            log.append(("interrupted", sim.now))

    sim.process(holder())
    waiter_coro = waiter()
    waiter_proc = sim.process(waiter_coro)
    # Scheduled for the same simulated instant as the holder's release
    # (t=1); whichever tiebreaker runs first, the interrupt must still be
    # processed correctly relative to the in-flight grant.
    sim.schedule_call(1.0, lambda: waiter_proc.interrupt(cause="race"))
    sim.run(until=10)  # must not raise

    assert log == [("holder released", 1.0), ("interrupted", 1.0)]
    assert waiter_proc.triggered is True
    assert res.in_use == 0  # no leak: the orphaned grant was released back
    assert res._waiters == []
