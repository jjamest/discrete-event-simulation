"""Exploratory/regression tests for interrupt() + Resource interaction.

A code reviewer flagged that Simulator.schedule()'s generation-check
mechanism (built to fix a stale-heap-entry bug for sim.sleep()) is also
exercised by Resource.request()/_renege()/release() - and asked whether
interrupting a process queued on Resource.acquire() interacts safely with
a later grant/renege for that same coroutine.

It does NOT interact safely. Empirically, with the current (pre-Task-7,
string-instruction "acquire") Resource implementation:

  1. interrupt() correctly injects an Interrupt into a coroutine suspended
     at the "acquire" yield point (VirtualAcquire.__await__'s bare yield),
     exactly as it does for sim.sleep()'s yield - coro.throw() doesn't care
     what shape the yielded instruction tuple was.

  2. interrupt() has no awareness of Resource internals, so the interrupted
     coroutine is NEVER removed from Resource._waiters. It remains a
     phantom waiter.

  3. What happens when that phantom waiter is later granted a slot depends
     on whether the coroutine finished as a result of the interrupt:

     a. If the interrupt causes the coroutine to finish (caught and the
        handler returns, or uncaught and it propagates out), Simulator
        pops the coroutine from `_process_by_coro` when it finishes. A
        later Resource.release() popping this waiter and calling
        `sim.schedule(delay=0.0, coroutine=waiter.coro, value=True)` then
        snapshots generation `None` (no owning Process is found, same
        bucket as "no Process at all") - so the generation check does NOT
        recognize this as stale. `run()` proceeds to call
        `coro.send(True)`/`coro.throw()` on an already-finished coroutine,
        which raises `RuntimeError: cannot reuse already awaited
        coroutine`. This is a crash, not a silent drop.

     b. If the coroutine survives the interrupt (e.g. it sleeps again
        inside its `except Interrupt` handler), it stays in
        `_process_by_coro` and the generation check *does* engage - but it
        can't distinguish "a stale grant meant for the pre-interrupt
        acquire" from "a legitimate new post-interrupt schedule call",
        because both get snapshotted under the same post-interrupt
        generation. The stale grant is silently delivered early (at the
        grant's own delay=0.0 schedule time, not whenever the coroutine's
        new suspension point would naturally resolve) into whatever new
        suspension point the coroutine has since reached - a premature,
        out-of-order resumption, not merely a dropped one.

  4. Because `Resource.in_use`/`release()`'s waiter-popping already ran
     before the crash/misdelivery, `Resource.in_use` can be left permanently
     elevated (a slot leak) with no live coroutine that will ever call
     `.release()` to give it back.

This is a real gap in the current, pre-Task-7 Resource - not just
theoretical. It is intentionally NOT patched here: fixing it properly
requires interrupt()/Resource cooperation to remove the coroutine from
`_waiters` at interrupt time, which is exactly the design Task 7's
Event-based Resource rewrite already commits to ("interrupting a process
waiting on a Resource/Event removes it from that thing's waiter list
first"). See the KNOWN LIMITATION note on Process.generation in
src/ordo/process.py. These tests act as a tripwire: if Task 7 changes this
behavior (as it should), revisit/update them then.
"""
import pytest

from ordo.exceptions import Interrupt
from ordo.simulator import Simulator


def test_interrupt_injects_correctly_at_acquire_yield_point():
    """Sanity check: coro.throw() injects Interrupt at the "acquire" yield
    exactly as it does at sim.sleep()'s yield - both are bare `yield`
    statements, so this isn't Event-specific machinery.
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


def test_interrupted_waiter_remains_a_phantom_entry_in_resource_waiters():
    """interrupt() never touches Resource._waiters - the coroutine stays
    queued even though it has already been interrupted and (in this test)
    finished.
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
    assert any(w.coro is waiter_coro for w in res._waiters)

    waiter_proc.interrupt(cause="give up")
    sim.run(until=1)

    assert waiter_proc.triggered is True  # coroutine has already finished
    # Known limitation: still present as a phantom waiter.
    assert any(w.coro is waiter_coro for w in res._waiters)


def test_grant_to_interrupted_finished_waiter_crashes_run():
    """KNOWN LIMITATION (pre-Task-7 Resource): granting a slot to a waiter
    that finished as a result of being interrupted crashes the event loop,
    because the generation check can't recognize the grant as stale once
    the coroutine has been popped from `_process_by_coro`.

    This is a tripwire for Task 7's Resource rewrite: once interrupt()
    correctly removes the coroutine from the waiter list up front, this
    scenario becomes impossible (release() will never see this waiter at
    all), and this test should be revisited/updated or deleted.
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

    # Simulate the holder releasing its slot. release() pops the phantom
    # waiter (still in _waiters) and schedules a grant for its coroutine -
    # which has already finished.
    res.release()

    with pytest.raises(RuntimeError, match="cannot reuse already awaited coroutine"):
        sim.run(until=10)

    # The slot was handed to the phantom waiter's grant before the crash;
    # nothing will ever call .release() for it now. Known slot leak.
    assert res.in_use == 1


def test_grant_to_interrupted_surviving_waiter_is_silently_misdelivered():
    """KNOWN LIMITATION (pre-Task-7 Resource): if the interrupted waiter's
    coroutine survives (keeps running after catching Interrupt), the
    generation check does not crash, but it also does not protect against
    the stale grant - it is silently delivered early into whatever new
    suspension point the coroutine has since reached, because both the
    stale grant and the coroutine's legitimate new schedule share the same
    post-interrupt generation.
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
            log.append(("interrupted", sim.now))
            # Survive the interrupt and suspend again elsewhere.
            got = await sim.sleep(50)
            log.append(("resumed_from_sleep", got, sim.now))

    sim.process(holder())
    waiter_coro = waiter()
    waiter_proc = sim.process(waiter_coro)
    sim.run(until=0)

    waiter_proc.interrupt(cause="give up")
    sim.run(until=1)
    assert log == [("interrupted", 0.0)]
    assert waiter_coro in sim._process_by_coro  # still alive, unlike the crash scenario

    # release() grants the phantom waiter a slot. The grant's generation
    # matches the coroutine's current (post-interrupt) generation, so it is
    # NOT dropped - it gets misdelivered as the return value of the
    # unrelated sim.sleep(50) await.
    res.release()
    sim.run(until=10)

    # sim.sleep()'s __await__ discards the sent value (`yield (...)` without
    # capturing it), so the misdelivered grant's `value=True` payload isn't
    # observable here - but the *timing* proves the misdelivery: the
    # coroutine resumes from sim.sleep(50) at t=0 (when the stale grant was
    # scheduled with delay=0.0), not at its real t=50 wakeup.
    assert log == [("interrupted", 0.0), ("resumed_from_sleep", None, 0.0)]
