from ordo.exceptions import Interrupt
from ordo.simulator import Simulator
from ordo.store import Store, TIMEOUT


def test_get_blocks_until_put():
    sim = Simulator()
    store = Store(sim)
    log = []

    async def consumer():
        item = await store.get()
        log.append((item, sim.now))

    async def producer():
        await sim.sleep(3)
        await store.put("widget")

    sim.process(consumer())
    sim.process(producer())
    sim.run(until=10)
    assert log == [("widget", 3.0)]


def test_put_blocks_when_at_capacity():
    sim = Simulator()
    store = Store(sim, capacity=1)
    log = []

    async def producer():
        await store.put("a")
        log.append(("put a", sim.now))
        await store.put("b")  # blocks until capacity frees
        log.append(("put b", sim.now))

    async def consumer():
        await sim.sleep(5)
        item = await store.get()
        log.append((f"got {item}", sim.now))

    sim.process(producer())
    sim.process(consumer())
    sim.run(until=10)
    assert log == [("put a", 0.0), ("got a", 5.0), ("put b", 5.0)]


def test_get_timeout_returns_sentinel():
    sim = Simulator()
    store = Store(sim)
    log = []

    async def consumer():
        item = await store.get(timeout=3)
        log.append(item is TIMEOUT)

    sim.process(consumer())
    sim.run(until=10)
    assert log == [True]


def test_fifo_order():
    sim = Simulator()
    store = Store(sim)
    log = []

    async def producer():
        await store.put("first")
        await store.put("second")

    async def consumer():
        await sim.sleep(1)
        log.append(await store.get())
        log.append(await store.get())

    sim.process(producer())
    sim.process(consumer())
    sim.run(until=10)
    assert log == ["first", "second"]


def test_put_accepts_priority_kwarg_without_crashing():
    """priority is accepted per the plan's sketch; not deeply tested here
    (Store, unlike Resource, isn't asked to prove priority ordering in
    this task).
    """
    sim = Simulator()
    store = Store(sim, capacity=1)
    log = []

    async def producer():
        await store.put("a")  # fills capacity
        await store.put("b", priority=5)
        log.append(("put b", sim.now))

    async def consumer():
        await sim.sleep(1)
        await store.get(priority=1)

    sim.process(producer())
    sim.process(consumer())
    sim.run(until=10)
    assert log == [("put b", 1.0)]


def test_interrupted_queued_getter_is_removed_from_waiters():
    """A getter that blocks (store is empty) and is then interrupted before
    any item arrives must be removed from _get_waiters -- otherwise it
    would linger as a phantom waiter that a later put() might try to
    resolve.
    """
    sim = Simulator()
    store = Store(sim)

    async def waiter():
        try:
            await store.get()
        except Interrupt:
            pass

    waiter_coro = waiter()
    waiter_proc = sim.process(waiter_coro)
    sim.run(until=0)
    assert len(store._get_waiters) == 1

    waiter_proc.interrupt(cause="give up")
    sim.run(until=1)

    assert waiter_proc.triggered is True
    assert store._get_waiters == []


def test_put_after_interrupted_getter_does_not_crash_or_misdeliver():
    """After the queued getter above is interrupted and removed, a put()
    must not find a phantom waiter to deliver to -- the item should just
    sit in the store for a future get().
    """
    sim = Simulator()
    store = Store(sim)
    log = []

    async def waiter():
        try:
            await store.get()
            log.append("got item unexpectedly")
        except Interrupt:
            log.append(("interrupted", sim.now))

    waiter_coro = waiter()
    waiter_proc = sim.process(waiter_coro)
    sim.run(until=0)

    waiter_proc.interrupt(cause="give up")
    sim.schedule_call(1.0, lambda: sim.process(store.put("widget")))
    sim.run(until=10)  # must not raise

    assert log == [("interrupted", 0.0)]
    assert store.queue == ("widget",)  # item sitting unclaimed, not lost


def test_interrupted_queued_putter_is_removed_from_waiters():
    """A putter that blocks (store at capacity) and is then interrupted
    before a slot frees up must be removed from _put_waiters.
    """
    sim = Simulator()
    store = Store(sim, capacity=1)

    async def filler():
        await store.put("a")

    async def blocked_putter():
        try:
            await store.put("b")
        except Interrupt:
            pass

    sim.process(filler())
    putter_coro = blocked_putter()
    putter_proc = sim.process(putter_coro)
    sim.run(until=0)
    assert len(store._put_waiters) == 1

    putter_proc.interrupt(cause="give up")
    sim.run(until=1)

    assert putter_proc.triggered is True
    assert store._put_waiters == []
    # The interrupted putter's item was never accepted, so capacity is
    # still fully held by "a" alone.
    assert store.queue == ("a",)


def test_get_after_interrupted_putter_does_not_deliver_dropped_item():
    """After the queued putter above is interrupted and removed, a get()
    that frees the slot must not receive "b" (it was never accepted into
    the store) -- there is nothing left to deliver from that abandoned
    put.
    """
    sim = Simulator()
    store = Store(sim, capacity=1)
    log = []

    async def filler():
        await store.put("a")

    async def blocked_putter():
        try:
            await store.put("b")
        except Interrupt:
            log.append(("putter interrupted", sim.now))

    async def consumer():
        await sim.sleep(1)
        item = await store.get()
        log.append(("got", item, sim.now))

    sim.process(filler())
    putter_coro = blocked_putter()
    putter_proc = sim.process(putter_coro)
    sim.process(consumer())
    # Interrupt after the initial t=0 setup (filler/putter queueing) has
    # happened but without an intermediate run(until=0) call, which would
    # pop-and-discard consumer's t=1 sleep wakeup (a documented pre-existing
    # Simulator.run() quirk -- heapq.heappop() happens before the
    # `event_time > until` check -- unrelated to Store; see
    # tests/test_interrupt_resource_interaction.py for the same caveat).
    sim.schedule_call(0.0, lambda: putter_proc.interrupt(cause="give up"))
    sim.run(until=10)

    assert log == [("putter interrupted", 0.0), ("got", "a", 1.0)]
    assert store.queue == ()


def test_interrupt_racing_a_just_delivered_get_grant_does_not_leak_or_crash():
    """Subtler race: a getter has already been granted an item by
    _wake_a_getter (the item popped off _items, its Event fired with the
    item as value), but the resumption is a separately scheduled
    schedule_call() callback that hasn't run yet. If interrupt() lands
    first, Simulator's generation check drops the stale resumption, and
    Store.get's abandonment cleanup must notice the Event already
    resolved and put the item back at the front of _items rather than
    losing it.
    """
    sim = Simulator()
    store = Store(sim)
    log = []

    async def putter():
        await sim.sleep(1)
        await store.put("widget")
        log.append(("put", sim.now))

    async def waiter():
        try:
            item = await store.get()
            log.append(("got", item, sim.now))
        except Interrupt:
            log.append(("interrupted", sim.now))

    sim.process(putter())
    waiter_coro = waiter()
    waiter_proc = sim.process(waiter_coro)
    # Scheduled for the same simulated instant as the put (t=1); whichever
    # tiebreaker runs first, the interrupt must still be handled correctly
    # relative to the in-flight grant.
    sim.schedule_call(1.0, lambda: waiter_proc.interrupt(cause="race"))
    sim.run(until=10)  # must not raise

    assert log == [("put", 1.0), ("interrupted", 1.0)]
    assert waiter_proc.triggered is True
    # No leak: the orphaned grant's item was put back, not lost.
    assert store.queue == ("widget",)
    assert store._get_waiters == []


def test_get_after_recovered_dropped_grant_receives_item():
    """Follow-on to the race above: once the item has been put back after
    an abandoned grant, a subsequent get() must still be able to retrieve
    it -- proving it's genuinely recovered, not just present-but-stuck.
    """
    sim = Simulator()
    store = Store(sim)
    log = []

    async def putter():
        await sim.sleep(1)
        await store.put("widget")

    async def first_waiter():
        try:
            await store.get()
        except Interrupt:
            pass

    async def second_waiter():
        await sim.sleep(2)
        item = await store.get()
        log.append((item, sim.now))

    sim.process(putter())
    first_coro = first_waiter()
    first_proc = sim.process(first_coro)
    sim.process(second_waiter())
    sim.schedule_call(1.0, lambda: first_proc.interrupt(cause="race"))
    sim.run(until=10)

    assert log == [("widget", 2.0)]
