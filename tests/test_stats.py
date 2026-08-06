from ordo.exceptions import Interrupt
from ordo.simulator import Simulator
from ordo.resource import Resource
from ordo.store import Store


def test_resource_utilization_time_weighted():
    sim = Simulator()
    res = Resource(sim, capacity=1)

    async def user():
        await res.acquire()
        await sim.sleep(4)  # busy for 4 of the next 10 seconds
        res.release()

    sim.process(user())
    # Simulator.run(until=N) only advances sim.now to the last processed
    # event's time, not all the way to N, if nothing is scheduled past
    # that point -- schedule a harmless no-op at t=10 so the clock (and
    # therefore the utilization denominator) actually reaches 10.
    sim.schedule_call(10.0, lambda: None)
    sim.run(until=10)
    assert res.stats.utilization == 0.4  # busy 4s out of 10s elapsed


def test_resource_mean_wait_time():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def holder():
        await res.acquire()
        await sim.sleep(5)
        res.release()

    async def waiter():
        await res.acquire()
        log.append(sim.now)
        res.release()

    sim.process(holder())
    sim.process(waiter())
    sim.run(until=10)
    assert res.stats.wait_times == [5.0]
    assert res.stats.mean_wait == 5.0


def test_store_mean_queue_length_zero_when_never_blocked():
    sim = Simulator()
    store = Store(sim)

    async def producer():
        await store.put("x")

    sim.process(producer())
    sim.run(until=10)
    assert store.stats.mean_queue_length == 0.0


def test_resource_queue_stats_recover_after_interrupted_waiter():
    """A queued waiter that's interrupted and abandons its acquire() must
    not leave stats.mean_queue_length permanently elevated -- the
    time-weighted queue length should drop back to 0 for the remainder of
    the run once the phantom waiter is cleaned up.
    """
    sim = Simulator()
    res = Resource(sim, capacity=1)

    async def holder():
        await res.acquire()
        await sim.sleep(100)
        res.release()

    async def waiter():
        try:
            await res.acquire()
        except Interrupt:
            pass

    sim.process(holder())
    waiter_proc = sim.process(waiter())
    sim.run(until=0)
    assert len(res._waiters) == 1

    # Queued for 2 seconds (t=0 to t=2) before being interrupted.
    sim.schedule_call(2.0, lambda: waiter_proc.interrupt(cause="give up"))
    # See test_resource_utilization_time_weighted's comment: run(until=N)
    # only advances sim.now to the last processed event, so schedule a
    # no-op at t=12 to make sure the clock actually gets there (the
    # holder's own release happens at t=100, well past our window).
    sim.schedule_call(12.0, lambda: None)
    sim.run(until=12)  # 2s queued + 10s empty queue afterwards

    # Time-weighted mean over [0, 12]: queue length 1 for 2s, 0 for 10s.
    assert res.stats.mean_queue_length == (1 * 2.0 + 0 * 10.0) / 12.0
    assert res._waiters == []  # cleaned up, not a phantom


def test_resource_stats_consistent_after_orphaned_grant_released():
    """If a queued waiter is granted a slot but is interrupted before it
    can act on the grant (the classic dropped-resumption race), the
    abandonment cleanup releases the slot back via resource.release().
    Stats (busy units / utilization) must reflect the slot as freed, not
    stuck "in use" forever.
    """
    sim = Simulator()
    res = Resource(sim, capacity=1)
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
    waiter_proc = sim.process(waiter())
    sim.schedule_call(1.0, lambda: waiter_proc.interrupt(cause="race"))
    sim.schedule_call(10.0, lambda: None)  # ensure sim.now reaches 10
    sim.run(until=10)

    assert log == [("holder released", 1.0), ("interrupted", 1.0)]
    assert res.in_use == 0
    # Busy for [0, 1) by the holder, then free -- the orphaned grant to the
    # waiter must not count as ongoing busy time after t=1.
    assert res.stats.utilization == 1.0 / 10.0


def test_store_queue_stats_recover_after_interrupted_getter():
    """A getter blocked on an empty store, then interrupted before any
    item arrives, must not leave the get-side queue length stuck
    elevated.
    """
    sim = Simulator()
    store = Store(sim)

    async def waiter():
        try:
            await store.get()
        except Interrupt:
            pass

    waiter_proc = sim.process(waiter())
    sim.run(until=0)
    assert len(store._get_waiters) == 1

    sim.schedule_call(3.0, lambda: waiter_proc.interrupt(cause="give up"))
    sim.schedule_call(13.0, lambda: None)  # ensure sim.now reaches 13
    sim.run(until=13)  # 3s queued + 10s empty queue afterwards

    assert store.stats.mean_queue_length == (1 * 3.0 + 0 * 10.0) / 13.0
    assert store._get_waiters == []


def test_store_busy_stats_consistent_after_recovered_dropped_grant():
    """A getter granted an item, then interrupted before it can receive
    delivery (the item was already popped from _items), has the item
    recovered back into the store. Busy units (len(_items)) must reflect
    the item as present the whole time from an external observer's
    perspective, not double-counted or lost.
    """
    sim = Simulator()
    # Bounded capacity so utilization (which divides by capacity) is
    # meaningful -- an unbounded Store's utilization is always 0 (dividing
    # by infinite capacity), so this scenario needs a finite one.
    store = Store(sim, capacity=1)
    log = []

    async def putter():
        await sim.sleep(1)
        await store.put("widget")

    async def waiter():
        try:
            item = await store.get()
            log.append(("got", item, sim.now))
        except Interrupt:
            log.append(("interrupted", sim.now))

    sim.process(putter())
    waiter_proc = sim.process(waiter())
    sim.schedule_call(1.0, lambda: waiter_proc.interrupt(cause="race"))
    sim.schedule_call(10.0, lambda: None)  # ensure sim.now reaches 10
    sim.run(until=10)

    assert log == [("interrupted", 1.0)]
    assert store.queue == ("widget",)
    # The item sat in the store the whole time from t=1 onward (recovered,
    # not lost), so busy units should be 1 for the [1, 10] window.
    assert store.stats.utilization == 9.0 / 10.0
