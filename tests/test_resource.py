import pytest

from ordo.resource import Resource
from ordo.simulator import Simulator


def test_acquire_release_no_timeout():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def user(name, hold):
        got = await res.acquire()
        log.append((name, got, sim.now))
        await sim.sleep(hold)
        res.release()

    sim.process(user("a", 5))
    sim.process(user("b", 5))
    sim.run(until=20)

    assert log == [("a", True, 0.0), ("b", True, 5.0)]


def test_waiter_reneges_after_timeout():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def holder():
        await res.acquire()
        await sim.sleep(10)
        res.release()

    async def impatient():
        got = await res.acquire(timeout=3)
        log.append(("impatient", got, sim.now))

    sim.process(holder())
    sim.schedule(0.5, impatient())
    sim.run(until=5)  # stop before the holder releases at t=10

    assert log == [("impatient", False, 3.5)]
    assert res.in_use == 1  # holder still holds it; reneged waiter never took a slot


def test_reneged_waiter_does_not_block_others():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def holder():
        await res.acquire()
        await sim.sleep(10)
        res.release()

    async def impatient():
        got = await res.acquire(timeout=3)
        log.append(("impatient", got, sim.now))

    async def patient():
        got = await res.acquire(timeout=20)
        log.append(("patient", got, sim.now))
        if got:
            res.release()

    sim.process(holder())
    sim.schedule(0.5, impatient())
    sim.schedule(0.5, patient())
    sim.run(until=50)

    assert log == [("impatient", False, 3.5), ("patient", True, 10.0)]


def test_grant_before_timeout_wins():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def holder():
        await res.acquire()
        await sim.sleep(2)
        res.release()

    async def waits_but_gets_served():
        got = await res.acquire(timeout=100)
        log.append(("waiter", got, sim.now))
        if got:
            res.release()

    sim.process(holder())
    sim.schedule(0.0, waits_but_gets_served())
    sim.run(until=50)

    assert log == [("waiter", True, 2.0)]
    # ensure the stale renege callback firing later is a harmless no-op
    sim.run(until=200)
    assert res.in_use == 0


def test_async_with_auto_releases_on_normal_exit():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def user(name):
        async with res.acquire() as got:
            log.append((name, got, sim.now))
            await sim.sleep(5)
        # released here automatically

    sim.process(user("a"))
    sim.process(user("b"))
    sim.run(until=20)
    assert log == [("a", True, 0.0), ("b", True, 5.0)]
    assert res.in_use == 0


def test_async_with_releases_on_exception():
    sim = Simulator()
    res = Resource(sim, capacity=1)

    async def bad_user():
        async with res.acquire():
            raise ValueError("oops")

    proc = sim.process(bad_user())
    with pytest.raises(Exception):
        sim.run(until=10)
    assert res.in_use == 0


def test_async_with_does_not_release_on_reneged_timeout():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def holder():
        async with res.acquire():
            await sim.sleep(10)

    async def impatient():
        async with res.acquire(timeout=3) as got:
            log.append(got)

    sim.process(holder())
    sim.process(impatient())
    sim.run(until=5)
    assert log == [False]
    assert res.in_use == 1  # holder still holds; impatient's False acquire released nothing


def test_priority_ordering_higher_priority_served_first():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def holder():
        got = await res.acquire()
        log.append(("holder", got, sim.now))
        await sim.sleep(5)
        res.release()

    async def low(name, prio, delay):
        await sim.sleep(delay)
        got = await res.acquire(priority=prio)
        log.append((name, got, sim.now))
        res.release()

    sim.process(holder())
    sim.schedule(0.0, low("low_prio", 10, 0.0))
    sim.schedule(0.0, low("high_prio", 1, 0.0))
    sim.run(until=20)

    assert log[0] == ("holder", True, 0.0)
    # both arrived before t=5 when the slot frees; high_prio (lower number) should win
    assert log[1][0] == "high_prio"
    assert log[2][0] == "low_prio"
