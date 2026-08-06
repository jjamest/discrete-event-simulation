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
