from ordo.event import Event
from ordo.simulator import Simulator


def test_peek_returns_next_event_time():
    sim = Simulator()
    assert sim.peek() == float("inf")

    async def worker():
        await sim.sleep(5)

    sim.process(worker())
    assert sim.peek() == 0.0  # process itself scheduled at t=0
    sim.run(until=0.0)
    assert sim.peek() == 5.0


def test_len_reflects_pending_events():
    sim = Simulator()
    assert len(sim) == 0

    async def worker():
        await sim.sleep(1)

    sim.process(worker())
    assert len(sim) == 1

    sim.run()  # drain so `worker`'s coroutine is actually driven to completion


def test_run_until_event():
    sim = Simulator()
    log = []
    ev = Event()

    async def worker():
        await sim.sleep(3)
        log.append(("worker done", sim.now))
        ev.succeed()

    async def late():
        await sim.sleep(100)
        log.append(("late", sim.now))

    sim.process(worker())
    sim.process(late())
    sim.run(until=ev)
    assert log == [("worker done", 3.0)]  # stops before "late" fires at t=100


def test_run_multi_call_does_not_discard_future_events():
    """Regression test: run(until=N) must not discard an event scheduled
    after N when it stops early. A later run(until=M > N) call should
    still be able to process it.
    """
    sim = Simulator()
    log = []

    async def worker():
        await sim.sleep(7)
        log.append(("worker", sim.now))

    sim.process(worker())
    sim.run(until=5)  # worker's sleep(7) event (t=7) is not due yet
    assert log == []
    assert sim.peek() == 7.0  # must still be pending, not discarded
    assert len(sim) == 1

    sim.run(until=10)  # now it should fire
    assert log == [("worker", 7.0)]
