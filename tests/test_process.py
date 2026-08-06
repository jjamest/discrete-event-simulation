import pytest

from ordo.exceptions import Interrupt, SimulationError
from ordo.simulator import Simulator


def test_process_returns_process_object_and_fires_on_completion():
    sim = Simulator()
    log = []

    async def worker():
        await sim.sleep(3)
        return "result"

    async def watcher(proc):
        result = await proc
        log.append((result, sim.now))

    proc = sim.process(worker())
    sim.process(watcher(proc))
    sim.run(until=10)
    assert log == [("result", 3.0)]
    assert proc.triggered is True
    assert proc.value == "result"


def test_awaiter_sees_exception_from_child_process():
    sim = Simulator()
    log = []

    async def failing_worker():
        await sim.sleep(1)
        raise ValueError("child failed")

    async def watcher(proc):
        try:
            await proc
        except ValueError as e:
            log.append(str(e))

    proc = sim.process(failing_worker())
    sim.process(watcher(proc))
    sim.run(until=10)
    assert log == ["child failed"]


def test_unhandled_exception_in_top_level_process_raises_simulation_error():
    sim = Simulator()

    async def failing_worker():
        await sim.sleep(1)
        raise ValueError("boom")

    sim.process(failing_worker())

    with pytest.raises(SimulationError) as exc_info:
        sim.run(until=10)
    assert exc_info.value.sim_time == 1.0
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_fire_and_forget_process_with_no_watcher_runs_fine():
    """Existing examples rely on sim.process(...) without ever awaiting the result."""
    sim = Simulator()
    log = []

    async def worker():
        await sim.sleep(2)
        log.append(sim.now)

    sim.process(worker())
    sim.run(until=10)
    assert log == [2.0]


def test_interrupt_raises_inside_sleeping_process():
    sim = Simulator()
    log = []

    async def worker():
        try:
            await sim.sleep(100)
        except Interrupt as e:
            log.append(("interrupted", e.cause, sim.now))

    proc = sim.process(worker())
    sim.schedule_call(5.0, lambda: proc.interrupt(cause="stop"))
    sim.run(until=10)
    assert log == [("interrupted", "stop", 5.0)]


def test_uncaught_interrupt_ends_process_like_any_exception():
    sim = Simulator()

    async def worker():
        await sim.sleep(100)

    proc = sim.process(worker())
    sim.schedule_call(5.0, lambda: proc.interrupt())
    sim.run(until=10)
    assert proc.triggered is True
    assert proc.ok is False
    assert isinstance(proc.exception, Interrupt)
