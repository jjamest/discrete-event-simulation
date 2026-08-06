from ordo.event import Event
from ordo.simulator import Simulator


def test_await_already_triggered_event_resumes_immediately():
    sim = Simulator()
    ev = Event()
    ev.succeed("hello")
    log = []

    async def waiter():
        result = await ev
        log.append(result)

    sim.process(waiter())
    sim.run(until=1)
    assert log == ["hello"]


def test_await_event_triggered_later_resumes_at_trigger_time():
    sim = Simulator()
    ev = Event()
    log = []

    async def waiter():
        result = await ev
        log.append((result, sim.now))

    def trigger_later():
        ev.succeed("go")

    sim.process(waiter())
    sim.schedule_call(5.0, trigger_later)
    sim.run(until=10)
    assert log == [("go", 5.0)]


def test_await_failed_event_raises_in_coroutine():
    sim = Simulator()
    ev = Event()
    log = []

    async def waiter():
        try:
            await ev
        except ValueError as e:
            log.append(str(e))

    sim.process(waiter())
    ev.fail(ValueError("nope"))
    sim.run(until=1)
    assert log == ["nope"]


def test_multiple_awaiters_all_resumed():
    sim = Simulator()
    ev = Event()
    log = []

    async def waiter(name):
        result = await ev
        log.append((name, result))

    sim.process(waiter("a"))
    sim.process(waiter("b"))
    sim.schedule_call(3.0, lambda: ev.succeed("done"))
    sim.run(until=10)
    assert sorted(log) == [("a", "done"), ("b", "done")]
