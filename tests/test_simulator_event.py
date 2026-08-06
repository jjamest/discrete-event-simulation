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


def test_event_resumption_goes_behind_same_tick_work_already_queued():
    """Resuming an event-awaiter must be re-scheduled via schedule_call,
    not run inline from within Event._fire_callbacks.

    If it ran inline, the resumed coroutine's effect would jump ahead of
    other same-tick work that was already sitting in the heap (queued
    before the succeed() call fires the callback). Because the real
    implementation enqueues the resumption fresh with a brand new
    tiebreaker, it must land behind anything already queued for tick 0.
    """
    sim = Simulator()
    ev = Event()
    log = []

    async def waiter():
        await ev
        log.append("waiter-resumed")

    def trigger():
        ev.succeed("go")
        log.append("trigger")

    sim.process(waiter())  # tiebreaker 0: suspends, registers on_fire callback
    sim.schedule_call(0.0, trigger)  # tiebreaker 1: fires the event mid-tick
    sim.schedule_call(0.0, lambda: log.append("other"))  # tiebreaker 2: already queued

    sim.run(until=1)

    assert log == ["trigger", "other", "waiter-resumed"]
