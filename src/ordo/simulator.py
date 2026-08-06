import heapq
import itertools
from typing import Coroutine, Any, Optional

from ordo.event import Event
from ordo.exceptions import Interrupt, SimulationError
from ordo.process import Process
from ordo.resource import Resource

class Simulator:
    def __init__(self):
        self.now: float = 0.0
        self.events = [] # heap
        self._counter = itertools.count()
        self._process_by_coro: dict = {}

    def schedule(self, delay: float, coroutine: Coroutine, value: Any = None) -> None:
        """Schedule a coroutine to resume after some delay, sending it `value`."""
        event_time = self.now + delay # when the event should be executed
        tiebreaker = next(self._counter)
        heapq.heappush(self.events, (event_time, tiebreaker, coroutine, value)) # (time, priority tiebreaker, coroutine, value)

    def schedule_call(self, delay: float, func) -> None:
        """Schedule a zero-arg callback to run after some delay (not a coroutine)."""
        event_time = self.now + delay
        tiebreaker = next(self._counter)
        heapq.heappush(self.events, (event_time, tiebreaker, func, None))

    def sleep(self, delay: float):
        """Awaitable sleep that pauses the caller coroutine and schedules resumption"""
        class VirtualSleep:
            def __init__(self, sim, delay) -> None:
                self.sim = sim
                self.delay = delay

            def __await__(self):
                yield ("sleep", self.delay)

        return VirtualSleep(self, delay)

    def process(self, coroutine: Coroutine) -> "Process":
        """Starts running a process; returns a Process event that fires on completion."""
        proc = Process(coroutine, sim=self)
        self._process_by_coro[coroutine] = proc
        self.schedule(delay=0.0, coroutine=coroutine)
        return proc

    def resource(self, capacity: int = 1) -> "Resource":
        """Create a shared resource with limited capacity."""
        return Resource(self, capacity)

    def any_of(self, events: list[Event]) -> Event:
        """Returns an Event that fires with the first event in `events` to fire."""
        result = Event()

        def on_any(ev: Event) -> None:
            if not result.triggered:
                result.succeed(ev)

        for ev in events:
            ev.add_callback(on_any)
        return result

    def all_of(self, events: list[Event]) -> Event:
        """Returns an Event that fires once every event in `events` has fired."""
        result = Event()
        events = list(events)
        remaining = len(events)

        if remaining == 0:
            result.succeed([])
            return result

        def on_one(_ev: Event) -> None:
            nonlocal remaining
            remaining -= 1
            if remaining == 0:
                result.succeed(events)

        for ev in events:
            ev.add_callback(on_one)
        return result

    def _resume(
        self,
        coro: Coroutine,
        sent_value: Any = None,
        sent_exception: Optional[BaseException] = None,
    ) -> None:
        """Advance a coroutine one step and act on its yielded instruction.

        Resumes via `coro.throw(sent_exception)` if an exception is given,
        otherwise via `coro.send(sent_value)`.
        """
        try:
            if sent_exception is not None:
                instruction, *payload = coro.throw(sent_exception)
            else:
                instruction, *payload = coro.send(sent_value)

            if instruction == "sleep":
                self.schedule(payload[0], coro)
            elif instruction == "acquire":
                resource, timeout = payload
                resource.request(coro, timeout)
            elif instruction == "event":
                event: Event = payload[0]

                def on_fire(ev: Event, coro=coro) -> None:
                    self.schedule_call(0.0, lambda: self._resume_from_event(coro, ev))

                event.add_callback(on_fire)
        except StopIteration as stop:
            # the coroutine finished; resolve its Process, if any
            proc = self._process_by_coro.pop(coro, None)
            if proc is not None and not proc.triggered:
                proc.succeed(stop.value)
        except Exception as exc:
            # the coroutine raised; fail its Process, escalating to a
            # SimulationError if nobody was awaiting it
            proc = self._process_by_coro.pop(coro, None)
            if proc is not None and not proc.triggered:
                proc.fail(exc)
                if not proc.exception_handled and not isinstance(exc, Interrupt):
                    raise SimulationError(
                        f"unhandled exception in process: {exc}",
                        sim_time=self.now,
                        process=proc,
                    ) from exc
            else:
                raise

    def _resume_from_event(self, coro: Coroutine, event: Event) -> None:
        """Resume a coroutine that was awaiting `event`, now that it has fired."""
        if event.ok:
            self._resume(coro, sent_value=event.value)
        else:
            self._resume(coro, sent_exception=event.exception)

    def run(self, until: float = float("inf")) -> None:
        """Main event loop"""
        while self.events:
            event_time, _, target, value = heapq.heappop(self.events)

            if event_time > until: # or >=
                break

            self.now = event_time # push clock forward

            if isinstance(target, Coroutine):
                self._resume(target, value)
            else:
                target() # plain zero-arg callback
