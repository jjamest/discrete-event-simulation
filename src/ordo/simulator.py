import heapq
import itertools
from typing import Coroutine, Any

from ordo.resource import Resource

class Simulator:
    def __init__(self):
        self.now: float = 0.0
        self.events = [] # heap
        self._counter = itertools.count()

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

    def process(self, coroutine: Coroutine) -> None:
        """Starts running a process"""
        self.schedule(delay=0.0, coroutine=coroutine)

    def resource(self, capacity: int = 1) -> "Resource":
        """Create a shared resource with limited capacity."""
        return Resource(self, capacity)

    def _resume(self, coro: Coroutine, sent_value: Any = None) -> None:
        """Advance a coroutine one step and act on its yielded instruction."""
        try:
            instruction, *payload = coro.send(sent_value)

            if instruction == "sleep":
                self.schedule(payload[0], coro)
            elif instruction == "acquire":
                resource, timeout = payload
                resource.request(coro, timeout)
        except StopIteration:
            # the coroutine finished
            pass

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
