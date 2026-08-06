from typing import Coroutine, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ordo.simulator import Simulator


class _Waiter:
    """Tracks a pending acquire request so a grant and a renege can race safely."""

    __slots__ = ("coro", "settled")

    def __init__(self, coro: Coroutine) -> None:
        self.coro = coro
        self.settled = False # True once granted or reneged, to make the other a no-op


class Resource:
    """A shared resource with limited capacity (mutex/semaphore for processes)."""

    def __init__(self, sim: "Simulator", capacity: int = 1) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.sim = sim
        self.capacity = capacity
        self.in_use = 0
        self._waiters: list[_Waiter] = [] # waiters for a slot, FIFO

    def acquire(self, timeout: Optional[float] = None):
        """Awaitable that blocks the caller until a slot is free.

        If `timeout` is given and no slot frees up within that time, the
        caller reneges (gives up its place in line) and the await resolves
        to False instead of True.
        """
        resource = self

        class VirtualAcquire:
            def __await__(self):
                result = yield ("acquire", resource, timeout)
                return result

        return VirtualAcquire()

    def request(self, coro: Coroutine, timeout: Optional[float] = None) -> None:
        """Called by the simulator when a process wants to acquire this resource."""
        if self.in_use < self.capacity:
            self.in_use += 1
            self.sim.schedule(delay=0.0, coroutine=coro, value=True)
        else:
            waiter = _Waiter(coro)
            self._waiters.append(waiter)
            if timeout is not None:
                self.sim.schedule_call(timeout, lambda: self._renege(waiter))

    def _renege(self, waiter: "_Waiter") -> None:
        """Called when a waiter's patience runs out before a slot freed up."""
        if waiter.settled:
            return # already granted a slot; ignore the stale timeout
        waiter.settled = True
        self._waiters.remove(waiter)
        self.sim.schedule(delay=0.0, coroutine=waiter.coro, value=False)

    def release(self) -> None:
        """Free a slot, handing it to the next waiting process, if any."""
        while self._waiters:
            waiter = self._waiters.pop(0)
            if waiter.settled:
                continue # already reneged; skip and try the next waiter
            waiter.settled = True
            self.sim.schedule(delay=0.0, coroutine=waiter.coro, value=True)
            return
        self.in_use -= 1
