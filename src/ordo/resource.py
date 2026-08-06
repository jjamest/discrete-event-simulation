import heapq
import itertools
from typing import Optional, TYPE_CHECKING

from ordo.event import Event

if TYPE_CHECKING:
    from ordo.simulator import Simulator


# temporary stub -- replaced in Task 9 with a real implementation in stats.py
class ResourceStats:
    def __init__(self, sim, resource):
        pass

    def _record_wait(self, wait):
        pass

    def _enter_queue(self):
        pass

    def _leave_queue(self):
        pass

    def _utilization_changed(self):
        pass


class Resource:
    """A shared resource with limited capacity (mutex/semaphore for processes)."""

    def __init__(self, sim: "Simulator", capacity: int = 1) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.sim = sim
        self.capacity = capacity
        self.in_use = 0
        self._waiters = []  # heap of [priority, seq, Event, settled]
        self._counter = itertools.count()
        self.stats = ResourceStats(sim, self)

    def acquire(self, timeout: Optional[float] = None, priority: int = 0):
        """Awaitable/async-context-manager that blocks the caller until a slot is free.

        Usable either as `got = await res.acquire()` or as
        `async with res.acquire() as got:` (which auto-releases on block
        exit, including via exception).

        If `timeout` is given and no slot frees up within that time, the
        caller reneges (gives up its place in line) and resolves to False
        instead of True.
        """
        return _AcquireAwaitable(self, timeout, priority)

    def _request(self, timeout: Optional[float], priority: int) -> Event:
        """Returns an Event that resolves to True (granted) or False (reneged)."""
        result = Event()
        if self.in_use < self.capacity:
            self.in_use += 1
            self.stats._record_wait(0.0)
            self.sim.schedule_call(0.0, lambda: result.succeed(True))
            return result

        seq = next(self._counter)
        entry = [priority, seq, result, False]  # last field: settled flag
        heapq.heappush(self._waiters, entry)
        self.stats._enter_queue()

        if timeout is not None:
            def on_timeout(entry=entry):
                if entry[3]:
                    return
                entry[3] = True
                self._remove_waiter(entry)
                self.stats._leave_queue()
                result.succeed(False)

            self.sim.schedule_call(timeout, on_timeout)

        return result

    def _remove_waiter(self, entry) -> None:
        """Mark a waiter entry settled and drop it from the heap, if still present.

        Used both by the timeout-renege path and by abandonment cleanup
        (see _AcquireAwaitable.__await__): if a queued waiter's coroutine
        is interrupted (or otherwise abandons the await via any exception)
        before being granted a slot, this removes it so a later release()
        can never resolve its Event a second time or hand a slot to a
        coroutine that isn't waiting for it anymore.
        """
        entry[3] = True
        try:
            self._waiters.remove(entry)
            heapq.heapify(self._waiters)
        except ValueError:
            pass

    def release(self) -> None:
        """Free a slot, handing it to the next waiting process, if any."""
        while self._waiters:
            entry = heapq.heappop(self._waiters)
            if entry[3]:
                continue  # already reneged/abandoned; skip and try the next waiter
            entry[3] = True
            self.stats._leave_queue()
            self.stats._record_wait(0.0)
            entry[2].succeed(True)
            return
        self.in_use -= 1
        self.stats._utilization_changed()

    @property
    def queue(self):
        return tuple(e[2] for e in sorted(self._waiters) if not e[3])


class _AcquireAwaitable:
    """Returned by Resource.acquire(); awaitable directly, or usable as `async with`."""

    def __init__(self, resource: Resource, timeout, priority) -> None:
        self.resource = resource
        self.timeout = timeout
        self.priority = priority
        self._got = None
        self._entry = None

    def __await__(self):
        result = self.resource._request(self.timeout, self.priority)
        # Find the waiter entry (if any) this request enqueued, so we can
        # clean it up if the wait is abandoned (e.g. the coroutine is
        # interrupted while still queued) rather than granted/reneged
        # normally. Immediate grants don't enqueue an entry at all.
        for entry in self.resource._waiters:
            if entry[2] is result:
                self._entry = entry
                break

        try:
            got = yield from result.__await__()
        except BaseException:
            # Abandoned mid-wait (Interrupt or any other exception thrown
            # into the coroutine while it was suspended here -- this is
            # general abandonment cleanup, not Interrupt-specific).
            #
            # Two distinct cases, both real:
            #
            # 1. Still queued (self._entry is not None and not yet
            #    settled): remove our waiter entry so a later release()
            #    never sees or resolves it. Without this the entry would
            #    linger as a phantom waiter that could later be granted a
            #    slot nobody is left to release -- a slot leak (or worse,
            #    a crash/misdelivery if the grant's resumption weren't
            #    also protected -- see Simulator's generation check on
            #    the "event" instruction).
            #
            # 2. Already granted (result.triggered and result.ok and
            #    result.value is True) but the coroutine never got to act
            #    on it: the grant's own resumption raced against the
            #    abandonment and lost (dropped as stale by Simulator's
            #    generation check), so the coroutine landed here via the
            #    *abandoning* exception instead of ever seeing `True`. The
            #    Resource has already incremented/kept `in_use` for this
            #    grant with no one left to call release() for it -- so we
            #    must release it back ourselves, exactly as if we'd
            #    acquired it and immediately given it up.
            if self._entry is not None and not self._entry[3]:
                self.resource._remove_waiter(self._entry)
            elif result.triggered and result.ok and result.value:
                self.resource.release()
            raise

        self._got = got
        return got

    async def __aenter__(self):
        return await self

    async def __aexit__(self, exc_type, exc, tb):
        if self._got:
            self.resource.release()
        return False
