import heapq
import itertools
from typing import Optional, TYPE_CHECKING

from ordo.event import Event
from ordo.stats import UsageStats

if TYPE_CHECKING:
    from ordo.simulator import Simulator

# Waiter entries are [priority, seq, Event, settled, request_time] lists (a
# heap needs a plain sequence, not a dataclass) - named indices keep call
# sites readable. request_time is the sim.now at which the entry was
# enqueued, used to compute wait duration when it's eventually granted (or
# to compute nothing, if it's instead reneged/abandoned).
_PRIORITY, _SEQ, _EVENT, _SETTLED, _REQUEST_TIME = range(5)


class Resource:
    """A shared resource with limited capacity (mutex/semaphore for processes)."""

    def __init__(self, sim: "Simulator", capacity: int = 1) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.sim = sim
        self.capacity = capacity
        self.in_use = 0
        self._waiters = []  # heap of entries, see _PRIORITY/_SEQ/_EVENT/_SETTLED/_REQUEST_TIME
        self._counter = itertools.count()
        self.stats = UsageStats(sim, capacity)

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

    def _request(self, timeout: Optional[float], priority: int):
        """Enqueues (or immediately grants) a request.

        Returns `(event, entry)`: `event` resolves to True (granted) or
        False (reneged); `entry` is the waiter list entry to pass to
        `_remove_waiter` for abandonment cleanup, or None if the request
        was granted immediately (never enqueued).
        """
        result = Event()
        if self.in_use < self.capacity:
            # Granted immediately: no queueing occurred, so this is not a
            # "wait" sample -- wait_times only records time actually spent
            # queued (see release()'s grant-to-waiter branch).
            self.in_use += 1
            self.stats._busy_changed(self.in_use)
            self.sim.schedule_call(0.0, lambda: result.succeed(True))
            return result, None

        seq = next(self._counter)
        now = self.sim.now
        entry = [priority, seq, result, False, now]
        heapq.heappush(self._waiters, entry)
        self.stats._queue_changed(len(self._waiters))

        if timeout is not None:
            def on_timeout(entry=entry):
                if entry[_SETTLED]:
                    return
                self._remove_waiter(entry)
                result.succeed(False)

            self.sim.schedule_call(timeout, on_timeout)

        return result, entry

    def _remove_waiter(self, entry) -> None:
        """Mark a waiter entry settled and drop it from the heap, if still present.

        Used both by the timeout-renege path and by abandonment cleanup
        (see _AcquireAwaitable.__await__): if a queued waiter's coroutine
        is interrupted (or otherwise abandons the await via any exception)
        before being granted a slot, this removes it so a later release()
        can never resolve its Event a second time or hand a slot to a
        coroutine that isn't waiting for it anymore.

        This is the single place a waiter leaves the queue without being
        granted a slot (timeout-renege and abandonment both funnel through
        here), so it's also the single place that updates queue-length
        stats for that transition -- callers don't need to remember to do
        it themselves.
        """
        already_settled = entry[_SETTLED]
        entry[_SETTLED] = True
        try:
            self._waiters.remove(entry)
            heapq.heapify(self._waiters)
            if not already_settled:
                self.stats._queue_changed(len(self._waiters))
        except ValueError:
            pass

    def release(self) -> None:
        """Free a slot, handing it to the next waiting process, if any."""
        while self._waiters:
            entry = heapq.heappop(self._waiters)
            if entry[_SETTLED]:
                continue  # already reneged/abandoned; skip and try the next waiter
            entry[_SETTLED] = True
            self.stats._queue_changed(len(self._waiters))
            self.stats._record_wait(self.sim.now - entry[_REQUEST_TIME])
            entry[_EVENT].succeed(True)
            return
        self.in_use -= 1
        self.stats._busy_changed(self.in_use)

    @property
    def queue(self):
        return tuple(e[_EVENT] for e in sorted(self._waiters) if not e[_SETTLED])


class _AcquireAwaitable:
    """Returned by Resource.acquire(); awaitable directly, or usable as `async with`."""

    def __init__(self, resource: Resource, timeout, priority) -> None:
        self.resource = resource
        self.timeout = timeout
        self.priority = priority
        self._got = None
        self._entry = None

    def __await__(self):
        result, self._entry = self.resource._request(self.timeout, self.priority)
        # self._entry is the waiter list entry this request enqueued (None
        # if it was granted immediately and never queued), kept so we can
        # clean it up if the wait is abandoned (e.g. the coroutine is
        # interrupted while still queued) rather than granted/reneged
        # normally.
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
            if self._entry is not None and not self._entry[_SETTLED]:
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
