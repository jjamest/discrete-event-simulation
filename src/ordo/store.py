import heapq
import itertools
from collections import deque
from typing import Optional, TYPE_CHECKING

from ordo.event import Event
from ordo.stats import UsageStats

if TYPE_CHECKING:
    from ordo.simulator import Simulator

TIMEOUT = object()  # sentinel distinct from a legitimate None item

# Waiter entries are [priority, seq, Event, settled, item, request_time]
# lists (a heap needs a plain sequence, not a dataclass) - named indices
# keep call sites readable. Put-waiter entries carry the pending item from
# the start (it's the payload of the request itself). Get-waiter entries
# start with item slot None and only get one written in (by
# _wake_a_getter, at the same moment `settled` flips to True) once a grant
# is decided - this lets abandonment cleanup recover a granted-but-
# undelivered item without racing result.succeed()'s own scheduled
# callback (see get()'s except block for why entry[_SETTLED] can be True
# before result.triggered is). request_time is the sim.now the entry was
# enqueued, used for get-side wait-time stats (see Store.stats).
_PRIORITY, _SEQ, _EVENT, _SETTLED, _ITEM, _REQUEST_TIME = range(6)


class Store:
    """Bounded FIFO item queue with blocking put/get.

    Unlike Resource, Store does not support `async with` - put()/get() are
    plain async methods, not a dedicated awaitable/context-manager (there
    is no "release" analogue to pair a block with).
    """

    def __init__(self, sim: "Simulator", capacity: float = float("inf")) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self.sim = sim
        self.capacity = capacity
        self._items = deque()
        self._get_waiters = []  # heap of [priority, seq, Event, settled, item, request_time]
        self._put_waiters = []  # heap of [priority, seq, Event, settled, item, request_time]
        self._counter = itertools.count()
        # busy_units tracks len(self._items) (occupied capacity);
        # queue_length tracks get-side waiters only. Simplification: a
        # Store has two waiter populations (_get_waiters, _put_waiters),
        # but get-side blocking (consumers waiting for items) is the more
        # common bottleneck to monitor, so a single UsageStats tracks that
        # one queue. Put-side queueing (producers blocked on a full store)
        # is not separately reported.
        self.stats = UsageStats(sim, capacity)

    async def put(self, item, timeout: Optional[float] = None, priority: int = 0) -> bool:
        if len(self._items) < self.capacity:
            self._items.append(item)
            self.stats._busy_changed(len(self._items))
            self._wake_a_getter()
            return True

        result = Event()
        seq = next(self._counter)
        entry = [priority, seq, result, False, item, self.sim.now]
        heapq.heappush(self._put_waiters, entry)

        if timeout is not None:
            def on_timeout(entry=entry):
                if entry[_SETTLED]:
                    return
                entry[_SETTLED] = True
                self._remove(self._put_waiters, entry)
                result.succeed(False)

            self.sim.schedule_call(timeout, on_timeout)

        try:
            return await result
        except BaseException:
            # Abandoned mid-wait (Interrupt or any other exception thrown
            # into the coroutine while suspended here). Mirrors
            # Resource._AcquireAwaitable's abandonment cleanup (see
            # resource.py).
            #
            # Two cases:
            #
            # 1. Still queued (entry not yet settled): remove it so a
            #    later _wake_a_putter() never resolves it or appends its
            #    item on its behalf. Without this the entry would linger
            #    as a phantom waiter.
            #
            # 2. Already accepted (entry[_SETTLED] is True, meaning
            #    _wake_a_putter already popped this entry and appended
            #    `item` to self._items) but the coroutine's own
            #    confirmation resumption raced an interrupt and was
            #    dropped as stale by Simulator's generation check. Unlike
            #    Resource (where an abandoned grant must be released back
            #    because in_use has no other owner), the ITEM here is
            #    already sitting in self._items for anyone to get() - it
            #    isn't tied to this coroutine continuing, only the
            #    True-return confirmation was lost. So no cleanup is
            #    needed on this path: the item simply stays queued.
            #
            # Note: entry[_SETTLED] is set synchronously by
            # _wake_a_putter as soon as the grant decision is made, but
            # result.succeed(True) is only called later via a scheduled
            # callback - so "settled but result not yet triggered" is a
            # real, reachable state (the abandonment race landing in that
            # gap), not just the "still queued" case. Either way here on
            # the put side there's nothing to undo: if settled, the item
            # is already correctly in self._items; if not, removing the
            # waiter entry is enough.
            if not entry[_SETTLED]:
                entry[_SETTLED] = True
                self._remove(self._put_waiters, entry)
            raise

    async def get(self, timeout: Optional[float] = None, priority: int = 0):
        if self._items:
            item = self._items.popleft()
            self.stats._busy_changed(len(self._items))
            self._wake_a_putter()
            return item

        result = Event()
        seq = next(self._counter)
        entry = [priority, seq, result, False, None, self.sim.now]
        heapq.heappush(self._get_waiters, entry)
        self.stats._queue_changed(len(self._get_waiters))

        if timeout is not None:
            def on_timeout(entry=entry):
                if entry[_SETTLED]:
                    return
                self._remove_get_waiter(entry)
                result.succeed(TIMEOUT)

            self.sim.schedule_call(timeout, on_timeout)

        try:
            return await result
        except BaseException:
            # Abandoned mid-wait. Mirrors Resource._AcquireAwaitable.
            #
            # 1. Still queued (entry not yet settled): remove it so a
            #    later _wake_a_getter() never resolves it or hands it an
            #    item nobody will collect.
            #
            # 2. Already granted an item (entry[_SETTLED] is True, and
            #    entry[_ITEM] holds what _wake_a_getter popped off
            #    self._items for it) but the coroutine's own delivery
            #    resumption raced an interrupt and was dropped as stale
            #    (Simulator's generation check on the "event"
            #    instruction). The item WAS removed from self._items with
            #    nobody left to receive it - a real analogue of Resource's
            #    in_use-leak. We must put it back at the FRONT of _items
            #    (not the back) so it doesn't lose its FIFO position
            #    relative to items queued behind it (those items arrived
            #    after this one; this one must still be served first to
            #    preserve FIFO order for everyone else).
            #
            # We deliberately read entry[_ITEM] here rather than
            # result.value: _wake_a_getter sets entry[_SETTLED] = True
            # (and, as of this fix, entry[_ITEM]) synchronously the
            # instant it decides to grant, but result.succeed(item) - the
            # thing that would set result.value - only runs later via a
            # separately scheduled callback. "Settled but result not yet
            # triggered" is a real, reachable state (this abandonment
            # landing in exactly that gap), and reading result.value there
            # would see None instead of the real item. entry[_ITEM] has no
            # such gap: it's written at the same synchronous instant as
            # entry[_SETTLED].
            if not entry[_SETTLED]:
                self._remove_get_waiter(entry)
            else:
                # The item was already popped out of self._items by
                # _wake_a_getter (busy count already dropped) with nobody
                # left to receive it. Putting it back at the front restores
                # self._items to what an external observer would consider
                # unchanged overall -- the item's presence in the store
                # never really left, it was just earmarked for a getter who
                # then abandoned -- so appendleft's busy_changed and the
                # immediately-following _wake_a_getter() call (which may
                # hand the item straight to a different waiter, popping it
                # right back out) net out correctly between themselves:
                # either the second call finds no other waiter and busy
                # ends up back where it started, or it does and the item's
                # departure is recorded again for the new recipient.
                self._items.appendleft(entry[_ITEM])
                self.stats._busy_changed(len(self._items))
                self._wake_a_getter()
            raise

    def _remove_get_waiter(self, entry) -> None:
        """Mark a get-waiter entry settled and drop it from the heap.

        Single chokepoint for a get-waiter leaving the queue without being
        granted an item (timeout-renege and abandonment cleanup both use
        this), so it's also the single place that updates get-side queue
        stats for that transition.
        """
        already_settled = entry[_SETTLED]
        entry[_SETTLED] = True
        self._remove(self._get_waiters, entry)
        if not already_settled:
            self.stats._queue_changed(len(self._get_waiters))

    def _wake_a_getter(self) -> None:
        while self._get_waiters:
            entry = heapq.heappop(self._get_waiters)
            if entry[_SETTLED]:
                continue
            entry[_SETTLED] = True
            self.stats._queue_changed(len(self._get_waiters))
            self.stats._record_wait(self.sim.now - entry[_REQUEST_TIME])
            item = self._items.popleft()
            self.stats._busy_changed(len(self._items))
            entry[_ITEM] = item
            self.sim.schedule_call(0.0, lambda e=entry, i=item: e[_EVENT].succeed(i))
            self._wake_a_putter()
            return

    def _wake_a_putter(self) -> None:
        while self._put_waiters:
            entry = heapq.heappop(self._put_waiters)
            if entry[_SETTLED]:
                continue
            entry[_SETTLED] = True
            self._items.append(entry[_ITEM])
            self.stats._busy_changed(len(self._items))
            self.sim.schedule_call(0.0, lambda e=entry: e[_EVENT].succeed(True))
            self._wake_a_getter()
            return

    @staticmethod
    def _remove(heap, entry) -> None:
        try:
            heap.remove(entry)
            heapq.heapify(heap)
        except ValueError:
            pass

    @property
    def queue(self):
        return tuple(self._items)
