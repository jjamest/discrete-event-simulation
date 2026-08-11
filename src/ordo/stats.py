from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ordo.simulator import Simulator


class UsageStats:
    """Provides utilization tracking and wait-time samples.

    Shared bookkeeping for Resource and Store. Caller is responsible for
    calling _busy_changed(n) whenever the number of in-use units changes,
    and _queue_changed(n) whenever the waiter count changes. Both use
    time-weighted (integral-over-elapsed-time) averaging, computed lazily
    at access time using sim.now rather than incrementally, so a caller
    doesn't need to know in advance when the "observation window" ends and
    a still-busy/still-queued state at query time is correctly counted for
    its partial elapsed period.
    """

    def __init__(self, sim: "Simulator", capacity: float) -> None:
        self.sim = sim
        self.capacity = capacity
        self.wait_times: list = []

        self._busy_units = 0
        self._busy_area = 0.0  # integral of busy_units dt
        self._last_busy_change_t = sim.now

        self._queue_len = 0
        self._queue_area = 0.0
        self._last_queue_change_t = sim.now

    def _busy_changed(self, new_busy_units: int) -> None:
        now = self.sim.now
        self._busy_area += self._busy_units * (now - self._last_busy_change_t)
        self._busy_units = new_busy_units
        self._last_busy_change_t = now

    def _queue_changed(self, new_queue_len: int) -> None:
        now = self.sim.now
        self._queue_area += self._queue_len * (now - self._last_queue_change_t)
        self._queue_len = new_queue_len
        self._last_queue_change_t = now

    def _record_wait(self, wait: float) -> None:
        self.wait_times.append(wait)

    @property
    def utilization(self) -> float:
        now = self.sim.now
        area = self._busy_area + self._busy_units * (now - self._last_busy_change_t)
        elapsed = now
        if elapsed <= 0:
            return 0.0
        return area / (self.capacity * elapsed)

    @property
    def mean_queue_length(self) -> float:
        now = self.sim.now
        area = self._queue_area + self._queue_len * (now - self._last_queue_change_t)
        elapsed = now
        if elapsed <= 0:
            return 0.0
        return area / elapsed

    @property
    def mean_wait(self) -> float:
        if not self.wait_times:
            return 0.0
        return sum(self.wait_times) / len(self.wait_times)
