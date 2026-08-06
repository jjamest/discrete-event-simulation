# DES Core Overhaul — Design

## Goal

Round out `ordo`'s core simulation engine from a minimal coroutine scheduler
with a single `Resource` primitive into a small but complete SimPy-style DES
toolkit: a generic `Event` primitive, process interruption, exception
propagation, context-manager resource release, priority queuing, a `Store`
primitive, built-in statistics, seeded RNG, and basic introspection.

## Scope

### 1. `Event` primitive (new `src/ordo/event.py`)

- `Event()` — starts untriggered.
- `.succeed(value=None)` / `.fail(exception)` — trigger exactly once;
  triggering an already-triggered event raises `RuntimeError`.
- `.triggered` / `.ok` properties.
- Awaitable: `await event` suspends the coroutine until the event fires,
  resuming with the value or raising the exception.
- Multiple processes may await the same `Event`; all are resumed when it
  fires.
- `sim.any_of([e1, e2, ...])` / `sim.all_of([...])` — combinators returning
  a new `Event` that fires when the first / all given events fire.

`sim.sleep(delay)` becomes sugar for an internally scheduled timer `Event`.
`Resource.acquire` is rebuilt on top of `Event` (fulfilled by `release()`).

### 2. `Process` wrapper + interruption

- `sim.process(coroutine)` returns a `Process` (an `Event` subclass that
  fires when the coroutine finishes, carrying its return value, or fails
  with its exception).
- `Process.interrupt(cause=None)` injects an `Interrupt` exception into the
  target coroutine at its next suspension point. The process may catch
  `Interrupt` to handle breakdowns/reneging, or let it propagate to end the
  process. Interrupting a process waiting on a `Resource`/`Event` removes it
  from that thing's waiter list first.

### 3. Exception propagation

- An unhandled exception in a process coroutine causes its `Process` event
  to `.fail(exc)`.
- If nothing awaits that `Process` (a fire-and-forget top-level process),
  the exception propagates out of `sim.run()` wrapped in a
  `SimulationError` (records `sim.now` and the failed process), chained
  from the original via `raise ... from original`.
- If another process is awaiting it (`await child_process`), the exception
  raises at the awaiter's suspension point — normal async propagation, so a
  parent can `try/except` around a spawned child.

### 4. `Resource`: `async with` + priority

- `resource.acquire()` return value works as before and additionally
  supports `async with resource.acquire() as ok:` — auto-releases on block
  exit (normal or exceptional), only if the acquire actually succeeded (not
  after a timeout/renege `False`).
- `resource.acquire(timeout=None, priority=0)` — waiters ordered by
  `(priority, insertion_order)`; lower number = higher priority, ties FIFO.
  No preemption of an in-progress holder.
- Manual `resource.release()` remains available.

### 5. `Store` (new `src/ordo/store.py`)

- `Store(sim, capacity=inf)` — bounded FIFO item queue built on the same
  `Event`/waiter machinery as `Resource` (inherits priority ordering and
  interrupt-safety).
- `await store.put(item, timeout=None)` — blocks if at capacity; returns
  `True`/`False`.
- `await store.get(timeout=None)` — blocks if empty; returns the item, or a
  `TIMEOUT` sentinel (distinct from a legitimate `None` item) on timeout.

### 6. Statistics (built into `Resource` and `Store`, always on)

- `.stats.utilization` — time-weighted fraction of capacity in use.
- `.stats.mean_queue_length` — time-weighted average waiter count.
- `.stats.wait_times` / `.stats.mean_wait` — realized queue-time samples
  for completed acquires/gets.
- Time-weighted metrics integrate incrementally on each state change.

### 7. RNG seeding

- `Simulator(seed=None)` creates `self.rng = numpy.random.default_rng(seed)`.
- `GammaPoissonBelief.sample_rate(rng=None)` accepts an optional
  `Generator`; falls back to a module-level default if omitted so existing
  callers are unaffected. Simulator-driven code passes `sim.rng` explicitly
  for reproducibility.

### 8. Introspection

- `sim.peek() -> float` — time of next event, or `inf` if empty.
- `Simulator.__len__` — count of pending events.
- `resource.queue` / `store.queue` — read-only tuple view of current
  waiters, for debugging/tests.
- `sim.run(until=...)` accepts a `float` (as today) or an `Event` (run
  until that event fires).

### 9. New exception types (exported from `ordo`)

- `SimulationError` — wraps an unhandled exception from a top-level
  process; carries `sim.now` and a process repr.
- `Interrupt` — raised inside an interrupted coroutine at its suspension
  point.
- `Resource(capacity < 1)` keeps its existing `ValueError`; `Store`
  mirrors this for `capacity <= 0`.

## Build order

`Event` → `Process`/interrupt/exceptions → `Resource` rewrite → `Store` →
stats → RNG → introspection. Tests land alongside each layer rather than
all at the end.

## Testing

New `tests/` package with pytest, covering:
- `Simulator`: scheduling order/tiebreaking, `run(until=...)` with float
  and `Event`, `peek()`, `__len__`.
- `Event`: succeed/fail once-only, multiple awaiters woken, `any_of`/
  `all_of`.
- `Process`: return-value propagation, exception propagation (to awaiter
  vs. top-level `run()`), `.interrupt()` at various suspension points.
- `Resource`: capacity enforcement, FIFO baseline, priority ordering,
  timeout/renege, `async with` auto-release (including release-on-
  exception), stats correctness against hand-computed scenarios.
- `Store`: put/get blocking, capacity, timeout, stats.
- RNG: seeded `Simulator` produces reproducible `sim.rng` draws;
  `GammaPoissonBelief.sample_rate(rng=sim.rng)` reproducible across runs
  with the same seed.

## Explicitly out of scope (deferred)

- Preemptive priority resources (kicking out an in-progress holder).
- Distributed/parallel simulation.
- Any visualization/reporting layer on top of `.stats`.
