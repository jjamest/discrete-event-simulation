# Ordo

A discrete-event simulator for Python, built on `async`/`await`. Processes are
plain coroutines that suspend on `sleep()`, shared resources, queues, or other
events, and a single-threaded event loop drives them forward in simulated
time.

## Install

```bash
pip install ordo-des
```

```python
from ordo import Simulator
```

## Quick start

```python
from ordo import Simulator

sim = Simulator()
sim.schedule_call(delay=1.0, func=lambda: print("hello"))
sim.run(until=10.0)
```

## Processes

A process is any coroutine started with `sim.process(...)`. It can pause with
`await sim.sleep(delay)` and simulated time advances around it:

```python
from ordo import Simulator

class Car:
    def __init__(self, sim):
        self.sim = sim
        sim.process(self.run())

    async def run(self):
        while True:
            print(f"parking at {self.sim.now}")
            await self.sim.sleep(5)
            print(f"driving at {self.sim.now}")
            await self.sim.sleep(2)

sim = Simulator()
Car(sim)
sim.run(until=15)
```

`sim.process(coro)` returns a `Process` (itself an `Event`) that fires with
the coroutine's return value on completion, or fails with the exception it
raised. Awaiting a `Process` lets one process wait on another.

## Events

`Event` is the low-level primitive everything else builds on: a one-shot
signal that coroutines can `await`. It fires exactly once, via `succeed(value)`
or `fail(exception)`, and any callback/coroutine waiting on it resumes then.

```python
from ordo import Event

sim.any_of([event_a, event_b])   # fires when the first of these fires
sim.all_of([event_a, event_b])   # fires once every one of these has fired
```

## Resources

`Resource` models a mutex/semaphore with limited capacity — e.g. a fixed pool
of workers or machines:

```python
res = sim.resource(capacity=2)

async def task(sim, res):
    async with res.acquire() as got:
        await sim.sleep(3)  # do work while holding a slot
# slot is released automatically on block exit, even on exception
```

`acquire()` also works as a plain awaitable (`got = await res.acquire()`,
paired with a manual `res.release()`), supports `priority` (lower value is
served first) and an optional `timeout`, after which the caller gives up its
place in line and the await resolves to `False` instead of `True`.

Every `Resource` exposes `.stats` (see [Stats](#stats)) for utilization,
queue length, and wait-time tracking.

## Stores

`Store` is a bounded FIFO queue with blocking `put`/`get`, for
producer/consumer pipelines:

```python
from ordo import Store, TIMEOUT

store = Store(sim, capacity=10)

async def producer(sim, store):
    await store.put("item")

async def consumer(sim, store):
    item = await store.get(timeout=5)
    if item is TIMEOUT:
        print("gave up waiting")
```

`put`/`get` accept `priority` and `timeout` just like `Resource.acquire`; a
timed-out `get()` resolves to the `TIMEOUT` sentinel (distinct from any
legitimate item value, including `None`).

## Stats

`Resource` and `Store` both expose a `.stats` object (`UsageStats`) with:

- `utilization` time-weighted fraction of capacity in use
- `mean_queue_length` time-weighted average queue length
- `mean_wait` mean time waiters spent queued before being served
- `wait_times` the raw list of recorded wait samples

These are computed lazily against `sim.now`, so they're accurate even if
queried mid-simulation.

## Interrupts

Any running process can be interrupted from the outside:

```python
proc = sim.process(worker())
proc.interrupt(cause="cancelled")
```

This raises `Interrupt(cause)` inside the coroutine at its next suspension
point. Ordo's internal bookkeeping (generation counters on each `Process`)
ensures a stale, already-scheduled resumption from before the interrupt is
never delivered on top of it, and any `Resource`/`Store` wait the coroutine
abandoned mid-await is cleaned up automatically (a queued waiter is
dequeued; an already-granted-but-undelivered slot or item is handed back).

An unhandled exception (other than `Interrupt`) that escapes a fire-and-forget
process (one nobody is awaiting) is re-raised from `sim.run()` wrapped in a
`SimulationError`.

## Bayesian belief tracking

`GammaExponentialBelief` is an online Gamma-Exponential conjugate model for
learning an unknown event rate (e.g. a server's true service rate) from
observed inter-event delays — handy for adaptive routing policies inside a
simulation:

```python
from ordo import GammaExponentialBelief

belief = GammaExponentialBelief(shape=2.0, rate=10.0)  # prior
belief.observe(observed_delay)                     # update after each observation

belief.mean            # posterior mean of the rate (lambda)
belief.variance        # posterior variance
belief.expected_delay  # posterior mean delay (1 / lambda), inf if undefined
belief.sample_rate(rng)  # Thompson-sampling draw from the posterior
```

See [examples/jobs.py](examples/jobs.py) for a full Thompson-sampling job
router built on this.

## Reproducibility

Each `Simulator` owns its own seeded `numpy.random.Generator`:

```python
sim = Simulator(seed=42)
sim.rng.exponential(scale=1.0)
```

## Running until a time or an event

```python
sim.run(until=100.0)        # run until no event remains at or before t=100
sim.run(until=some_event)   # run until that event fires (or the heap drains)
sim.peek()                  # time of the next scheduled event, or inf
len(sim)                    # number of events currently pending
```

## Examples

- [examples/car.py](examples/car.py) — minimal process/sleep loop
- [examples/jobs.py](examples/jobs.py) — Thompson-sampling job router
