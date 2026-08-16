"""Performance comparison: Ordo vs SimPy.

Runs matched scenarios on both engines and reports wall-clock time and
peak memory. Scenarios are chosen to stress different parts of the
engine: raw event-loop throughput, resource contention (M/M/c queue),
and a bounded producer/consumer pipeline (Store vs simpy.Store).

Usage:
    python benchmarks/bench_ordo_vs_simpy.py [--scale N]
"""

import argparse
import gc
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import simpy

from ordo import Simulator


@dataclass
class Result:
    label: str
    engine: str
    seconds: float
    peak_mb: float


def measure(label, engine, fn):
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Result(label, engine, elapsed, peak / 1e6)


# ---------------------------------------------------------------------------
# Scenario 1: raw event throughput (no contention, just sleep/reschedule)
# ---------------------------------------------------------------------------

def ordo_raw_events(n):
    sim = Simulator()

    async def ticker():
        for _ in range(n):
            await sim.sleep(1.0)

    sim.process(ticker())
    sim.run(until=float(n) + 1)


def simpy_raw_events(n):
    env = simpy.Environment()

    def ticker():
        for _ in range(n):
            yield env.timeout(1.0)

    env.process(ticker())
    env.run(until=float(n) + 1)


# ---------------------------------------------------------------------------
# Scenario 2: M/M/c queue - many short-lived processes contending for a
# capacity-limited resource (car-wash style)
# ---------------------------------------------------------------------------

def ordo_mmc_bounded(n_arrivals, capacity=4, seed=7):
    sim = Simulator(seed=seed)
    res = sim.resource(capacity=capacity)
    rng = np.random.default_rng(seed)
    remaining = n_arrivals

    async def customer():
        async with res.acquire():
            await sim.sleep(rng.exponential(1.0))

    async def driver():
        nonlocal remaining
        while remaining > 0:
            await sim.sleep(rng.exponential(0.5))
            sim.process(customer())
            remaining -= 1

    sim.process(driver())
    # Run long enough for arrivals to finish + drain the queue.
    sim.run(until=n_arrivals * 0.5 * 3 + 100)


def simpy_mmc_bounded(n_arrivals, capacity=4, seed=7):
    env = simpy.Environment()
    res = simpy.Resource(env, capacity=capacity)
    rng = np.random.default_rng(seed)
    remaining = n_arrivals

    def customer():
        with res.request() as req:
            yield req
            yield env.timeout(rng.exponential(1.0))

    def driver():
        nonlocal remaining
        while remaining > 0:
            yield env.timeout(rng.exponential(0.5))
            env.process(customer())
            remaining -= 1

    env.process(driver())
    env.run(until=n_arrivals * 0.5 * 3 + 100)


# ---------------------------------------------------------------------------
# Scenario 3: bounded producer/consumer pipeline (Store)
# ---------------------------------------------------------------------------

def ordo_store_pipeline(n_items, capacity=50, n_consumers=8):
    from ordo import Store

    sim = Simulator(seed=1)
    store = Store(sim, capacity=capacity)
    produced = {"n": 0}
    consumed = {"n": 0}

    async def producer():
        for i in range(n_items):
            await store.put(i)
            produced["n"] += 1

    async def consumer():
        while consumed["n"] < n_items:
            item = await store.get()
            consumed["n"] += 1

    sim.process(producer())
    for _ in range(n_consumers):
        sim.process(consumer())
    sim.run(until=10 ** 9)


def simpy_store_pipeline(n_items, capacity=50, n_consumers=8):
    env = simpy.Environment()
    store = simpy.Store(env, capacity=capacity)
    consumed = {"n": 0}

    def producer():
        for i in range(n_items):
            yield store.put(i)

    def consumer():
        while consumed["n"] < n_items:
            yield store.get()
            consumed["n"] += 1

    env.process(producer())
    for _ in range(n_consumers):
        env.process(consumer())
    env.run(until=10 ** 9)


# ---------------------------------------------------------------------------

def print_table(results):
    by_label = {}
    for r in results:
        by_label.setdefault(r.label, {})[r.engine] = r

    header = f"{'scenario':<28}{'ordo (s)':>12}{'simpy (s)':>12}{'ratio':>10}{'ordo MB':>10}{'simpy MB':>10}"
    print(header)
    print("-" * len(header))
    for label, engines in by_label.items():
        o = engines.get("ordo")
        s = engines.get("simpy")
        ratio = f"{o.seconds / s.seconds:.2f}x" if o and s and s.seconds else "n/a"
        print(
            f"{label:<28}"
            f"{o.seconds if o else float('nan'):>12.4f}"
            f"{s.seconds if s else float('nan'):>12.4f}"
            f"{ratio:>10}"
            f"{o.peak_mb if o else float('nan'):>10.2f}"
            f"{s.peak_mb if s else float('nan'):>10.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1, help="multiplier for problem sizes")
    args = parser.parse_args()
    scale = args.scale

    n_raw = 200_000 * scale
    n_mmc = 20_000 * scale
    n_store = 50_000 * scale

    # Warm up both libraries' lazy imports (e.g. numpy's default_rng() pulls
    # in `random`/`secrets` on first use) before measuring, so neither one's
    # first scenario gets unfairly charged for the other's import cost.
    ordo_raw_events(10)
    simpy_raw_events(10)

    results = []

    results.append(measure("raw_events", "ordo", lambda: ordo_raw_events(n_raw)))
    results.append(measure("raw_events", "simpy", lambda: simpy_raw_events(n_raw)))

    results.append(measure("mmc_resource_queue", "ordo", lambda: ordo_mmc_bounded(n_mmc)))
    results.append(measure("mmc_resource_queue", "simpy", lambda: simpy_mmc_bounded(n_mmc)))

    results.append(measure("store_pipeline", "ordo", lambda: ordo_store_pipeline(n_store)))
    results.append(measure("store_pipeline", "simpy", lambda: simpy_store_pipeline(n_store)))

    print()
    print_table(results)


if __name__ == "__main__":
    main()
