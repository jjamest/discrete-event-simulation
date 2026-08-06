# DES Core Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild `ordo`'s core engine on a generic `Event` primitive, adding process interruption, exception propagation, context-manager resource release, priority queuing, a `Store` primitive, built-in stats, seeded RNG, and introspection helpers.

**Architecture:** A new `Event` class (`src/ordo/event.py`) becomes the single low-level awaitable everything resumes through. `Simulator._resume` is generalized to drive arbitrary `Event`-yielding awaitables instead of hardcoded `"sleep"`/`"acquire"` string instructions. `Resource` is rewritten on top of `Event`; `Store` and `Process` follow the same pattern. Each layer lands with tests before the next layer depends on it.

**Tech Stack:** Python 3.9+, pytest, numpy (already a dependency).

**Design doc:** `docs/plans/2026-08-06-des-core-overhaul-design.md` — read this first for the full rationale; this plan only covers the *how* and *in what order*.

---

## Important context for the engineer

- This repo is **not currently a git repository** (`git status` will fail). Do not attempt `git init` yourself — ask the user first if git tracking seems needed. Skip all "commit" steps below **unless** a `.git` directory exists by the time you reach them; if it doesn't exist, just move to the next step. If the user has since run `git init`, commit as normal with descriptive messages.
- Run tests with: `cd "c:\Users\0651j\Documents\GitHub\ordo"` then `python -m pytest tests/ -v` (pytest and numpy are already installed in `.venv`; activate it or call `.venv\Scripts\python.exe -m pytest tests/ -v` on Windows if `python` doesn't resolve to the venv).
- Existing files you'll be modifying: [simulator.py](../../src/ordo/simulator.py), [resource.py](../../src/ordo/resource.py), [__init__.py](../../src/ordo/__init__.py), [bayes.py](../../src/ordo/bayes.py).
- Existing tests you must keep passing throughout: [tests/test_resource.py](../../tests/test_resource.py), [tests/test_bayes.py](../../tests/test_bayes.py). Read `test_resource.py` now — the new `Resource` must keep every one of those 4 tests passing verbatim (they use `res.acquire()`/`res.release()`/`res.in_use`, no context manager), since it documents current expected renege/timeout semantics.
- The two existing examples ([examples/car.py](../../examples/car.py), [examples/jobs.py](../../examples/jobs.py)) use `sim.process(...)` for fire-and-forget and `sim.sleep(...)`. Both must keep running unmodified after this work (the `Process` object returned by `sim.process` must still be usable in "don't care about the return value" style).

---

## Task 1: `Event` primitive

**Files:**
- Create: `src/ordo/event.py`
- Test: `tests/test_event.py`

**Step 1: Write the failing tests**

```python
# tests/test_event.py
import pytest

from ordo.event import Event


def test_starts_untriggered():
    ev = Event()
    assert ev.triggered is False


def test_succeed_sets_triggered_and_value():
    ev = Event()
    ev.succeed(42)
    assert ev.triggered is True
    assert ev.ok is True
    assert ev.value == 42


def test_succeed_twice_raises():
    ev = Event()
    ev.succeed(1)
    with pytest.raises(RuntimeError):
        ev.succeed(2)


def test_fail_sets_triggered_and_exception():
    ev = Event()
    exc = ValueError("boom")
    ev.fail(exc)
    assert ev.triggered is True
    assert ev.ok is False
    assert ev.exception is exc


def test_fail_twice_raises():
    ev = Event()
    ev.fail(ValueError("first"))
    with pytest.raises(RuntimeError):
        ev.fail(ValueError("second"))
```

**Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_event.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ordo.event'`

**Step 3: Write minimal implementation**

```python
# src/ordo/event.py
from typing import Any, Optional


class Event:
    """A one-shot signal that coroutines can await.

    Triggers exactly once via succeed()/fail(). Awaiting an already-
    triggered event resolves immediately (see await support added in
    Task 2, once the simulator drives Event-based yields).
    """

    __slots__ = ("triggered", "ok", "value", "exception", "_callbacks")

    def __init__(self) -> None:
        self.triggered = False
        self.ok: Optional[bool] = None
        self.value: Any = None
        self.exception: Optional[BaseException] = None
        self._callbacks: list = []

    def succeed(self, value: Any = None) -> None:
        if self.triggered:
            raise RuntimeError("Event already triggered")
        self.triggered = True
        self.ok = True
        self.value = value
        self._fire_callbacks()

    def fail(self, exception: BaseException) -> None:
        if self.triggered:
            raise RuntimeError("Event already triggered")
        self.triggered = True
        self.ok = False
        self.exception = exception
        self._fire_callbacks()

    def _fire_callbacks(self) -> None:
        callbacks, self._callbacks = self._callbacks, []
        for cb in callbacks:
            cb(self)

    def add_callback(self, cb) -> None:
        """Register a callback(event) to run when the event fires.

        If already triggered, the callback runs immediately.
        """
        if self.triggered:
            cb(self)
        else:
            self._callbacks.append(cb)

    def __await__(self):
        result = yield ("event", self)
        return result
```

**Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_event.py -v`
Expected: PASS (5 tests)

**Step 5: Commit** (skip if no `.git` directory exists — see "Important context" above)

```bash
git add src/ordo/event.py tests/test_event.py
git commit -m "feat: add Event primitive"
```

---

## Task 2: Wire `Event` into the Simulator's resume loop

The simulator's `_resume` currently pattern-matches on the string instruction yielded (`"sleep"`, `"acquire"`). Generalize it to also handle `("event", event)`, where resuming means: register a callback on the event that resumes the coroutine (with the value, or raising the exception) once it fires.

**Files:**
- Modify: `src/ordo/simulator.py`
- Test: `tests/test_simulator_event.py`

**Step 1: Write the failing tests**

```python
# tests/test_simulator_event.py
from ordo.event import Event
from ordo.simulator import Simulator


def test_await_already_triggered_event_resumes_immediately():
    sim = Simulator()
    ev = Event()
    ev.succeed("hello")
    log = []

    async def waiter():
        result = await ev
        log.append(result)

    sim.process(waiter())
    sim.run(until=1)
    assert log == ["hello"]


def test_await_event_triggered_later_resumes_at_trigger_time():
    sim = Simulator()
    ev = Event()
    log = []

    async def waiter():
        result = await ev
        log.append((result, sim.now))

    def trigger_later():
        ev.succeed("go")

    sim.process(waiter())
    sim.schedule_call(5.0, trigger_later)
    sim.run(until=10)
    assert log == [("go", 5.0)]


def test_await_failed_event_raises_in_coroutine():
    sim = Simulator()
    ev = Event()
    log = []

    async def waiter():
        try:
            await ev
        except ValueError as e:
            log.append(str(e))

    sim.process(waiter())
    ev.fail(ValueError("nope"))
    sim.run(until=1)
    assert log == ["nope"]


def test_multiple_awaiters_all_resumed():
    sim = Simulator()
    ev = Event()
    log = []

    async def waiter(name):
        result = await ev
        log.append((name, result))

    sim.process(waiter("a"))
    sim.process(waiter("b"))
    sim.schedule_call(3.0, lambda: ev.succeed("done"))
    sim.run(until=10)
    assert sorted(log) == [("a", "done"), ("b", "done")]
```

**Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_simulator_event.py -v`
Expected: FAIL — `_resume` doesn't understand the `"event"` instruction (will raise or silently break; confirm the actual failure mode before moving on).

**Step 3: Implement**

In `src/ordo/simulator.py`, modify `_resume` to handle the new instruction. Note events fire callbacks synchronously in `succeed`/`fail`, but we still want the *resumption of the coroutine* to go through the scheduler (so ordering/tiebreaking stays consistent with everything else) — so the callback should `schedule(0.0, ...)`, not resume inline.

```python
from ordo.event import Event

# ...

def _resume(self, coro: Coroutine, sent_value: Any = None, sent_exception: BaseException = None) -> None:
    """Advance a coroutine one step and act on its yielded instruction."""
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
                if ev.ok:
                    self.schedule(0.0, coro, ev.value)
                else:
                    self.schedule(0.0, coro, None)
                    # value unused on the exception path; see _resume_exception below

            event.add_callback(on_fire)
    except StopIteration:
        pass
```

This needs one more piece: when the event failed, resuming must `coro.throw(...)`, not `coro.send(...)`. Extend `schedule`/the event loop to carry an optional exception through instead of overloading `value`. Simplest approach: change the callback to directly call a new helper `self._resume_from_event(coro, event)` scheduled via `schedule_call`, rather than routing failed-events through the generic `(time, tiebreaker, coroutine, value)` tuple:

```python
    elif instruction == "event":
        event: Event = payload[0]

        def on_fire(ev: Event, coro=coro) -> None:
            self.schedule_call(0.0, lambda: self._resume_from_event(coro, ev))

        event.add_callback(on_fire)


def _resume_from_event(self, coro: Coroutine, event: "Event") -> None:
    if event.ok:
        self._resume(coro, sent_value=event.value)
    else:
        self._resume(coro, sent_exception=event.exception)
```

Update `run()`'s dispatch loop: it currently does `isinstance(target, Coroutine)` and calls `self._resume(target, value)` directly for scheduled coroutines — that path is unaffected (still a plain send with no exception), so no change needed there. Double check `_resume`'s signature default (`sent_exception: BaseException = None`) — Python requires this be `Optional[BaseException] = None`; add the import.

**Step 4: Run to verify it passes**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS, including the pre-existing `test_resource.py` and `test_bayes.py` (this task must not break them — `Resource` still uses the old `"acquire"` instruction path untouched).

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/simulator.py tests/test_simulator_event.py
git commit -m "feat: drive Event-based awaits through the simulator"
```

---

## Task 3: `sim.sleep` rebuilt on `Event`; `any_of` / `all_of`

**Files:**
- Modify: `src/ordo/simulator.py`
- Test: `tests/test_simulator_event.py` (extend), new tests in same file

**Step 1: Write failing tests**

Append to `tests/test_simulator_event.py`:

```python
from ordo.event import Event


def test_sleep_still_works_as_before():
    sim = Simulator()
    log = []

    async def waiter():
        await sim.sleep(5)
        log.append(sim.now)

    sim.process(waiter())
    sim.run(until=10)
    assert log == [5.0]


def test_any_of_resolves_on_first_event():
    sim = Simulator()
    log = []
    e1, e2 = Event(), Event()

    async def waiter():
        result = await sim.any_of([e1, e2])
        log.append((result, sim.now))

    sim.process(waiter())
    sim.schedule_call(3.0, lambda: e1.succeed("first"))
    sim.schedule_call(7.0, lambda: e2.succeed("second"))
    sim.run(until=10)
    assert log == [(e1, 3.0)]
    assert e1.value == "first"


def test_all_of_resolves_when_all_fire():
    sim = Simulator()
    log = []
    e1, e2 = Event(), Event()

    async def waiter():
        result = await sim.all_of([e1, e2])
        log.append((sorted(result, key=id) == sorted([e1, e2], key=id), sim.now))

    sim.process(waiter())
    sim.schedule_call(3.0, lambda: e1.succeed())
    sim.schedule_call(7.0, lambda: e2.succeed())
    sim.run(until=10)
    assert log == [(True, 7.0)]
```

Decide the return type of `any_of`/`all_of` before implementing: `any_of` resolves to the **event object** that fired first (so callers can inspect `.value`/`.ok`); `all_of` resolves to the list of events (all now triggered). This matches the tests above.

**Step 2: Run — verify fail**

Run: `python -m pytest tests/test_simulator_event.py -v`
Expected: FAIL — `sim.any_of` / `sim.all_of` don't exist yet. (`test_sleep_still_works_as_before` should already PASS since `sleep` isn't touched yet — confirm, then proceed.)

**Step 3: Implement**

Keep `sim.sleep`'s current string-based `"sleep"` instruction — it already works and rewriting it to route through `Event` adds risk for no behavioral gain (nothing external needs a sleep's `Event` object). Leave it as-is. Only add the combinators:

```python
def any_of(self, events: list["Event"]) -> "Event":
    """Returns an Event that fires with the first event in `events` to fire."""
    result = Event()

    def on_any(ev: "Event") -> None:
        if not result.triggered:
            result.succeed(ev)

    for ev in events:
        ev.add_callback(on_any)
    return result

def all_of(self, events: list["Event"]) -> "Event":
    """Returns an Event that fires once every event in `events` has fired."""
    result = Event()
    events = list(events)
    remaining = len(events)

    if remaining == 0:
        result.succeed([])
        return result

    def on_one(_ev: "Event") -> None:
        nonlocal remaining
        remaining -= 1
        if remaining == 0:
            result.succeed(events)

    for ev in events:
        ev.add_callback(on_one)
    return result
```

Add `from ordo.event import Event` to the top of `simulator.py` (may already be present from Task 2).

**Step 4: Run — verify pass**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/simulator.py tests/test_simulator_event.py
git commit -m "feat: add any_of/all_of Event combinators"
```

---

## Task 4: `SimulationError` and `Interrupt` exception types

**Files:**
- Create: `src/ordo/exceptions.py`
- Modify: `src/ordo/__init__.py`
- Test: `tests/test_exceptions.py`

**Step 1: Write failing test**

```python
# tests/test_exceptions.py
from ordo.exceptions import SimulationError, Interrupt


def test_simulation_error_carries_time_and_process():
    exc = SimulationError("boom", sim_time=5.0, process=None)
    assert exc.sim_time == 5.0
    assert "boom" in str(exc)


def test_interrupt_carries_cause():
    exc = Interrupt(cause="breakdown")
    assert exc.cause == "breakdown"
```

**Step 2: Verify fail**

Run: `python -m pytest tests/test_exceptions.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Implement**

```python
# src/ordo/exceptions.py
from typing import Any, Optional


class SimulationError(Exception):
    """Wraps an unhandled exception from a top-level (fire-and-forget) process."""

    def __init__(self, message: str, sim_time: float, process: Any = None) -> None:
        super().__init__(f"{message} (t={sim_time}, process={process!r})")
        self.sim_time = sim_time
        self.process = process


class Interrupt(Exception):
    """Raised inside a process coroutine when it is interrupted."""

    def __init__(self, cause: Optional[Any] = None) -> None:
        super().__init__(cause)
        self.cause = cause
```

Update `src/ordo/__init__.py`:

```python
from ordo.simulator import Simulator
from ordo.resource import Resource
from ordo.bayes import GammaPoissonBelief
from ordo.event import Event
from ordo.exceptions import SimulationError, Interrupt

__all__ = [
    "Simulator",
    "Resource",
    "GammaPoissonBelief",
    "Event",
    "SimulationError",
    "Interrupt",
]
```

**Step 4: Verify pass**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/exceptions.py src/ordo/__init__.py tests/test_exceptions.py
git commit -m "feat: add SimulationError and Interrupt exception types"
```

---

## Task 5: `Process` wrapper with exception propagation

This is the most architecturally significant task: `sim.process()` changes return type from `None` to a `Process` (an `Event` subclass). Every coroutine run through the scheduler needs to know which `Process` it belongs to, so that when it finishes (via `StopIteration` or an uncaught exception) the right `Process` event fires.

**Files:**
- Create: `src/ordo/process.py`
- Modify: `src/ordo/simulator.py`
- Test: `tests/test_process.py`

**Step 1: Write failing tests**

```python
# tests/test_process.py
import pytest

from ordo.exceptions import SimulationError
from ordo.simulator import Simulator


def test_process_returns_process_object_and_fires_on_completion():
    sim = Simulator()
    log = []

    async def worker():
        await sim.sleep(3)
        return "result"

    async def watcher(proc):
        result = await proc
        log.append((result, sim.now))

    proc = sim.process(worker())
    sim.process(watcher(proc))
    sim.run(until=10)
    assert log == [("result", 3.0)]
    assert proc.triggered is True
    assert proc.value == "result"


def test_awaiter_sees_exception_from_child_process():
    sim = Simulator()
    log = []

    async def failing_worker():
        await sim.sleep(1)
        raise ValueError("child failed")

    async def watcher(proc):
        try:
            await proc
        except ValueError as e:
            log.append(str(e))

    proc = sim.process(failing_worker())
    sim.process(watcher(proc))
    sim.run(until=10)
    assert log == ["child failed"]


def test_unhandled_exception_in_top_level_process_raises_simulation_error():
    sim = Simulator()

    async def failing_worker():
        await sim.sleep(1)
        raise ValueError("boom")

    sim.process(failing_worker())

    with pytest.raises(SimulationError) as exc_info:
        sim.run(until=10)
    assert exc_info.value.sim_time == 1.0
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_fire_and_forget_process_with_no_watcher_runs_fine():
    """Existing examples rely on sim.process(...) without ever awaiting the result."""
    sim = Simulator()
    log = []

    async def worker():
        await sim.sleep(2)
        log.append(sim.now)

    sim.process(worker())
    sim.run(until=10)
    assert log == [2.0]
```

**Step 2: Verify fail**

Run: `python -m pytest tests/test_process.py -v`
Expected: FAIL — `sim.process(...)` currently returns `None`; `await proc` will error.

**Step 3: Implement**

```python
# src/ordo/process.py
from ordo.event import Event


class Process(Event):
    """An Event that fires when the wrapped coroutine finishes.

    Succeeds with the coroutine's return value, or fails with the
    exception it raised.
    """

    __slots__ = ("coro",)

    def __init__(self, coro) -> None:
        super().__init__()
        self.coro = coro

    def __repr__(self) -> str:
        name = getattr(self.coro, "__qualname__", repr(self.coro))
        return f"Process({name})"
```

Modify `src/ordo/simulator.py`:

- `process()` now creates a `Process`, schedules the coroutine, and returns the `Process`. Need a way for `_resume` to know which `Process` owns a given coroutine when it finishes — track this in a dict `self._processes: dict[Coroutine, Process]`.
- `_resume` on `StopIteration`: look up the owning `Process` (if any) and `.succeed(return_value)`. Python's `StopIteration.value` carries the coroutine's `return` value.
- `_resume` on any other exception: look up the owning `Process`. If found, `.fail(exc)`. If the `Process` has no awaiters registered (i.e., nobody called `add_callback`/awaited it — check via a new `Event.has_watchers` flag, or simpler: always fail it, and separately decide whether to raise `SimulationError` out of `run()`), raise `SimulationError` wrapping the exception, chained with `from exc`, and propagate it out of `run()`.

Simplify by NOT trying to detect "was anyone watching" — instead: always `.fail(exc)` the `Process` (so an awaiter, if any, sees it normally via the existing `on_fire`/`_resume_from_event` machinery from Task 2). *Separately*, `_resume` also needs to decide whether to let `run()`'s loop keep going or stop-and-raise. Approach: if `Process.fail()` had **zero callbacks registered at the time of failure** (meaning nobody was awaiting it — check `Event._callbacks` was empty right before `_fire_callbacks` ran), treat it as unhandled and raise `SimulationError` out of `run`.

Add an `Event` helper to make this observable without reaching into private state:

```python
# in Event, add:
def fail(self, exception: BaseException) -> None:
    if self.triggered:
        raise RuntimeError("Event already triggered")
    had_watchers = bool(self._callbacks)
    self.triggered = True
    self.ok = False
    self.exception = exception
    self._fire_callbacks()
    self.exception_handled = had_watchers  # new __slots__ entry, default False in __init__
```

Add `"exception_handled"` to `Event.__slots__` and initialize `self.exception_handled = False` in `__init__`.

Now in `simulator.py`:

```python
from ordo.process import Process

class Simulator:
    def __init__(self):
        self.now: float = 0.0
        self.events = []
        self._counter = itertools.count()
        self._process_by_coro: dict = {}

    def process(self, coroutine: Coroutine) -> "Process":
        """Starts running a process; returns a Process event that fires on completion."""
        proc = Process(coroutine)
        self._process_by_coro[coroutine] = proc
        self.schedule(delay=0.0, coroutine=coroutine)
        return proc

    def _resume(self, coro, sent_value=None, sent_exception=None) -> None:
        try:
            if sent_exception is not None:
                instruction, *payload = coro.throw(sent_exception)
            else:
                instruction, *payload = coro.send(sent_value)
            # ... unchanged instruction handling ...
        except StopIteration as stop:
            proc = self._process_by_coro.pop(coro, None)
            if proc is not None and not proc.triggered:
                proc.succeed(stop.value)
        except BaseException as exc:
            proc = self._process_by_coro.pop(coro, None)
            if proc is not None and not proc.triggered:
                proc.fail(exc)
                if not proc.exception_handled:
                    raise SimulationError(
                        f"unhandled exception in process: {exc}",
                        sim_time=self.now,
                        process=proc,
                    ) from exc
            else:
                raise
```

Import `SimulationError` from `ordo.exceptions`.

Note: `except BaseException` after `except StopIteration` is fine since `StopIteration` is caught first. But watch out — `_resume` is called from within `event.add_callback` → `schedule_call` → eventually `run()`'s loop, so raising here will propagate naturally out of the `while self.events:` loop in `run()`. Confirm this with the test (`test_unhandled_exception_in_top_level_process_raises_simulation_error`).

Also double check: when `_resume` re-raises `SimulationError`, the process dict entry was already popped, so no cleanup issue if the caller catches and continues (not supported yet, but shouldn't corrupt state).

**Step 4: Verify pass**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS. Pay close attention to `test_process.py` — if `test_awaiter_sees_exception_from_child_process` fails, the likely bug is that `Event.add_callback` for the watcher wasn't registered *before* `fail()` ran (ordering issue between when `watcher` awaits `proc` vs. when `failing_worker` raises) — check `sim.process(watcher(proc))` runs its first step (registering the await) before `t=1` when the worker fails; since both start at `t=0` and `watcher`'s first line is `await proc`, this should register the callback at `t=0`, well before the failure at `t=1`. If it doesn't, look at scheduling order/tiebreakers.

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/process.py src/ordo/simulator.py tests/test_process.py
git commit -m "feat: Process wrapper with exception propagation"
```

---

## Task 6: `Process.interrupt()`

**Files:**
- Modify: `src/ordo/process.py`, `src/ordo/simulator.py`
- Test: `tests/test_process.py` (extend)

**Step 1: Write failing tests**

Append to `tests/test_process.py`:

```python
from ordo.exceptions import Interrupt


def test_interrupt_raises_inside_sleeping_process():
    sim = Simulator()
    log = []

    async def worker():
        try:
            await sim.sleep(100)
        except Interrupt as e:
            log.append(("interrupted", e.cause, sim.now))

    proc = sim.process(worker())
    sim.schedule_call(5.0, lambda: proc.interrupt(cause="stop"))
    sim.run(until=10)
    assert log == [("interrupted", "stop", 5.0)]


def test_uncaught_interrupt_ends_process_like_any_exception():
    sim = Simulator()

    async def worker():
        await sim.sleep(100)

    proc = sim.process(worker())
    sim.schedule_call(5.0, lambda: proc.interrupt())
    sim.run(until=10)
    assert proc.triggered is True
    assert proc.ok is False
    assert isinstance(proc.exception, Interrupt)
```

Note: the second test's process has no watcher, so per Task 5's logic an unhandled `Interrupt` would raise `SimulationError` out of `run()`. Decide: should `Interrupt` be treated as "always considered handled" even with no watcher (since interrupting is an intentional external action, not a bug)? **Yes** — adjust the `_resume` exception branch so `Interrupt` never triggers the `SimulationError` escalation, regardless of watchers. Update the test above accordingly — it should NOT raise, and the process should simply end with `.ok == False`.

**Step 2: Verify fail**

Run: `python -m pytest tests/test_process.py -v`
Expected: FAIL — `Process.interrupt` doesn't exist.

**Step 3: Implement**

Add to `Process`:

```python
def interrupt(self, cause=None) -> None:
    """Raise Interrupt inside this process's coroutine at its next suspension point."""
    if self.triggered:
        return  # already finished; nothing to interrupt
    self._sim.schedule_call(0.0, lambda: self._sim._resume(self.coro, sent_exception=Interrupt(cause)))
```

This requires `Process` to know its `Simulator` — pass it in at construction. Update `Simulator.process()`:

```python
def process(self, coroutine: Coroutine) -> "Process":
    proc = Process(coroutine, sim=self)
    ...
```

And `Process.__init__(self, coro, sim)`, storing `self._sim = sim` (add to `__slots__`).

In `simulator.py`'s `_resume` exception branch, special-case `Interrupt` so it doesn't escalate to `SimulationError` even with no watchers:

```python
except BaseException as exc:
    proc = self._process_by_coro.pop(coro, None)
    if proc is not None and not proc.triggered:
        proc.fail(exc)
        if not proc.exception_handled and not isinstance(exc, Interrupt):
            raise SimulationError(...) from exc
    else:
        raise
```

Import `Interrupt` in `simulator.py`.

**Step 4: Verify pass**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/process.py src/ordo/simulator.py tests/test_process.py
git commit -m "feat: add Process.interrupt()"
```

---

## Task 7: Rewrite `Resource` on `Event`, add `async with` and `priority`

This replaces the `"acquire"` string-instruction path entirely with `Event`-based waiting, and adds priority ordering and context-manager support. Must keep all 4 existing tests in `tests/test_resource.py` passing verbatim.

**Files:**
- Modify: `src/ordo/resource.py`
- Modify: `src/ordo/simulator.py` (remove now-dead `"acquire"` instruction branch — see below)
- Test: `tests/test_resource.py` (extend, do not delete existing tests)

**Step 1: Write failing tests**

Append to `tests/test_resource.py`:

```python
import pytest


def test_async_with_auto_releases_on_normal_exit():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def user(name):
        async with res.acquire() as got:
            log.append((name, got, sim.now))
            await sim.sleep(5)
        # released here automatically

    sim.process(user("a"))
    sim.process(user("b"))
    sim.run(until=20)
    assert log == [("a", True, 0.0), ("b", True, 5.0)]
    assert res.in_use == 0


def test_async_with_releases_on_exception():
    sim = Simulator()
    res = Resource(sim, capacity=1)

    async def bad_user():
        async with res.acquire():
            raise ValueError("oops")

    proc = sim.process(bad_user())
    with pytest.raises(Exception):
        sim.run(until=10)
    assert res.in_use == 0


def test_async_with_does_not_release_on_reneged_timeout():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def holder():
        async with res.acquire():
            await sim.sleep(10)

    async def impatient():
        async with res.acquire(timeout=3) as got:
            log.append(got)

    sim.process(holder())
    sim.process(impatient())
    sim.run(until=5)
    assert log == [False]
    assert res.in_use == 1  # holder still holds; impatient's False acquire released nothing


def test_priority_ordering_higher_priority_served_first():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def holder():
        got = await res.acquire()
        log.append(("holder", got, sim.now))
        await sim.sleep(5)
        res.release()

    async def low(name, prio, delay):
        await sim.sleep(delay)
        got = await res.acquire(priority=prio)
        log.append((name, got, sim.now))
        res.release()

    sim.process(holder())
    sim.schedule(0.0, low("low_prio", 10, 0.0))
    sim.schedule(0.0, low("high_prio", 1, 0.0))
    sim.run(until=20)

    assert log[0] == ("holder", True, 0.0)
    # both arrived before t=5 when the slot frees; high_prio (lower number) should win
    assert log[1][0] == "high_prio"
    assert log[2][0] == "low_prio"
```

**Step 2: Verify fail**

Run: `python -m pytest tests/test_resource.py -v`
Expected: existing 4 tests PASS, new ones FAIL (no `async with` support, no `priority` kwarg).

**Step 3: Implement**

Rewrite `src/ordo/resource.py`:

```python
import heapq
import itertools
from typing import Coroutine, Optional, TYPE_CHECKING

from ordo.event import Event

if TYPE_CHECKING:
    from ordo.simulator import Simulator


class Resource:
    """A shared resource with limited capacity (mutex/semaphore for processes)."""

    def __init__(self, sim: "Simulator", capacity: int = 1) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.sim = sim
        self.capacity = capacity
        self.in_use = 0
        self._waiters = []  # heap of (priority, seq, Event, timeout_cancelled_flag)
        self._counter = itertools.count()
        self.stats = ResourceStats(sim, self)

    def acquire(self, timeout: Optional[float] = None, priority: int = 0):
        return _AcquireAwaitable(self, timeout, priority)

    def _request(self, timeout, priority) -> Event:
        """Returns an Event that resolves to True (granted) or False (reneged)."""
        result = Event()
        request_time = self.sim.now
        if self.in_use < self.capacity:
            self.in_use += 1
            self.sim.schedule_call(0.0, lambda: (self.stats._record_wait(0.0), result.succeed(True)))
            return result

        seq = next(self._counter)
        entry = [priority, seq, result, False]  # last field: settled flag
        heapq.heappush(self._waiters, entry)
        self.stats._enter_queue()

        if timeout is not None:
            def on_timeout():
                if entry[3]:
                    return
                entry[3] = True
                self._remove_waiter(entry)
                self.stats._leave_queue()
                result.succeed(False)
            self.sim.schedule_call(timeout, on_timeout)

        return result

    def _remove_waiter(self, entry) -> None:
        try:
            self._waiters.remove(entry)
            heapq.heapify(self._waiters)
        except ValueError:
            pass

    def release(self) -> None:
        """Free a slot, handing it to the next waiting process, if any."""
        while self._waiters:
            entry = heapq.heappop(self._waiters)
            priority, seq, result, settled = entry
            if entry[3]:
                continue
            entry[3] = True
            self.stats._leave_queue()
            self.stats._record_wait(self.sim.now - result._request_time if hasattr(result, "_request_time") else 0.0)
            result.succeed(True)
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

    def __await__(self):
        event = self.resource._request(self.timeout, self.priority)
        result = yield from event.__await__()
        self._got = result
        return result

    async def __aenter__(self):
        return await self

    async def __aexit__(self, exc_type, exc, tb):
        if self._got:
            self.resource.release()
        return False
```

This sketch has a rough edge (the `_request_time`/wait-time bookkeeping) that will be cleaned up properly in Task 9 (stats) — **for this task, stub `ResourceStats` as a no-op class** so `Resource` compiles and the priority/async-with tests pass without worrying about correctness of stats yet:

```python
# temporary stub — replaced in Task 9
class ResourceStats:
    def __init__(self, sim, resource):
        pass
    def _record_wait(self, w): pass
    def _enter_queue(self): pass
    def _leave_queue(self): pass
    def _utilization_changed(self): pass
```

Put this stub at the top of `resource.py` for now; Task 9 will replace it with the real implementation in a shared `stats.py`.

Also: **remove the dead `"acquire"` instruction branch** from `Simulator._resume` (added originally for the old `Resource.request`/`VirtualAcquire` design) — the new `Resource` no longer yields `("acquire", ...)`, it goes through `Event.__await__` → `("event", ...)`, already handled by Task 2's code. Search `simulator.py` for `elif instruction == "acquire":` and delete that branch along with the now-unused `Resource` import if any.

**Step 4: Verify pass**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS, including all 4 original `test_resource.py` tests (confirm `res.acquire()` awaited directly, without `async with`, still returns `True`/`False` as before — `_AcquireAwaitable.__await__` must support being awaited standalone, which it does).

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/resource.py src/ordo/simulator.py tests/test_resource.py
git commit -m "feat: rewrite Resource on Event; add async with and priority"
```

---

## Task 8: `Store` primitive

**Files:**
- Create: `src/ordo/store.py`
- Modify: `src/ordo/__init__.py`
- Test: `tests/test_store.py`

**Step 1: Write failing tests**

```python
# tests/test_store.py
from ordo.simulator import Simulator
from ordo.store import Store, TIMEOUT


def test_get_blocks_until_put():
    sim = Simulator()
    store = Store(sim)
    log = []

    async def consumer():
        item = await store.get()
        log.append((item, sim.now))

    async def producer():
        await sim.sleep(3)
        await store.put("widget")

    sim.process(consumer())
    sim.process(producer())
    sim.run(until=10)
    assert log == [("widget", 3.0)]


def test_put_blocks_when_at_capacity():
    sim = Simulator()
    store = Store(sim, capacity=1)
    log = []

    async def producer():
        await store.put("a")
        log.append(("put a", sim.now))
        await store.put("b")  # blocks until capacity frees
        log.append(("put b", sim.now))

    async def consumer():
        await sim.sleep(5)
        item = await store.get()
        log.append((f"got {item}", sim.now))

    sim.process(producer())
    sim.process(consumer())
    sim.run(until=10)
    assert log == [("put a", 0.0), ("got a", 5.0), ("put b", 5.0)]


def test_get_timeout_returns_sentinel():
    sim = Simulator()
    store = Store(sim)
    log = []

    async def consumer():
        item = await store.get(timeout=3)
        log.append(item is TIMEOUT)

    sim.process(consumer())
    sim.run(until=10)
    assert log == [True]


def test_fifo_order():
    sim = Simulator()
    store = Store(sim)
    log = []

    async def producer():
        await store.put("first")
        await store.put("second")

    async def consumer():
        await sim.sleep(1)
        log.append(await store.get())
        log.append(await store.get())

    sim.process(producer())
    sim.process(consumer())
    sim.run(until=10)
    assert log == ["first", "second"]
```

**Step 2: Verify fail**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Implement**

```python
# src/ordo/store.py
import heapq
import itertools
from collections import deque
from typing import Optional, TYPE_CHECKING

from ordo.event import Event

if TYPE_CHECKING:
    from ordo.simulator import Simulator

TIMEOUT = object()  # sentinel distinct from a legitimate None item


class Store:
    """Bounded FIFO item queue with blocking put/get."""

    def __init__(self, sim: "Simulator", capacity: float = float("inf")) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self.sim = sim
        self.capacity = capacity
        self._items = deque()
        self._get_waiters = []  # heap of [priority, seq, Event, settled]
        self._put_waiters = []
        self._counter = itertools.count()

    async def put(self, item, timeout: Optional[float] = None, priority: int = 0) -> bool:
        if len(self._items) < self.capacity:
            self._items.append(item)
            self._wake_a_getter()
            return True

        result = Event()
        seq = next(self._counter)
        entry = [priority, seq, result, False, item]
        heapq.heappush(self._put_waiters, entry)

        if timeout is not None:
            def on_timeout():
                if entry[3]:
                    return
                entry[3] = True
                self._remove(self._put_waiters, entry)
                result.succeed(False)
            self.sim.schedule_call(timeout, on_timeout)

        return await result

    async def get(self, timeout: Optional[float] = None, priority: int = 0):
        if self._items:
            item = self._items.popleft()
            self._wake_a_putter()
            return item

        result = Event()
        seq = next(self._counter)
        entry = [priority, seq, result, False]
        heapq.heappush(self._get_waiters, entry)

        if timeout is not None:
            def on_timeout():
                if entry[3]:
                    return
                entry[3] = True
                self._remove(self._get_waiters, entry)
                result.succeed(TIMEOUT)
            self.sim.schedule_call(timeout, on_timeout)

        return await result

    def _wake_a_getter(self) -> None:
        while self._get_waiters:
            entry = heapq.heappop(self._get_waiters)
            if entry[3]:
                continue
            entry[3] = True
            item = self._items.popleft()
            self.sim.schedule_call(0.0, lambda e=entry, i=item: e[2].succeed(i))
            self._wake_a_putter()
            return

    def _wake_a_putter(self) -> None:
        while self._put_waiters:
            entry = heapq.heappop(self._put_waiters)
            if entry[3]:
                continue
            entry[3] = True
            self._items.append(entry[4])
            self.sim.schedule_call(0.0, lambda e=entry: e[2].succeed(True))
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
```

Careful review point: `_wake_a_getter` pops an item then immediately calls `_wake_a_putter` (to let a blocked put take that freed slot) — but at that point `self._items` already has the popped item removed, so a waiting putter's item can now be appended. Trace through `test_put_blocks_when_at_capacity` by hand before running: capacity 1, `put("a")` fills it immediately; `put("b")` blocks (queued putter, item "b" stored in the waiter entry, NOT yet in `_items`); at t=5, `get()` finds `_items` non-empty, pops "a", calls `_wake_a_putter()` which pops the waiting entry for "b" and appends "b" to `_items`, then schedules the putter's resume. This matches the expected log `[("put a", 0.0), ("got a", 5.0), ("put b", 5.0)]`.

Update `src/ordo/__init__.py` to export `Store` and `TIMEOUT`:

```python
from ordo.store import Store, TIMEOUT

__all__ = [
    "Simulator", "Resource", "GammaPoissonBelief", "Event",
    "SimulationError", "Interrupt", "Store", "TIMEOUT",
]
```

**Step 4: Verify pass**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/store.py src/ordo/__init__.py tests/test_store.py
git commit -m "feat: add Store primitive"
```

---

## Task 9: Statistics — shared `stats.py`, wired into `Resource` and `Store`

**Files:**
- Create: `src/ordo/stats.py`
- Modify: `src/ordo/resource.py` (replace stub `ResourceStats`)
- Modify: `src/ordo/store.py` (add stats)
- Test: `tests/test_stats.py`

**Step 1: Write failing tests**

```python
# tests/test_stats.py
from ordo.simulator import Simulator
from ordo.resource import Resource
from ordo.store import Store


def test_resource_utilization_time_weighted():
    sim = Simulator()
    res = Resource(sim, capacity=1)

    async def user():
        await res.acquire()
        await sim.sleep(4)  # busy for 4 of the next 10 seconds
        res.release()

    sim.process(user())
    sim.run(until=10)
    assert res.stats.utilization == 0.4  # busy 4s out of 10s elapsed... see note below


def test_resource_mean_wait_time():
    sim = Simulator()
    res = Resource(sim, capacity=1)
    log = []

    async def holder():
        await res.acquire()
        await sim.sleep(5)
        res.release()

    async def waiter():
        await res.acquire()
        log.append(sim.now)
        res.release()

    sim.process(holder())
    sim.process(waiter())
    sim.run(until=10)
    assert res.stats.wait_times == [5.0]
    assert res.stats.mean_wait == 5.0


def test_store_mean_queue_length_zero_when_never_blocked():
    sim = Simulator()
    store = Store(sim)

    async def producer():
        await store.put("x")

    sim.process(producer())
    sim.run(until=10)
    assert store.stats.mean_queue_length == 0.0
```

Note on `test_resource_utilization_time_weighted`: utilization needs a well-defined denominator. Simplest correct definition: `utilization = total_busy_time / sim.now` computed at query time (i.e. as of "now", not as of when `run()` stopped in the past) — but since `stats` is queried right after `run(until=10)` while `sim.now == 10`, this is equivalent. Implement utilization as computed lazily at access time using `sim.now`, not incrementally, to avoid needing to know in advance when the "observation window" ends. If the resource is still busy at query time, count the partial elapsed busy period too.

**Step 2: Verify fail**

Run: `python -m pytest tests/test_stats.py -v`
Expected: FAIL — `res.stats` currently the no-op stub from Task 7.

**Step 3: Implement**

```python
# src/ordo/stats.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ordo.simulator import Simulator


class UsageStats:
    """Time-weighted utilization/queue-length tracking plus wait-time samples.

    Shared bookkeeping for Resource and Store. Caller is responsible for
    calling _busy_changed(n) whenever the number of in-use units changes,
    and _queue_changed(n) whenever the waiter count changes.
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
```

Rewrite `Resource` to use `UsageStats` instead of the stub, tracking busy units as `self.in_use`:

- In `_request`, when granted immediately: record wait `0.0`, call `self.stats._busy_changed(self.in_use)` after incrementing `in_use`.
- When queued: call `self.stats._queue_changed(len(self._waiters))`, and record the request time on the entry (`entry.append(self.sim.now)`, index 4) so `release()` can compute `wait = sim.now - request_time`.
- In `release()`: when granting to a waiter, compute wait from the stored request time, call `self.stats._record_wait(wait)` and `self.stats._queue_changed(len(self._waiters))` (post-pop count); when no waiters, `self.stats._busy_changed(self.in_use - 1)` then decrement `in_use`.
- Also update `_remove_waiter` (renege path) to call `self.stats._queue_changed(len(self._waiters))` after removal.

Wire the same pattern into `Store` (`_busy_units` = `len(self._items)`, capacity = `self.capacity`; queue length = combined get+put waiters, or track separately if you'd rather report them distinctly — keep it simple and track one `UsageStats` for get-side queueing since that's the more common bottleneck to monitor; document the simplification with a one-line comment).

Rewrite the exact entry list mutations carefully — this is the fiddliest part of the whole plan. Recommend the engineer write out `Resource.release()` and `Resource._request()` in full before running tests, then trace `test_resource_mean_wait_time` by hand: holder acquires at t=0 (immediate, wait=0), waiter requests at t=0 (queued, request_time=0), holder releases at t=5 → waiter granted, wait = 5 - 0 = 5. Matches expected `[5.0]`.

**Step 4: Verify pass**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS — this is the highest-risk task in the plan for regressions in `test_resource.py`'s existing 4 tests plus Task 7's new ones, since `Resource` internals are being touched again. Run the full suite, not just `test_stats.py`.

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/stats.py src/ordo/resource.py src/ordo/store.py tests/test_stats.py
git commit -m "feat: add time-weighted utilization/queue/wait stats to Resource and Store"
```

---

## Task 10: Seeded RNG on `Simulator`; `GammaPoissonBelief.sample_rate(rng=...)`

**Files:**
- Modify: `src/ordo/simulator.py`
- Modify: `src/ordo/bayes.py`
- Test: `tests/test_simulator_rng.py`, extend `tests/test_bayes.py`

**Step 1: Write failing tests**

```python
# tests/test_simulator_rng.py
from ordo.simulator import Simulator


def test_seeded_simulator_reproducible():
    sim1 = Simulator(seed=42)
    sim2 = Simulator(seed=42)
    draws1 = [sim1.rng.random() for _ in range(5)]
    draws2 = [sim2.rng.random() for _ in range(5)]
    assert draws1 == draws2


def test_unseeded_simulator_still_has_rng():
    sim = Simulator()
    assert 0.0 <= sim.rng.random() < 1.0
```

Append to `tests/test_bayes.py`:

```python
import numpy as np

def test_sample_rate_with_explicit_generator_reproducible():
    from ordo.bayes import GammaPoissonBelief
    belief = GammaPoissonBelief(shape=3.0, rate=2.0)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    assert belief.sample_rate(rng=rng1) == belief.sample_rate(rng=rng2)
```

**Step 2: Verify fail**

Run: `python -m pytest tests/test_simulator_rng.py tests/test_bayes.py -v`
Expected: FAIL — `Simulator(seed=...)` / `sim.rng` don't exist; `sample_rate` doesn't accept `rng`.

**Step 3: Implement**

In `src/ordo/simulator.py`, add to `__init__`:

```python
import numpy as np

def __init__(self, seed=None) -> None:
    self.now: float = 0.0
    self.events = []
    self._counter = itertools.count()
    self._process_by_coro = {}
    self.rng = np.random.default_rng(seed)
```

In `src/ordo/bayes.py`, change `sample_rate`:

```python
def sample_rate(self, rng=None) -> float:
    """Thompson Sampling: Draw a plausible service rate lambda from posterior distribution."""
    generator = rng if rng is not None else np.random
    return generator.gamma(shape=self.shape, scale=1.0 / self.rate)
```

Note `np.random.gamma` (module-level) and `Generator.gamma` (instance method) have the same call signature, so this fallback works without an if/else on argument shape.

**Step 4: Verify pass**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/simulator.py src/ordo/bayes.py tests/test_simulator_rng.py tests/test_bayes.py
git commit -m "feat: seeded per-Simulator RNG; GammaPoissonBelief accepts explicit generator"
```

---

## Task 11: Introspection — `peek()`, `__len__`, `run(until=Event)`

**Files:**
- Modify: `src/ordo/simulator.py`
- Test: `tests/test_simulator_introspection.py`

**Step 1: Write failing tests**

```python
# tests/test_simulator_introspection.py
from ordo.event import Event
from ordo.simulator import Simulator


def test_peek_returns_next_event_time():
    sim = Simulator()
    assert sim.peek() == float("inf")

    async def worker():
        await sim.sleep(5)

    sim.process(worker())
    assert sim.peek() == 0.0  # process itself scheduled at t=0
    sim.run(until=0.0)
    assert sim.peek() == 5.0


def test_len_reflects_pending_events():
    sim = Simulator()
    assert len(sim) == 0

    async def worker():
        await sim.sleep(1)

    sim.process(worker())
    assert len(sim) == 1


def test_run_until_event():
    sim = Simulator()
    log = []
    ev = Event()

    async def worker():
        await sim.sleep(3)
        log.append(("worker done", sim.now))
        ev.succeed()

    async def late():
        await sim.sleep(100)
        log.append(("late", sim.now))

    sim.process(worker())
    sim.process(late())
    sim.run(until=ev)
    assert log == [("worker done", 3.0)]  # stops before "late" fires at t=100
```

**Step 2: Verify fail**

Run: `python -m pytest tests/test_simulator_introspection.py -v`
Expected: FAIL — `peek`/`__len__`/`run(until=Event)` don't exist yet.

**Step 3: Implement**

Add to `Simulator`:

```python
def peek(self) -> float:
    """Time of the next scheduled event, or inf if none pending."""
    return self.events[0][0] if self.events else float("inf")

def __len__(self) -> int:
    return len(self.events)
```

Modify `run()` to accept an `Event`:

```python
def run(self, until=float("inf")) -> None:
    """Main event loop. `until` may be a time (float) or an Event to run until."""
    if isinstance(until, Event):
        stop_event = until
        until = float("inf")
    else:
        stop_event = None

    while self.events:
        if stop_event is not None and stop_event.triggered:
            break

        event_time, _, target, value = heapq.heappop(self.events)

        if event_time > until:
            break

        self.now = event_time

        if isinstance(target, Coroutine):
            self._resume(target, value)
        else:
            target()

        if stop_event is not None and stop_event.triggered:
            break
```

Import `Event` in `simulator.py` (should already be imported from earlier tasks).

Double-check the "peek before running" test: `sim.process(worker())` schedules the coroutine itself at `t=0` (that's the outer process, not yet inside `sleep`), so `peek()` right after `process()` is `0.0`, not `5.0` — confirmed by the test above, which asserts `0.0` first, then runs one step (`until=0.0`), then re-checks `peek() == 5.0`.

**Step 4: Verify pass**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS — run the complete suite one final time here.

**Step 5: Commit** (skip if no git)

```bash
git add src/ordo/simulator.py tests/test_simulator_introspection.py
git commit -m "feat: add peek(), __len__, and Event-based run(until=...)"
```

---

## Task 12: Final full-suite verification and example sanity check

**Files:** none modified; verification only.

**Step 1:** Run the complete test suite:

Run: `python -m pytest tests/ -v`
Expected: ALL PASS (should be roughly 35-40 tests across all new + existing files).

**Step 2:** Manually run both examples to confirm nothing broke at the integration level (they don't use pytest):

Run: `python examples/car.py`
Expected: prints alternating "Start parking..." / "Start driving..." lines up to t=15, no exceptions.

Run: `python examples/jobs.py`
Expected: prints routing/completion logs up to t=35 and a final summary per server, no exceptions.

**Step 3:** If either example fails, that's a regression from this plan — stop and diagnose before considering the work done (do not silently patch the examples to work around a behavior change; find which task's change broke the documented contract, per @superpowers:systematic-debugging if the cause isn't immediately obvious from the traceback).

**Step 4: Commit** (skip if no git) — only if Step 3 required a fix:

```bash
git add -A
git commit -m "fix: <describe regression fix>"
```

---

## Post-plan cleanup note

`docs/plans/2026-08-06-des-core-overhaul-design.md` (the design doc) references `Resource.acquire` returning a `VirtualAcquire` in a couple of places from the original codebase description — that's fine, it's a historical design doc, not living documentation; no need to edit it after implementation diverges in small ways (e.g. the class is now `_AcquireAwaitable`, not `VirtualAcquire`).
