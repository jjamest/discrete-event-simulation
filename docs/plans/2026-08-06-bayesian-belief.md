# Bayesian Belief for Event Rates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `GammaPoissonBelief` class that lets a simulation process maintain and update an online Bayesian belief about an unknown event rate λ from observed inter-event delays, and read out its posterior mean/variance.

**Architecture:** New standalone module `src/ordo/bayes.py`, fully decoupled from `simulator.py`. A process (or any code) creates a `GammaPoissonBelief`, calls `.observe(delay)` each time it sees an inter-event delay, and reads `.mean` / `.variance` whenever it wants its current best estimate of λ. No changes to the simulator event loop.

**Tech Stack:** Python stdlib for the implementation (closed-form Gamma mean/variance, no external deps needed for the class itself). numpy added as a dependency for generating synthetic Exponential samples in the convergence test. pytest added as the test runner (project currently has none).

---

## Task 0: Project test/dependency setup

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add pytest and numpy to project config**

Add an optional dependency group for dev/test tooling:

```toml
[project.optional-dependencies]
dev = ["pytest>=8", "numpy>=1.26"]
```

Append this section at the end of `pyproject.toml` (after the `[tool.setuptools.packages.find]` block).

**Step 2: Install into the venv**

Run: `./.venv/Scripts/python.exe -m pip install -e ".[dev]"`
Expected: pytest and numpy install successfully, `ordo` installs in editable mode.

**Step 3: Verify pytest runs (with no tests yet)**

Run: `./.venv/Scripts/python.exe -m pytest --collect-only`
Expected: exits cleanly, "no tests collected" (or similar), no import errors.

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pytest and numpy dev dependencies"
```

---

## Task 1: GammaPoissonBelief — construction and defaults

**Files:**
- Create: `src/ordo/bayes.py`
- Test: `tests/test_bayes.py`

**Step 1: Write the failing test**

Create `tests/test_bayes.py`:

```python
from ordo.bayes import GammaPoissonBelief


def test_default_prior_is_gamma_1_1():
    belief = GammaPoissonBelief()
    assert belief.shape == 1.0
    assert belief.rate == 1.0


def test_custom_prior():
    belief = GammaPoissonBelief(shape=2.0, rate=3.0)
    assert belief.shape == 2.0
    assert belief.rate == 3.0
```

**Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_bayes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ordo.bayes'`

**Step 3: Write minimal implementation**

Create `src/ordo/bayes.py`:

```python
class GammaPoissonBelief:
    """Online Bayesian belief about an event rate, using the
    Gamma-Exponential conjugate model updated from observed inter-event
    delays."""

    def __init__(self, shape: float = 1.0, rate: float = 1.0) -> None:
        self.shape = shape
        self.rate = rate
```

**Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_bayes.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/ordo/bayes.py tests/test_bayes.py
git commit -m "feat: add GammaPoissonBelief with configurable prior"
```

---

## Task 2: observe() — conjugate update from a delay

**Files:**
- Modify: `src/ordo/bayes.py`
- Test: `tests/test_bayes.py`

**Step 1: Write the failing test**

Append to `tests/test_bayes.py`:

```python
def test_observe_updates_shape_and_rate():
    belief = GammaPoissonBelief(shape=1.0, rate=1.0)

    belief.observe(2.0)

    assert belief.shape == 2.0
    assert belief.rate == 3.0


def test_observe_multiple_delays_accumulates():
    belief = GammaPoissonBelief(shape=1.0, rate=1.0)

    belief.observe(2.0)
    belief.observe(0.5)
    belief.observe(1.5)

    assert belief.shape == 4.0
    assert belief.rate == 5.0
```

**Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_bayes.py -v`
Expected: FAIL with `AttributeError: 'GammaPoissonBelief' object has no attribute 'observe'`

**Step 3: Write minimal implementation**

Add to `GammaPoissonBelief` in `src/ordo/bayes.py`:

```python
    def observe(self, delay: float) -> None:
        """Update the belief with one observed inter-event delay."""
        self.shape += 1
        self.rate += delay
```

**Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_bayes.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/ordo/bayes.py tests/test_bayes.py
git commit -m "feat: add GammaPoissonBelief.observe for conjugate updates"
```

---

## Task 3: mean and variance properties

**Files:**
- Modify: `src/ordo/bayes.py`
- Test: `tests/test_bayes.py`

**Step 1: Write the failing test**

Append to `tests/test_bayes.py`:

```python
def test_mean_and_variance_of_prior():
    belief = GammaPoissonBelief(shape=2.0, rate=4.0)

    assert belief.mean == 0.5
    assert belief.variance == 0.125


def test_mean_and_variance_after_observations():
    belief = GammaPoissonBelief(shape=1.0, rate=1.0)

    belief.observe(2.0)
    belief.observe(0.5)

    # shape=3.0, rate=3.5
    assert belief.mean == 3.0 / 3.5
    assert belief.variance == 3.0 / (3.5 ** 2)
```

**Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_bayes.py -v`
Expected: FAIL with `AttributeError: 'GammaPoissonBelief' object has no attribute 'mean'`

**Step 3: Write minimal implementation**

Add to `GammaPoissonBelief` in `src/ordo/bayes.py`:

```python
    @property
    def mean(self) -> float:
        """Posterior mean of lambda."""
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        """Posterior variance of lambda."""
        return self.shape / self.rate ** 2
```

**Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_bayes.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/ordo/bayes.py tests/test_bayes.py
git commit -m "feat: add posterior mean and variance properties"
```

---

## Task 4: Convergence test with synthetic data

**Files:**
- Test: `tests/test_bayes.py`

**Step 1: Write the test**

Append to `tests/test_bayes.py`:

```python
import numpy as np


def test_belief_converges_to_true_rate():
    true_lambda = 4.0
    rng = np.random.default_rng(seed=42)
    delays = rng.exponential(scale=1.0 / true_lambda, size=5000)

    belief = GammaPoissonBelief()
    for delay in delays:
        belief.observe(float(delay))

    assert belief.mean == pytest.approx(true_lambda, rel=0.05)
```

Add `import pytest` at the top of `tests/test_bayes.py` alongside the existing imports.

**Step 2: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_bayes.py -v`
Expected: PASS (7 tests). If it fails due to randomness, confirm the seed is fixed at 42 and rel tolerance is 0.05 before adjusting.

**Step 3: Commit**

```bash
git add tests/test_bayes.py
git commit -m "test: verify GammaPoissonBelief converges to true rate"
```

---

## Task 5: Export from package root

**Files:**
- Modify: `src/ordo/__init__.py`

**Step 1: Update the failing-ish check (manual, no new test needed — covered by import)**

Modify `src/ordo/__init__.py`:

```python
from ordo.simulator import Simulator
from ordo.bayes import GammaPoissonBelief

__all__ = ["Simulator", "GammaPoissonBelief"]
```

**Step 2: Run full test suite**

Run: `./.venv/Scripts/python.exe -m pytest -v`
Expected: all tests PASS, no import errors.

**Step 3: Commit**

```bash
git add src/ordo/__init__.py
git commit -m "feat: export GammaPoissonBelief from ordo package root"
```
