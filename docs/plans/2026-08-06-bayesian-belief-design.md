# Bayesian Belief for Event Rates — Design

## Goal

Let a simulation process maintain an online Bayesian belief about an unknown
event rate λ, updated as it observes inter-event delays during the
simulation, and read out that belief (mean/variance) to drive its own
behavior.

## Scope

- New module `src/ordo/bayes.py`, independent of `simulator.py`.
- `GammaPoissonBelief` class:
  - `GammaPoissonBelief(shape: float = 1.0, rate: float = 1.0)` — defaults to
    a weak Gamma(1, 1) prior.
  - `.observe(delay: float) -> None` — conjugate update for Exponential
    waiting times: `shape += 1`, `rate += delay`.
  - `.mean -> float` property — `shape / rate`.
  - `.variance -> float` property — `shape / rate**2`.
- Mutated in place: a process holds one belief object across its lifetime
  and feeds it observations as they occur.
- Fully decoupled from the simulator event loop — a process calls
  `belief.observe(...)` itself (e.g. after `await sim.sleep(...)` or upon
  observing an external arrival gap), and any code can read `belief.mean` /
  `belief.variance` at any time.

## Dependencies

numpy is an acceptable dependency for this project going forward (used for
tests generating synthetic samples now; may be used for sampling/quantiles
in later extensions).

## Testing

- Pure numeric unit tests against hand-computed shape/rate/mean/variance
  for known sequences of observed delays.
- Convergence test: feed many synthetic Exponential(λ) samples (via numpy
  RNG) and assert `belief.mean` converges toward the true λ.

## Explicitly out of scope (deferred)

- Sampling from the posterior (`belief.sample()`).
- Credible intervals (would need scipy or a manual Gamma quantile).
- Other conjugate families (e.g. Beta-Bernoulli).
- Count-window updates (`observe_count(count, duration)`).
- Any simulator-level auto-wiring (e.g. `sim.sleep` automatically updating
  a belief).

These are natural follow-ups once the core `GammaPoissonBelief` lands.
