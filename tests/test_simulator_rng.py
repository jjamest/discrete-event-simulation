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
