from ordo.exceptions import SimulationError, Interrupt


def test_simulation_error_carries_time_and_process():
    exc = SimulationError("boom", sim_time=5.0, process=None)
    assert exc.sim_time == 5.0
    assert "boom" in str(exc)


def test_interrupt_carries_cause():
    exc = Interrupt(cause="breakdown")
    assert exc.cause == "breakdown"
