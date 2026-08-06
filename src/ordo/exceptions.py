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
