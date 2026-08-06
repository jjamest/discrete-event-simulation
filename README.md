# Ordo

A discrete event simulator for Python.

## Install (editable, for development)

```bash
pip install -e .
```

## Usage

```python
from ordo import Simulator

sim = Simulator()
sim.schedule(delay=1.0, action=lambda: print("hello"))
sim.run(until=10.0)
```

See [examples/bank_queue.py](examples/bank_queue.py) for a full example.
