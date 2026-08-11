import numpy as np

from ordo.simulator import Simulator


class CarWash:
    def __init__(self, sim: Simulator, num_bays: int, wash_duration_mean: float):
        self.resource = sim.resource(capacity=num_bays)
        self.wash_duration_mean = wash_duration_mean
        self.cars_washed = 0


async def wash_car(sim: Simulator, wash: CarWash, car_id: int):
    arrival_time = sim.now
    print(f"[{sim.now:5.1f}] Car {car_id:02d} arrives")

    async with wash.resource.acquire():
        wait = sim.now - arrival_time
        print(f"[{sim.now:5.1f}] Car {car_id:02d} enters bay (waited {wait:4.2f})")

        wash_duration = np.random.exponential(scale=wash.wash_duration_mean)
        await sim.sleep(wash_duration)

        wash.cars_washed += 1
        print(f"[{sim.now:5.1f}] Car {car_id:02d} leaves after {wash_duration:4.2f} wash")


async def car_arrivals(sim: Simulator, wash: CarWash, inter_arrival_mean: float):
    car_id = 1
    while True:
        inter_arrival_time = np.random.exponential(scale=inter_arrival_mean)
        await sim.sleep(inter_arrival_time)

        sim.process(wash_car(sim, wash, car_id))
        car_id += 1


np.random.seed(7)

sim = Simulator()

wash = CarWash(sim, num_bays=2, wash_duration_mean=6.0)

sim.process(car_arrivals(sim, wash, inter_arrival_mean=4.0))
sim.run(until=100.0)

stats = wash.resource.stats
print()
print(f"Cars washed:          {wash.cars_washed}")
print(f"Bay utilization:      {stats.utilization:.2%}")
print(f"Mean queue length:    {stats.mean_queue_length:.2f}")
print(f"Mean wait time:       {stats.mean_wait:.2f}")
