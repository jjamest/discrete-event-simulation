from ordo.simulator import Simulator

class Car:
    def __init__(self, env):
        self.env = env
        self.env.process(self.run())

    async def run(self):
        while True:
            print("Start parking and charging at %d" % self.env.now)
            charge_duration = 5

            await self.charge(charge_duration)

            print("Start driving at %d" % self.env.now)
            trip_duration = 2
            await self.env.sleep(trip_duration)

    async def charge(self, duration):
        await self.env.sleep(duration)


env = Simulator()
car = Car(env)
env.run(until=15)
