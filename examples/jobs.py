import numpy as np

from ordo.bayes import GammaExponentialBelief
from ordo.simulator import Simulator


class ServerNode:
    def __init__(
        self,
        node_id: int,
        true_service_rate: float,
        prior_shape: float = 2.0,
        prior_rate: float = 10.0,
    ):
        self.node_id = node_id
        self.true_rate = true_service_rate # Ground truth parameter (Unknown to the router)
        self.belief = GammaExponentialBelief(shape=prior_shape, rate=prior_rate) # updating live
        self.jobs_processed = 0

    async def process_job(self, sim: Simulator, job_id: int):
        # Sample duration from ground truth exponential distribution
        actual_duration = np.random.exponential(scale=1.0 / self.true_rate)

        # Pause during processing time
        await sim.sleep(actual_duration)

        # when job completes update Bayesian belief with observed delay
        self.belief.observe(actual_duration)
        self.jobs_processed += 1

        print(
            f"[{sim.now:5.1f}s] Node {self.node_id} finished Job {job_id:02d} in {actual_duration:4.2f}s "
            f"| New Est. Rate: {self.belief.mean:4.2f} (True: {self.true_rate:4.2f})"
        )


async def smart_job_router(sim: Simulator, servers: list[ServerNode]):
    job_id = 1
    while True:
        # we use thompson sampling
        # Draw a random rate sample from each server's posterior Gamma distribution
        # Select the server with the highest sampled processing speed
        sampled_rates = [s.belief.sample_rate() for s in servers]
        chosen_server_idx = int(np.argmax(sampled_rates))
        chosen_server = servers[chosen_server_idx]

        print(
            f"[{sim.now:5.1f}s] Routing Job {job_id:02d} to Node {chosen_server.node_id} "
            f"(Sampled Rates: {[round(r, 2) for r in sampled_rates]})"
        )

        # Dispatch job to chosen server node
        sim.process(chosen_server.process_job(sim, job_id))

        # Wait for next incoming job arrival (Poisson arrivals ~ 1 job every 0.8s)
        inter_arrival_time = np.random.exponential(scale=0.8)
        await sim.sleep(inter_arrival_time)
        job_id += 1


if __name__ == "__main__":
    np.random.seed(21)

    sim = Simulator()

    # Three servers with hidden ground-truth speeds
    servers = [
        ServerNode(node_id=0, true_service_rate=0.2), # 0.2 jobs/sec ~ 5s/job
        ServerNode(node_id=1, true_service_rate=0.5), # 0.5 jobs/sec ~ 2s/job
        ServerNode(node_id=2, true_service_rate=1.5), # 1.5 jobs/sec ~ 0.66s/job
    ]

    sim.process(smart_job_router(sim, servers))
    sim.run(until=350.0)

    for s in servers:
        print(
            f"Node {s.node_id} | Jobs Handled: {s.jobs_processed:2d} "
            f"| True Rate: {s.true_rate:.2f} "
            f"| Posterior Est. Rate: {s.belief.mean:.2f} "
            f"| Est. Delay: {s.belief.expected_delay:.2f}s"
        )
