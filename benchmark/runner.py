import time

from benchmark.workloads.generator import WorkloadGenerator
from core.queue.request import InferenceRequest
from core.scheduler.batch_scheduler import DynamicBatchScheduler
from inference.service import InferenceService


class BenchmarkRunner:
    def __init__(
        self,
        request_count: int = 100,
        max_batch_size: int = 8,
        max_batch_delay_ms: float = 10.0,
    ):
        self.request_count = request_count

        self.generator = WorkloadGenerator()

        self.scheduler = DynamicBatchScheduler(
            inference_service=InferenceService(),
            max_queue_size=request_count + 10,
        )

        self.scheduler.batch_config.max_batch_size = max_batch_size
        self.scheduler.batch_config.max_batch_delay_ms = max_batch_delay_ms

    def run(self):
        requests_data = self.generator.generate_requests(self.request_count)

        requests = [
            InferenceRequest(values=values)
            for values in requests_data
        ]

        start_time = time.perf_counter()

        self.scheduler.start()

        for request in requests:
            self.scheduler.submit(request)

        while self.scheduler.metrics()["requests_completed"] < self.request_count:
            time.sleep(0.01)

        end_time = time.perf_counter()

        self.scheduler.stop()

        runtime_ms = (end_time - start_time) * 1000

        metrics = self.scheduler.metrics()
        metrics["runtime_ms"] = runtime_ms

        return metrics


if __name__ == "__main__":
    runner = BenchmarkRunner(request_count=100)

    results = runner.run()

    print("=== GPUFlow-X Benchmark ===")
    print("Requests:", results["requests_submitted"])
    print("Completed:", results["requests_completed"])
    print("Failed:", results["requests_failed"])
    print("Batches:", results["batches_executed"])
    print("Runtime (ms):", results["runtime_ms"])
    print("Telemetry events:", results["telemetry_events"])
