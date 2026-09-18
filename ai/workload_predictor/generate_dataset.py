import csv
from pathlib import Path

from ai.workload_predictor.dataset import WorkloadDatasetBuilder
from core.batching.batch_config import BatchConfig
from core.policy.adaptive_policy import AdaptivePolicy
from core.queue.request import InferenceRequest
from core.scheduler.adaptive_scheduler import AdaptiveScheduler
from inference.service import InferenceService


def generate_dataset(request_count=200):
    config = BatchConfig(
        max_batch_size=4,
        max_batch_delay_ms=5
    )

    policy = AdaptivePolicy(
        min_batch_size=1,
        max_batch_size=8,
        min_delay_ms=1,
        max_delay_ms=20,
        target_p95_latency_ms=200
    )

    scheduler = AdaptiveScheduler(
        inference_service=InferenceService(),
        policy=policy,
        batch_config=config,
        max_queue_size=request_count + 20
    )

    scheduler.start()

    requests = [
        InferenceRequest([i % 10] * 10)
        for i in range(request_count)
    ]

    for request in requests:
        scheduler.submit(request)

    deadline = __import__("time").perf_counter() + 10

    while scheduler.metrics()["requests_completed"] < request_count:
        if __import__("time").perf_counter() >= deadline:
            break

        __import__("time").sleep(0.01)

    scheduler.stop()

    events = scheduler.telemetry.get_events()

    builder = WorkloadDatasetBuilder()
    rows = builder.build_rows(events)

    output_path = Path("ai/workload_predictor/workload_dataset.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if rows:
        with open(output_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(rows)

    print("=== GPUFlow-X Workload Dataset ===")
    print("Requests generated:", request_count)
    print("Telemetry events:", len(events))
    print("Dataset rows:", len(rows))
    print("Dataset path:", output_path)


if __name__ == "__main__":
    generate_dataset()
