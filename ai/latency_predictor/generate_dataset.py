import csv
import time
from pathlib import Path

from core.batching.batch_config import BatchConfig
from core.queue.request import InferenceRequest
from core.scheduler.batch_scheduler import DynamicBatchScheduler
from inference.service import InferenceService


OUTPUT_PATH = Path(
    "ai/latency_predictor/latency_dataset.csv"
)


def run_experiment(
    batch_size,
    batch_delay_ms,
    request_count=32,
):
    scheduler = DynamicBatchScheduler(
        inference_service=InferenceService(),
        batch_config=BatchConfig(
            max_batch_size=batch_size,
            max_batch_delay_ms=batch_delay_ms,
        ),
        max_queue_size=request_count + 10,
    )

    scheduler.start()

    # Warm up the model before measuring.
    warmup_request = InferenceRequest([0] * 10)
    scheduler.submit(warmup_request)

    time.sleep(0.2)

    requests = [
        InferenceRequest(
            [(i + j) % 10 for j in range(10)]
        )
        for i in range(request_count)
    ]

    start = time.perf_counter()

    for request in requests:
        scheduler.submit(request)

    deadline = time.perf_counter() + 10

    while scheduler.metrics()["requests_completed"] < request_count:
        if time.perf_counter() >= deadline:
            break

        time.sleep(0.01)

    runtime_ms = (
        time.perf_counter() - start
    ) * 1000

    scheduler.stop()

    events = scheduler.telemetry.get_events()

    completed = [
        event
        for event in events
        if event.status == "completed"
        and event.request_id != warmup_request.request_id
    ]

    if not completed:
        return None

    latencies = [
        event.total_latency_ms
        for event in completed
        if event.total_latency_ms is not None
    ]

    queue_times = [
        event.queue_time_ms
        for event in completed
        if event.queue_time_ms is not None
    ]

    inference_times = [
        event.inference_time_ms
        for event in completed
        if event.inference_time_ms is not None
    ]

    sorted_latencies = sorted(latencies)

    p95_index = max(
        0,
        int(len(sorted_latencies) * 0.95) - 1,
    )

    return {
        "configured_batch_size": batch_size,
        "batch_delay_ms": batch_delay_ms,
        "request_count": request_count,
        "completed_requests": len(completed),
        "average_latency_ms": (
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        ),
        "p95_latency_ms": (
            sorted_latencies[p95_index]
            if sorted_latencies
            else 0.0
        ),
        "average_queue_time_ms": (
            sum(queue_times) / len(queue_times)
            if queue_times
            else 0.0
        ),
        "average_inference_time_ms": (
            sum(inference_times) / len(inference_times)
            if inference_times
            else 0.0
        ),
        "runtime_ms": runtime_ms,
    }


def generate_dataset():

    rows = []

    batch_sizes = [1, 2, 4, 8]

    batch_delays = [
        0,
        1,
        2,
        5,
        10,
        20,
        30,
        50,
    ]

    repetitions = 5

    total_experiments = (
        len(batch_sizes)
        * len(batch_delays)
        * repetitions
    )

    experiment_number = 0

    for batch_size in batch_sizes:

        for batch_delay_ms in batch_delays:

            for repetition in range(repetitions):

                experiment_number += 1

                print(
                    f"[{experiment_number}/{total_experiments}] "
                    f"batch_size={batch_size} "
                    f"delay_ms={batch_delay_ms} "
                    f"repeat={repetition + 1}"
                )

                result = run_experiment(
                    batch_size=batch_size,
                    batch_delay_ms=batch_delay_ms,
                )

                if result is not None:
                    rows.append(result)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "configured_batch_size",
                "batch_delay_ms",
                "request_count",
                "completed_requests",
                "average_latency_ms",
                "p95_latency_ms",
                "average_queue_time_ms",
                "average_inference_time_ms",
                "runtime_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=== GPUFlow-X Latency Dataset ===")
    print("Experiments completed:", len(rows))
    print("Expected experiments:", total_experiments)
    print("Dataset path:", OUTPUT_PATH)


if __name__ == "__main__":
    generate_dataset()
