import threading
import time

from core.queue.request import InferenceRequest
from core.scheduler.batch_scheduler import DynamicBatchScheduler
from core.batching.batch_config import BatchConfig
from inference.service import InferenceService


def main():

    config = BatchConfig(
        max_batch_size=8,
        max_batch_delay_ms=10
    )

    scheduler = DynamicBatchScheduler(
        inference_service=InferenceService(),
        batch_config=config
    )

    scheduler.start()

    requests = [
        InferenceRequest([i] * 10)
        for i in range(1, 9)
    ]

    threads = []

    start_time = time.perf_counter()

    for request in requests:

        thread = threading.Thread(
            target=scheduler.submit,
            args=(request,)
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Allow the worker to finish.
    time.sleep(1)

    end_time = time.perf_counter()

    scheduler.stop()

    metrics = scheduler.metrics()

    print("\n===== GPUFlow-X Dynamic Batching Benchmark =====")

    print("Requests submitted:", metrics["requests_submitted"])
    print("Requests completed:", metrics["requests_completed"])
    print("Requests failed:", metrics["requests_failed"])
    print("Batches executed:", metrics["batches_executed"])
    print("Maximum batch size:", metrics["max_batch_size"])
    print("Maximum batch delay (ms):", metrics["max_batch_delay_ms"])
    print(
        "Benchmark runtime (ms):",
        (end_time - start_time) * 1000
    )

    print(
        "Request statuses:",
        [request.status for request in requests]
    )


if __name__ == "__main__":
    main()
