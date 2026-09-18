import time

from core.queue.request import InferenceRequest
from core.scheduler.batch_scheduler import DynamicBatchScheduler
from core.batching.batch_config import BatchConfig
from inference.service import InferenceService


def test_dynamic_batching_completes_requests():

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

    for request in requests:
        scheduler.submit(request)

    # Wait until all requests complete.
    deadline = time.perf_counter() + 5

    while scheduler.metrics()["requests_completed"] < len(requests):
        if time.perf_counter() >= deadline:
            break
        time.sleep(0.01)

    scheduler.stop()

    metrics = scheduler.metrics()

    assert metrics["requests_submitted"] == 8
    assert metrics["requests_completed"] == 8
    assert metrics["requests_failed"] == 0
    assert metrics["batches_executed"] >= 1
    assert metrics["telemetry_events"] == 8
    assert all(request.status == "completed" for request in requests)
