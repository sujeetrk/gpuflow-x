import time

from core.batching.batch_config import BatchConfig
from core.policy.adaptive_policy import AdaptivePolicy
from core.scheduler.adaptive_scheduler import AdaptiveScheduler
from inference.service import InferenceService
from core.queue.request import InferenceRequest


def test_adaptive_scheduler_completes_requests():

    config = BatchConfig(
        max_batch_size=2,
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
        max_queue_size=20
    )

    scheduler.start()

    requests = [
        InferenceRequest([i] * 10)
        for i in range(1, 9)
    ]

    for request in requests:
        scheduler.submit(request)

    deadline = time.perf_counter() + 5

    while scheduler.metrics()["requests_completed"] < 8:
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
    assert all(
        request.status == "completed"
        for request in requests
    )


def test_adaptive_scheduler_changes_batch_configuration():

    config = BatchConfig(
        max_batch_size=2,
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
        max_queue_size=20
    )

    scheduler.start()

    requests = [
        InferenceRequest([i] * 10)
        for i in range(1, 9)
    ]

    for request in requests:
        scheduler.submit(request)

    deadline = time.perf_counter() + 5

    while scheduler.metrics()["requests_completed"] < 8:
        if time.perf_counter() >= deadline:
            break

        time.sleep(0.01)

    scheduler.stop()

    metrics = scheduler.metrics()

    assert metrics["requests_completed"] == 8
    assert metrics["max_batch_size"] > 2
