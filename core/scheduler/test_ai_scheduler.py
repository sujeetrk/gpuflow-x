import time

from core.batching.batch_config import BatchConfig
from core.policy.ai_policy import AIPolicy
from core.scheduler.ai_scheduler import AIScheduler
from core.queue.request import InferenceRequest
from inference.service import InferenceService


def test_ai_scheduler_completes_requests():

    config = BatchConfig(
        max_batch_size=4,
        max_batch_delay_ms=5
    )

    policy = AIPolicy(
        target_p95_latency_ms=200
    )

    scheduler = AIScheduler(
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

    deadline = time.perf_counter() + 10

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


def test_ai_scheduler_records_ai_decision():

    scheduler = AIScheduler(
        inference_service=InferenceService(),
        policy=AIPolicy(),
        batch_config=BatchConfig(
            max_batch_size=4,
            max_batch_delay_ms=5
        ),
        max_queue_size=20
    )

    scheduler.start()

    requests = [
        InferenceRequest([i] * 10)
        for i in range(1, 5)
    ]

    for request in requests:
        scheduler.submit(request)

    deadline = time.perf_counter() + 10

    while scheduler.metrics()["requests_completed"] < 4:
        if time.perf_counter() >= deadline:
            break

        time.sleep(0.01)

    scheduler.stop()

    decision = scheduler.metrics()["last_ai_decision"]

    assert decision is not None
    assert "predicted_latency_ms" in decision
    assert "ai_reason" in decision
    assert decision["predicted_latency_ms"] >= 0
