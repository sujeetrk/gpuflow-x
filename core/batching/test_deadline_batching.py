import time

from core.queue.request import InferenceRequest
from core.queue.request_queue import RequestQueue
from core.batching.batch_collector import BatchCollector


def test_batch_respects_max_batch_size():
    queue = RequestQueue()

    for i in range(5):
        queue.enqueue(
            InferenceRequest(values=[float(i)] * 10)
        )

    collector = BatchCollector(
        queue=queue,
        max_batch_size=3,
        max_batch_delay_ms=50,
    )

    batch = collector.collect()

    assert batch.size == 3


def test_batch_collects_available_requests():
    queue = RequestQueue()

    for i in range(3):
        queue.enqueue(
            InferenceRequest(values=[float(i)] * 10)
        )

    collector = BatchCollector(
        queue=queue,
        max_batch_size=8,
        max_batch_delay_ms=10,
    )

    batch = collector.collect()

    assert batch.size == 3


def test_deadline_limits_batch_collection_wait():
    queue = RequestQueue()

    request = InferenceRequest(
        values=[1.0] * 10,
        deadline_time=time.perf_counter() + 0.03,
    )

    queue.enqueue(request)

    collector = BatchCollector(
        queue=queue,
        max_batch_size=8,
        max_batch_delay_ms=200,
        deadline_guard_ms=5,
    )

    start = time.perf_counter()
    batch = collector.collect()
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert batch.size == 1
    assert elapsed_ms < 100


def test_no_deadline_uses_configured_batch_delay():
    queue = RequestQueue()

    queue.enqueue(
        InferenceRequest(values=[1.0] * 10)
    )

    collector = BatchCollector(
        queue=queue,
        max_batch_size=8,
        max_batch_delay_ms=30,
    )

    start = time.perf_counter()
    batch = collector.collect()
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert batch.size == 1
    assert elapsed_ms >= 20
    assert elapsed_ms < 150
