import time

from core.queue.request import InferenceRequest
from core.scheduler.fifo_scheduler import FIFOScheduler
from inference.service import InferenceService


def test_fifo_scheduler_completes_requests():

    service = InferenceService()
    scheduler = FIFOScheduler(service)

    scheduler.start()

    requests = [
        InferenceRequest([1] * 10),
        InferenceRequest([2] * 10),
        InferenceRequest([3] * 10),
    ]

    for request in requests:
        scheduler.submit(request)

    time.sleep(1)

    scheduler.stop()

    assert len(requests) == 3
    assert all(request.status == "completed" for request in requests)
    assert all(request.total_latency_ms >= 0 for request in requests)
