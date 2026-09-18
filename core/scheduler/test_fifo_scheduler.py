import time

from core.queue.request import InferenceRequest
from core.scheduler.fifo_scheduler import FIFOScheduler
from inference.service import InferenceService


def main():
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
        print(
            "Submitted:",
            request.request_id,
            "value:",
            request.values[0]
        )

    # Give the worker time to process the requests.
    time.sleep(1)

    scheduler.stop()

    print("\nFinal Results:")

    for request in requests:
        print(
            "Request:",
            request.request_id,
            "| value:",
            request.values[0],
            "| status:",
            request.status,
            "| queue_time_ms:",
            request.queue_time_ms,
            "| inference_time_ms:",
            request.inference_time_ms,
            "| total_latency_ms:",
            request.total_latency_ms
        )


if __name__ == "__main__":
    main()
