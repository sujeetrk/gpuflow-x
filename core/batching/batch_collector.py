import time
from typing import Optional

from core.batching.batch import InferenceBatch
from core.queue.request_queue import RequestQueue


class BatchCollector:
    """
    Collects requests from the queue and creates
    batches using size and time constraints.
    """

    def __init__(
        self,
        queue: RequestQueue,
        max_batch_size: int = 8,
        max_batch_delay_ms: float = 10.0
    ):
        self.queue = queue
        self.max_batch_size = max_batch_size
        self.max_batch_delay_ms = max_batch_delay_ms

    def collect(self) -> Optional[InferenceBatch]:
        """
        Collect requests until either:

        1. Maximum batch size is reached, or
        2. Maximum batch delay is reached.
        """

        first_request = self.queue.dequeue(timeout=0.1)

        if first_request is None:
            return None

        batch = InferenceBatch(
            created_time=time.perf_counter()
        )

        batch.add(first_request)

        deadline = (
            time.perf_counter()
            + self.max_batch_delay_ms / 1000
        )

        while not batch.is_full(self.max_batch_size):

            remaining_time = (
                deadline - time.perf_counter()
            )

            if remaining_time <= 0:
                break

            request = self.queue.dequeue(
                timeout=remaining_time
            )

            if request is None:
                break

            batch.add(request)

        return batch
