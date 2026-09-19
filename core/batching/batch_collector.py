
import time
from typing import Optional

from core.batching.batch import InferenceBatch
from core.queue.request_queue import RequestQueue


class BatchCollector:
    """
    Collects requests while respecting maximum batch size,
    collection delay, and request deadlines.

    Even when the collection window expires, the collector
    drains requests that are already available in the queue.
    """

    def __init__(
        self,
        queue: RequestQueue,
        max_batch_size: int = 8,
        max_batch_delay_ms: float = 10.0,
        deadline_guard_ms: float = 1.0,
    ):
        if max_batch_size < 1:
            raise ValueError(
                "max_batch_size must be at least 1."
            )

        if max_batch_delay_ms < 0:
            raise ValueError(
                "max_batch_delay_ms cannot be negative."
            )

        if deadline_guard_ms < 0:
            raise ValueError(
                "deadline_guard_ms cannot be negative."
            )

        self.queue = queue
        self.max_batch_size = max_batch_size
        self.max_batch_delay_ms = max_batch_delay_ms
        self.deadline_guard_ms = deadline_guard_ms

    def _collection_deadline(
        self,
        batch,
        default_deadline,
    ):
        """
        Stop waiting before the earliest deadline in the
        current batch, using the configured safety guard.
        """

        deadlines = [
            request.deadline_time
            for request in batch.requests
            if request.deadline_time is not None
        ]

        if not deadlines:
            return default_deadline

        guarded_deadline = (
            min(deadlines)
            - self.deadline_guard_ms / 1000
        )

        return min(
            default_deadline,
            guarded_deadline,
        )

    def collect(self) -> Optional[InferenceBatch]:
        """
        Collect available requests up to max_batch_size.

        If the collection deadline has passed, immediately
        drain already-queued requests without waiting.
        """

        first_request = self.queue.dequeue(timeout=0.1)

        if first_request is None:
            return None

        batch = InferenceBatch(
            created_time=time.perf_counter()
        )

        batch.add(first_request)

        collection_deadline = (
            time.perf_counter()
            + self.max_batch_delay_ms / 1000
        )

        while not batch.is_full(self.max_batch_size):

            safe_deadline = self._collection_deadline(
                batch,
                collection_deadline,
            )

            remaining_time = (
                safe_deadline - time.perf_counter()
            )

            # If the deadline has passed, do not wait.
            # Still collect requests already in the queue.
            if remaining_time <= 0:
                request = self.queue.dequeue(timeout=0)
            else:
                request = self.queue.dequeue(
                    timeout=remaining_time
                )

            if request is None:
                break

            batch.add(request)

        return batch
