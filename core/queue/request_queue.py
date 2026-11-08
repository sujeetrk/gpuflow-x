import time
import threading
from queue import PriorityQueue, Empty, Full
from typing import Optional

from core.queue.request import InferenceRequest


class RequestQueue:
    """
    Thread-safe priority queue for GPUFlow-X.

    Scheduling order:
    1. Requests with deadlines before requests without deadlines.
    2. Earliest deadline first.
    3. Higher priority first when deadlines are equal.
    4. Earlier arrival first as a tie-breaker.

    Enqueue is non-blocking by default. A full queue raises
    queue.Full without leaving the request marked as queued.
    """

    def __init__(self, max_size: int = 100):
        if not isinstance(max_size, int) or isinstance(max_size, bool):
            raise TypeError("max_size must be an integer.")

        if max_size < 1:
            raise ValueError("max_size must be at least 1.")

        self._queue = PriorityQueue(maxsize=max_size)
        self._sequence = 0
        self._lock = threading.Lock()

    def _priority_key(self, request):
        deadline = getattr(request, "deadline_time", None)
        priority = getattr(request, "priority", 0)
        arrival = getattr(
            request,
            "arrival_time",
            time.perf_counter()
        )

        if deadline is not None:
            return (0, deadline, -priority, arrival)

        return (1, 0, -priority, arrival)

    def enqueue(
        self,
        request: InferenceRequest,
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Add a request according to its scheduling urgency.

        By default, submission never waits for queue capacity.
        Raises queue.Full when the queue is full.

        Set block=True to explicitly allow waiting for capacity.
        """

        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative.")

        if not block and timeout is not None:
            raise ValueError(
                "timeout cannot be specified when block=False."
            )

        with self._lock:
            sequence = self._sequence
            key = self._priority_key(request)

            previous_status = request.status
            previous_enqueue_time = request.enqueue_time

            request.mark_enqueued()

            try:
                item = (key, sequence, request)

                if block:
                    self._queue.put(
                        item,
                        block=True,
                        timeout=timeout
                    )
                else:
                    self._queue.put_nowait(item)

                self._sequence += 1

            except Full:
                request.status = previous_status
                request.enqueue_time = previous_enqueue_time
                raise

    def dequeue(
        self,
        timeout: Optional[float] = None
    ) -> Optional[InferenceRequest]:
        """Remove and return the most urgent queued request."""

        try:
            _, _, request = self._queue.get(timeout=timeout)
            return request
        except Empty:
            return None

    def task_done(self) -> None:
        """Mark a dequeued request as processed."""
        self._queue.task_done()

    def peek(self) -> Optional[InferenceRequest]:
        """Return the most urgent queued request without removing it."""

        with self._queue.mutex:
            if not self._queue.queue:
                return None

            return self._queue.queue[0][2]

    def size(self) -> int:
        """Return the number of requests currently waiting."""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """Return whether the queue is empty."""
        return self._queue.empty()
