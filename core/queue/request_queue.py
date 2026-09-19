
import time
import threading
from queue import PriorityQueue, Empty
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
    """

    def __init__(self, max_size: int = 100):
        self._queue = PriorityQueue(maxsize=max_size)
        self._sequence = 0
        self._lock = threading.Lock()

    def _priority_key(self, request):
        deadline = getattr(request, "deadline_time", None)
        priority = getattr(request, "priority", 0)
        arrival = getattr(request, "arrival_time", time.perf_counter())

        if deadline is not None:
            return (0, deadline, -priority, arrival)

        return (1, 0, -priority, arrival)

    def enqueue(self, request: InferenceRequest) -> None:
        """Add a request according to its scheduling urgency."""
        request.mark_enqueued()

        with self._lock:
            sequence = self._sequence
            self._sequence += 1

        key = self._priority_key(request)

        self._queue.put((key, sequence, request))

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
        """Return the most urgent request without removing it."""
        with self._queue.mutex:
            if not self._queue.queue:
                return None

            return self._queue.queue[0][2]

    def size(self) -> int:
        return self._queue.qsize()

    def is_empty(self) -> bool:
        return self._queue.empty()
