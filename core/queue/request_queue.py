from queue import Queue, Empty
from typing import Optional

from core.queue.request import InferenceRequest


class RequestQueue:
    """
    Thread-safe FIFO queue for GPUFlow-X inference requests.
    """

    def __init__(self, max_size: int = 100):
        self._queue = Queue(maxsize=max_size)

    def enqueue(self, request: InferenceRequest) -> None:
        """
        Add a request to the end of the queue.
        """
        request.mark_enqueued()
        self._queue.put(request)

    def dequeue(self, timeout: Optional[float] = None) -> Optional[InferenceRequest]:
        """
        Remove and return the oldest request.
        """
        try:
            return self._queue.get(
                timeout=timeout
            )
        except Empty:
            return None

    def task_done(self) -> None:
        """
        Mark the current request as completed by the worker.
        """
        self._queue.task_done()

    def peek(self) -> Optional[InferenceRequest]:
        """
        Return the oldest request without removing it.
        """
        try:
            request = self._queue.get_nowait()
            self._queue.put(request)
            return request
        except Empty:
            return None

    def size(self) -> int:
        """
        Return the current queue size.
        """
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """
        Check whether the queue is empty.
        """
        return self._queue.empty()
