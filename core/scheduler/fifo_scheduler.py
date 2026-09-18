import threading
from typing import Optional

from core.queue.request import InferenceRequest
from core.queue.request_queue import RequestQueue
from inference.service import InferenceService


class FIFOScheduler:
    """
    Baseline First-In-First-Out inference scheduler.
    """

    def __init__(
        self,
        inference_service: InferenceService,
        max_queue_size: int = 100
    ):
        self.queue = RequestQueue(max_queue_size)
        self.inference_service = inference_service

        self._running = False
        self._worker: Optional[threading.Thread] = None

        self._submitted = 0
        self._completed = 0
        self._failed = 0

        self._metrics_lock = threading.Lock()

    def start(self) -> None:
        """Start the scheduler worker."""

        if self._running:
            return

        self._running = True

        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self._worker.start()

    def stop(self) -> None:
        """Stop the scheduler worker."""

        self._running = False

        if self._worker is not None:
            self._worker.join(timeout=2)

    def submit(self, request: InferenceRequest) -> str:
        """Submit a request to the FIFO queue."""

        self.queue.enqueue(request)

        with self._metrics_lock:
            self._submitted += 1

        return request.request_id

    def _worker_loop(self) -> None:
        """Continuously process queued requests."""

        while self._running:

            request = self.queue.dequeue(timeout=0.1)

            if request is None:
                continue

            try:
                request.mark_started()

                result = self.inference_service.infer(
                    request.values,
                    request.arrival_time
                )

                request.end_time = (
                    request.start_time
                    + result["inference_time_ms"] / 1000
                )

                request.status = "completed"

                with self._metrics_lock:
                    self._completed += 1

            except Exception:
                request.mark_failed()

                with self._metrics_lock:
                    self._failed += 1

            finally:
                self.queue.task_done()

    def queue_size(self) -> int:
        """Return current queue size."""

        return self.queue.size()

    def metrics(self) -> dict:
        """Return scheduler metrics."""

        with self._metrics_lock:
            return {
                "queue_length": self.queue.size(),
                "requests_submitted": self._submitted,
                "requests_completed": self._completed,
                "requests_failed": self._failed
            }
