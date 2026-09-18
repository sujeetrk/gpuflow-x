import threading
from typing import Optional

from core.batching.batch_collector import BatchCollector
from core.batching.batch_config import BatchConfig
from core.queue.request import InferenceRequest
from core.queue.request_queue import RequestQueue
from inference.service import InferenceService


class DynamicBatchScheduler:
    """
    Dynamic batching scheduler.

    Collects requests from the FIFO queue and executes
    them together when either the maximum batch size
    or maximum batch delay is reached.
    """

    def __init__(
        self,
        inference_service: InferenceService,
        batch_config: Optional[BatchConfig] = None,
        max_queue_size: int = 100
    ):
        self.inference_service = inference_service

        self.queue = RequestQueue(
            max_size=max_queue_size
        )

        self.batch_config = (
            batch_config
            if batch_config is not None
            else BatchConfig()
        )

        self.collector = BatchCollector(
            queue=self.queue,
            max_batch_size=self.batch_config.max_batch_size,
            max_batch_delay_ms=self.batch_config.max_batch_delay_ms
        )

        self._running = False
        self._worker: Optional[threading.Thread] = None

        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._batches_executed = 0

        self._metrics_lock = threading.Lock()

    def start(self) -> None:
        """Start the dynamic batch worker."""

        if self._running:
            return

        self._running = True

        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self._worker.start()

    def stop(self) -> None:
        """Stop the dynamic batch worker."""

        self._running = False

        if self._worker is not None:
            self._worker.join(timeout=2)

    def submit(
        self,
        request: InferenceRequest
    ) -> str:
        """Submit a request to the queue."""

        self.queue.enqueue(request)

        with self._metrics_lock:
            self._submitted += 1

        return request.request_id

    def _worker_loop(self) -> None:
        """Collect and execute dynamic batches."""

        while self._running:

            batch = self.collector.collect()

            if batch is None:
                continue

            try:
                # Mark all requests as processing.
                for request in batch.requests:
                    request.mark_started()

                # Execute the complete batch.
                result = self.inference_service.infer_batch_object(
                    batch
                )

                # Calculate timing for each request.
                for request in batch.requests:
                    request.mark_completed()

                with self._metrics_lock:
                    self._completed += batch.size
                    self._batches_executed += 1

            except Exception:
                for request in batch.requests:
                    request.mark_failed()

                with self._metrics_lock:
                    self._failed += batch.size

            finally:
                for _ in batch.requests:
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
                "requests_failed": self._failed,
                "batches_executed": self._batches_executed,
                "max_batch_size": (
                    self.batch_config.max_batch_size
                ),
                "max_batch_delay_ms": (
                    self.batch_config.max_batch_delay_ms
                )
            }
