import threading
import time

from core.batching.batch_collector import BatchCollector
from core.batching.batch_config import BatchConfig
from core.policy.ai_policy import AIPolicy
from core.policy.state import WorkloadState
from core.queue.request import InferenceRequest
from core.queue.request_queue import RequestQueue
from core.telemetry.collector import TelemetryCollector
from core.telemetry.event import TelemetryEvent
from inference.service import InferenceService


class AIScheduler:

    def __init__(
        self,
        inference_service: InferenceService,
        policy: AIPolicy | None = None,
        batch_config: BatchConfig | None = None,
        max_queue_size: int = 100,
    ):
        self.inference_service = inference_service
        self.policy = policy or AIPolicy()
        self.queue = RequestQueue(max_size=max_queue_size)

        self.batch_config = batch_config or BatchConfig()

        self.collector = BatchCollector(
            queue=self.queue,
            max_batch_size=self.batch_config.max_batch_size,
            max_batch_delay_ms=self.batch_config.max_batch_delay_ms,
        )

        self.telemetry = TelemetryCollector()

        self._running = False
        self._worker = None

        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._batches_executed = 0

        self._metrics_lock = threading.Lock()
        self._last_decision = None

    def start(self):
        if self._running:
            return

        self._running = True

        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )

        self._worker.start()

    def stop(self):
        self._running = False

        if self._worker is not None:
            self._worker.join(timeout=5)

    def submit(self, request: InferenceRequest):
        self.queue.enqueue(request)

        with self._metrics_lock:
            self._submitted += 1

        return request.request_id

    def _build_state(self):
        events = self.telemetry.get_events()

        completed = [
            event
            for event in events
            if event.status == "completed"
        ]

        if completed:

            latencies = [
                event.total_latency_ms
                for event in completed
                if event.total_latency_ms is not None
            ]

            queue_times = [
                event.queue_time_ms
                for event in completed
                if event.queue_time_ms is not None
            ]

            inference_times = [
                event.inference_time_ms
                for event in completed
                if event.inference_time_ms is not None
            ]

            average_latency = (
                sum(latencies) / len(latencies)
                if latencies
                else 0.0
            )

            average_queue_time = (
                sum(queue_times) / len(queue_times)
                if queue_times
                else 0.0
            )

            average_inference_time = (
                sum(inference_times) / len(inference_times)
                if inference_times
                else 0.0
            )

            sorted_latencies = sorted(latencies)

            p95_index = max(
                0,
                int(len(sorted_latencies) * 0.95) - 1,
            )

            p95_latency = (
                sorted_latencies[p95_index]
                if sorted_latencies
                else 0.0
            )

            arrival_times = [
                event.arrival_time
                for event in completed
            ]

            elapsed = max(
                time.perf_counter() - min(arrival_times),
                0.001,
            )

            arrival_rate = len(completed) / elapsed

        else:
            average_latency = 0.0
            average_queue_time = 0.0
            average_inference_time = 0.0
            p95_latency = 0.0
            arrival_rate = 0.0

        return WorkloadState(
            queue_length=self.queue.size(),
            arrival_rate_rps=arrival_rate,
            gpu_utilization=0.0,
            vram_utilization=0.0,
            average_latency_ms=average_latency,
            p95_latency_ms=p95_latency,
            current_batch_size=self.batch_config.max_batch_size,
            current_batch_delay_ms=self.batch_config.max_batch_delay_ms,
            average_queue_time_ms=average_queue_time,
            average_inference_time_ms=average_inference_time,
        )

    def _apply_policy(self):
        state = self._build_state()

        decision = self.policy.decide(state)

        self.batch_config.max_batch_size = decision["batch_size"]
        self.batch_config.max_batch_delay_ms = (
            decision["batch_delay_ms"]
        )

        self.collector.max_batch_size = decision["batch_size"]
        self.collector.max_batch_delay_ms = (
            decision["batch_delay_ms"]
        )

        self._last_decision = decision

        return decision

    def _record_telemetry(self, request, batch_size):
        event = TelemetryEvent(
            request_id=request.request_id,
            arrival_time=request.arrival_time,
            enqueue_time=request.enqueue_time,
            start_time=request.start_time,
            end_time=request.end_time,
            queue_time_ms=request.queue_time_ms,
            inference_time_ms=request.inference_time_ms,
            total_latency_ms=request.total_latency_ms,
            batch_size=batch_size,
            status=request.status,
            device="cpu",
        )

        self.telemetry.record(event)

    def _worker_loop(self):

        while self._running:

            # First wait for actual work.
            batch = self.collector.collect()

            if batch is None:
                continue

            # Make one AI decision for this batch.
            self._apply_policy()

            try:

                for request in batch.requests:
                    request.mark_started()

                self.inference_service.infer_batch_object(batch)

                for request in batch.requests:

                    request.mark_completed()

                    self._record_telemetry(
                        request,
                        batch.size,
                    )

                with self._metrics_lock:
                    self._completed += batch.size
                    self._batches_executed += 1

            except Exception:

                for request in batch.requests:

                    request.mark_failed()

                    self._record_telemetry(
                        request,
                        batch.size,
                    )

                with self._metrics_lock:
                    self._failed += batch.size

            finally:

                for _ in batch.requests:
                    self.queue.task_done()

    def metrics(self):

        with self._metrics_lock:

            return {
                "queue_length": self.queue.size(),
                "requests_submitted": self._submitted,
                "requests_completed": self._completed,
                "requests_failed": self._failed,
                "batches_executed": self._batches_executed,
                "max_batch_size": self.batch_config.max_batch_size,
                "max_batch_delay_ms": (
                    self.batch_config.max_batch_delay_ms
                ),
                "telemetry_events": self.telemetry.count(),
                "last_ai_decision": self._last_decision,
            }
