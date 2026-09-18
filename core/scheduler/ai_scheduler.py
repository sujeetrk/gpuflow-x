import threading
import time

from core.batching.batch_collector import BatchCollector
from core.batching.batch_config import BatchConfig

from core.policy.ai_policy import AIPolicy
from core.policy.learning_agent import LearningAgent
from core.policy.policy_storage import PolicyStorage
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
        learning_enabled: bool = True,
        policy_path: str = "core/policy/learned_policy.json",
    ):

        self.inference_service = inference_service

        self.policy = policy or AIPolicy()

        self.learning_agent = LearningAgent()

        self.policy_storage = PolicyStorage(
            policy_path
        )

        if learning_enabled:
            self.policy_storage.load(
                self.learning_agent.policy
            )

        self.learning_enabled = learning_enabled

        self.queue = RequestQueue(
            max_size=max_queue_size
        )

        self.batch_config = (
            batch_config or BatchConfig()
        )

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
        self._last_reward = None

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

        if self.learning_enabled:

            self.policy_storage.save(
                self.learning_agent.policy
            )

    def submit(
        self,
        request: InferenceRequest,
    ):

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

            sorted_latencies = sorted(
                latencies
            )

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
                time.perf_counter()
                - min(arrival_times),
                0.001,
            )

            arrival_rate = (
                len(completed) / elapsed
            )

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
            current_batch_size=(
                self.batch_config.max_batch_size
            ),
            current_batch_delay_ms=(
                self.batch_config.max_batch_delay_ms
            ),
            average_queue_time_ms=(
                average_queue_time
            ),
            average_inference_time_ms=(
                average_inference_time
            ),
        )

    def _apply_policy(self):

        state = self._build_state()

        # Existing AI latency prediction policy.
        base_decision = self.policy.decide(
            state
        )

        predicted_latency = base_decision[
            "predicted_latency_ms"
        ]

        # Self-learning policy chooses the actual action.
        if self.learning_enabled:

            action = (
                self.learning_agent.choose_action(
                    queue_length=state.queue_length,
                    predicted_latency_ms=predicted_latency,
                    current_batch_size=(
                        state.current_batch_size
                    ),
                )
            )

            decision = dict(
                base_decision
            )

            decision["batch_size"] = (
                action.batch_size
            )

            decision["batch_delay_ms"] = (
                action.batch_delay_ms
            )

            decision["learning_enabled"] = True

            decision["learning_action"] = {
                "batch_size": action.batch_size,
                "batch_delay_ms": (
                    action.batch_delay_ms
                ),
            }

        else:

            decision = dict(
                base_decision
            )

            decision[
                "learning_enabled"
            ] = False

        self.batch_config.max_batch_size = (
            decision["batch_size"]
        )

        self.batch_config.max_batch_delay_ms = (
            decision["batch_delay_ms"]
        )

        self.collector.max_batch_size = (
            decision["batch_size"]
        )

        self.collector.max_batch_delay_ms = (
            decision["batch_delay_ms"]
        )

        self._last_decision = decision

        return state, decision

    def _record_telemetry(
        self,
        request,
        batch_size,
    ):

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

    def _learn_from_batch(
        self,
        state,
        decision,
        batch_runtime_ms,
    ):

        if not self.learning_enabled:
            return None

        learning_action_data = decision.get(
            "learning_action"
        )

        if learning_action_data is None:
            return None

        from core.policy.self_learning import (
            SchedulingAction,
        )

        action = SchedulingAction(
            batch_size=learning_action_data[
                "batch_size"
            ],
            batch_delay_ms=learning_action_data[
                "batch_delay_ms"
            ],
        )

        batch_events = self.telemetry.get_events()

        recent_events = [
            event
            for event in batch_events
            if event.status == "completed"
            and event.batch_size == action.batch_size
        ]

        if not recent_events:
            recent_events = [
                event
                for event in batch_events
                if event.status == "completed"
            ]

        if not recent_events:
            return None

        recent_latencies = [
            event.total_latency_ms
            for event in recent_events
            if event.total_latency_ms is not None
        ]

        recent_queue_times = [
            event.queue_time_ms
            for event in recent_events
            if event.queue_time_ms is not None
        ]

        average_latency = (
            sum(recent_latencies)
            / len(recent_latencies)
            if recent_latencies
            else 0.0
        )

        average_queue_time = (
            sum(recent_queue_times)
            / len(recent_queue_times)
            if recent_queue_times
            else 0.0
        )

        completed_count = len(
            recent_events
        )

        throughput = (
            completed_count
            / max(batch_runtime_ms / 1000.0, 0.001)
        )

        reward = self.learning_agent.learn(
            queue_length=state.queue_length,
            predicted_latency_ms=(
                decision[
                    "predicted_latency_ms"
                ]
            ),
            current_batch_size=(
                state.current_batch_size
            ),
            action=action,
            average_latency_ms=average_latency,
            average_queue_time_ms=average_queue_time,
            throughput_requests_per_sec=throughput,
        )

        self._last_reward = reward

        self.policy_storage.save(
            self.learning_agent.policy
        )

        return reward

    def _worker_loop(self):

        while self._running:

            batch = self.collector.collect()

            if batch is None:
                continue

            state, decision = (
                self._apply_policy()
            )

            batch_start = time.perf_counter()

            try:

                for request in batch.requests:
                    request.mark_started()

                self.inference_service.infer_batch_object(
                    batch
                )

                batch_end = time.perf_counter()

                batch_runtime_ms = (
                    batch_end - batch_start
                ) * 1000

                for request in batch.requests:

                    request.mark_completed()

                    self._record_telemetry(
                        request,
                        batch.size,
                    )

                with self._metrics_lock:

                    self._completed += (
                        batch.size
                    )

                    self._batches_executed += 1

                self._learn_from_batch(
                    state=state,
                    decision=decision,
                    batch_runtime_ms=batch_runtime_ms,
                )

            except Exception:

                for request in batch.requests:

                    request.mark_failed()

                    self._record_telemetry(
                        request,
                        batch.size,
                    )

                with self._metrics_lock:

                    self._failed += (
                        batch.size
                    )

            finally:

                for _ in batch.requests:
                    self.queue.task_done()

    def metrics(self):

        with self._metrics_lock:

            learning_stats = (
                self.learning_agent.statistics()
                if self.learning_enabled
                else {}
            )

            return {
                "queue_length": self.queue.size(),
                "requests_submitted": (
                    self._submitted
                ),
                "requests_completed": (
                    self._completed
                ),
                "requests_failed": (
                    self._failed
                ),
                "batches_executed": (
                    self._batches_executed
                ),
                "max_batch_size": (
                    self.batch_config.max_batch_size
                ),
                "max_batch_delay_ms": (
                    self.batch_config.max_batch_delay_ms
                ),
                "telemetry_events": (
                    self.telemetry.count()
                ),
                "last_ai_decision": (
                    self._last_decision
                ),
                "last_reward": (
                    self._last_reward
                ),
                "learning": learning_stats,
            }
