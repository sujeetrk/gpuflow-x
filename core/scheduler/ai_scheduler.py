
import time
import threading
from pathlib import Path

from core.batching.batch_collector import BatchCollector
from core.batching.batch_config import BatchConfig
from core.policy.ai_policy import AIPolicy
from core.policy.learning_agent import LearningAgent
from core.policy.policy_storage import PolicyStorage
from core.policy.self_learning import SchedulingAction
from core.queue.request_queue import RequestQueue
from core.policy.state import WorkloadState
from core.telemetry.collector import TelemetryCollector
from core.telemetry.event import TelemetryEvent
from inference.service import InferenceService


class AIScheduler:
    """
    AI-driven, priority- and SLA-aware inference scheduler.

    Combines AI latency prediction, self-learning decisions,
    deadline-aware batching, telemetry, and policy persistence.
    """

    def __init__(
        self,
        inference_service,
        policy=None,
        batch_config=None,
        max_queue_size=100,
        learning_enabled=True,
        policy_path="gpu_flow_policy.json",
        deadline_guard_ms=1.0,
    ):
        if deadline_guard_ms < 0:
            raise ValueError(
                "deadline_guard_ms cannot be negative."
            )

        self.inference_service = (
            inference_service
            if inference_service is not None
            else InferenceService()
        )

        self.policy = policy if policy is not None else AIPolicy()

        self.batch_config = (
            batch_config
            if batch_config is not None
            else BatchConfig()
        )

        # Keep the caller's configured ceiling immutable.  The live batch
        # size is a policy output and must not become the next iteration's
        # permanent ceiling after one conservative decision.
        self._max_batch_size = self.batch_config.max_batch_size
        self._max_batch_delay_ms = self.batch_config.max_batch_delay_ms

        self.queue = RequestQueue(max_size=max_queue_size)

        self.collector = BatchCollector(
            queue=self.queue,
            max_batch_size=self.batch_config.max_batch_size,
            max_batch_delay_ms=self.batch_config.max_batch_delay_ms,
            deadline_guard_ms=deadline_guard_ms,
        )

        self.learning_enabled = learning_enabled
        self.learning_agent = LearningAgent()
        self.deadline_guard_ms = deadline_guard_ms

        self.policy_storage = PolicyStorage(policy_path)

        self.telemetry = TelemetryCollector()

        self._running = False
        self._worker = None

        self._metrics_lock = threading.Lock()

        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._batches_executed = 0
        self._deadline_misses = 0

        self._last_decision = None
        self._last_reward = None

        # Restore previously learned policy, if available.
        try:
            self.policy_storage.load(self.learning_agent.policy)
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            pass

    def start(self):
        """Start the background scheduling worker."""
        if self._running:
            return

        self._running = True

        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self._worker.start()

    def stop(self):
        """Stop the worker after queued requests are handled."""
        self._running = False

        if self._worker is not None:
            self._worker.join(timeout=10)

    def submit(self, request):
        """Submit an inference request to the priority queue."""
        self.queue.enqueue(request)

        with self._metrics_lock:
            self._submitted += 1

    def _build_state(self):
        """Build workload state from recent completed telemetry."""
        events = [
            event
            for event in self.telemetry.get_events()
            if event.status == "completed"
        ]

        if events:
            latencies = [
                event.total_latency_ms
                for event in events
                if event.total_latency_ms is not None
            ]

            queue_times = [
                event.queue_time_ms
                for event in events
                if event.queue_time_ms is not None
            ]

            inference_times = [
                event.inference_time_ms
                for event in events
                if event.inference_time_ms is not None
            ]

            average_latency = (
                sum(latencies) / len(latencies)
                if latencies else 0.0
            )

            average_queue_time = (
                sum(queue_times) / len(queue_times)
                if queue_times else 0.0
            )

            average_inference_time = (
                sum(inference_times) / len(inference_times)
                if inference_times else 0.0
            )

            sorted_latencies = sorted(latencies)

            p95_index = max(
                0,
                int(len(sorted_latencies) * 0.95) - 1,
            )

            p95_latency = (
                sorted_latencies[p95_index]
                if sorted_latencies else 0.0
            )

            arrival_times = [
                event.arrival_time
                for event in events
            ]

            elapsed = max(
                time.perf_counter() - min(arrival_times),
                0.001,
            )

            arrival_rate = len(events) / elapsed

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
            current_batch_size=self._max_batch_size,
            current_batch_delay_ms=self._max_batch_delay_ms,
            average_queue_time_ms=average_queue_time,
            average_inference_time_ms=average_inference_time,
        )

    def _deadline_pressure(self):
        """Return earliest queued deadline's remaining milliseconds."""
        now = time.perf_counter()

        with self.queue._queue.mutex:
            queued_items = list(self.queue._queue.queue)

        deadlines = [
            request.deadline_time - now
            for _, _, request in queued_items
            if request.deadline_time is not None
        ]

        if not deadlines:
            return None

        return min(deadlines) * 1000

    def _apply_policy(self):
        """Choose a safe batch policy while avoiding costly model calls on backlogs."""
        state = self._build_state()
        deadline_pressure_ms = self._deadline_pressure()
        cap = self._max_batch_size

        # Handle deadline states before invoking the sklearn predictor.
        # Once a deadline has expired, singleton batches only prolong the
        # backlog; drain immediately at the configured batch ceiling.
        if deadline_pressure_ms is not None and deadline_pressure_ms <= 0:
            decision = {
                "batch_size": cap,
                "batch_delay_ms": 0.0,
                "ai_reason": "expired_deadline_drain",
                "predicted_latency_ms": 0.0,
                "policy_bypassed_for_backlog": state.queue_length >= cap,
                "policy_bypassed_for_deadline": True,
            }
        elif (
            deadline_pressure_ms is not None
            and deadline_pressure_ms <= self.deadline_guard_ms
        ):
            # An imminent, still-live deadline gets no collection delay.
            # Preserve the single-request urgency behavior covered by tests.
            decision = {
                "batch_size": 1,
                "batch_delay_ms": 0.0,
                "ai_reason": "urgent_deadline",
                "predicted_latency_ms": 0.0,
                "policy_bypassed_for_backlog": False,
                "policy_bypassed_for_deadline": True,
            }
        elif state.queue_length >= cap:
            # The workload is already queued. Avoid one expensive model
            # prediction per batch and clear the backlog efficiently.
            decision = {
                "batch_size": cap,
                "batch_delay_ms": 0.0,
                "ai_reason": "backlog_throughput",
                "predicted_latency_ms": 0.0,
                "policy_bypassed_for_backlog": True,
                "policy_bypassed_for_deadline": False,
            }
        else:
            decision = dict(self.policy.decide(state))
            decision["policy_bypassed_for_backlog"] = False
            decision["policy_bypassed_for_deadline"] = False

        # Self-learning is useful for ordinary/light traffic, but must not
        # undo a deadline-drain or backlog-throughput safety decision.
        can_learn_action = (
            self.learning_enabled
            and not decision["policy_bypassed_for_backlog"]
            and decision["ai_reason"] != "expired_deadline_drain"
        )

        if can_learn_action:
            action = self.learning_agent.choose_action(
                queue_length=state.queue_length,
                predicted_latency_ms=decision["predicted_latency_ms"],
                current_batch_size=state.current_batch_size,
            )
            decision["batch_size"] = action.batch_size
            decision["batch_delay_ms"] = action.batch_delay_ms
            decision["learning_action"] = {
                "batch_size": action.batch_size,
                "batch_delay_ms": action.batch_delay_ms,
            }
        elif self.learning_enabled:
            decision["learning_skipped_for_safety_bypass"] = True

        decision["learning_enabled"] = self.learning_enabled
        decision["deadline_pressure_ms"] = deadline_pressure_ms
        decision["deadline_override"] = False

        # Apply the deadline guard after any learned action. A positive
        # imminent deadline is isolated; expired deadlines are drained.
        if deadline_pressure_ms is not None:
            if deadline_pressure_ms <= 0:
                decision["batch_size"] = cap
                decision["batch_delay_ms"] = 0.0
                decision["deadline_override"] = True
                decision["ai_reason"] = "expired_deadline_drain"
            elif deadline_pressure_ms <= self.deadline_guard_ms:
                decision["batch_size"] = 1
                decision["batch_delay_ms"] = 0.0
                decision["deadline_override"] = True
                decision["ai_reason"] = "urgent_deadline"
            else:
                safe_wait_ms = max(
                    0.0,
                    deadline_pressure_ms - self.deadline_guard_ms,
                )
                if safe_wait_ms < decision["batch_delay_ms"]:
                    decision["batch_delay_ms"] = safe_wait_ms
                    decision["deadline_override"] = True
                    decision["ai_reason"] = "deadline_constrained"

        # Preserve the proposal for diagnostics, but only train on it when
        # the final, safety-clamped action is exactly what was proposed.
        proposed_action = decision.get("learning_action")
        decision["proposed_learning_action"] = proposed_action

        decision["batch_size"] = max(1, min(decision["batch_size"], cap))
        decision["batch_delay_ms"] = max(0.0, decision["batch_delay_ms"])
        effective_action = {
            "batch_size": decision["batch_size"],
            "batch_delay_ms": decision["batch_delay_ms"],
        }
        decision["effective_action"] = effective_action

        if proposed_action is not None:
            if proposed_action == effective_action:
                # Store the executed action as the learner's attribution key.
                decision["learning_action"] = dict(effective_action)
            else:
                decision.pop("learning_action", None)
                decision["learning_skipped_for_action_override"] = True

        self.batch_config.max_batch_size = decision["batch_size"]
        self.batch_config.max_batch_delay_ms = decision["batch_delay_ms"]
        self.collector.max_batch_size = decision["batch_size"]
        self.collector.max_batch_delay_ms = decision["batch_delay_ms"]
        self._last_decision = decision

        return state, decision

    def _record_telemetry(self, request, batch_size):
        """Record request timing and deadline misses."""
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

        if request.deadline_missed:
            with self._metrics_lock:
                self._deadline_misses += 1

    def _learn_from_batch(
        self,
        state,
        decision,
        batch_runtime_ms,
        batch_requests,
    ):
        """Learn from telemetry belonging only to this batch."""
        if not self.learning_enabled:
            return None

        action_data = decision.get("learning_action")

        if action_data is None:
            return None

        action = SchedulingAction(
            batch_size=action_data["batch_size"],
            batch_delay_ms=action_data["batch_delay_ms"],
        )

        request_ids = {
            request.request_id
            for request in batch_requests
        }

        batch_events = [
            event
            for event in self.telemetry.get_events()
            if (
                event.request_id in request_ids
                and event.status == "completed"
            )
        ]

        if not batch_events:
            return None

        latencies = [
            event.total_latency_ms
            for event in batch_events
            if event.total_latency_ms is not None
        ]

        queue_times = [
            event.queue_time_ms
            for event in batch_events
            if event.queue_time_ms is not None
        ]

        average_latency = (
            sum(latencies) / len(latencies)
            if latencies else 0.0
        )

        average_queue_time = (
            sum(queue_times) / len(queue_times)
            if queue_times else 0.0
        )

        completed_ids = {
            event.request_id
            for event in batch_events
        }

        completed_requests = [
            request
            for request in batch_requests
            if request.request_id in completed_ids
        ]

        deadline_misses = sum(
            1
            for request in completed_requests
            if request.deadline_missed
        )

        throughput = len(completed_requests) / max(
            batch_runtime_ms / 1000.0,
            0.001,
        )

        reward = self.learning_agent.learn(
            queue_length=state.queue_length,
            predicted_latency_ms=decision["predicted_latency_ms"],
            current_batch_size=state.current_batch_size,
            action=action,
            average_latency_ms=average_latency,
            average_queue_time_ms=average_queue_time,
            throughput_requests_per_sec=throughput,
            deadline_misses=deadline_misses,
            request_count=len(completed_requests),
        )

        self._last_reward = reward

        try:
            self.policy_storage.save(self.learning_agent.policy)
        except (OSError, ValueError, TypeError):
            pass

        return reward

    def _worker_loop(self):
        """Collect and execute batches while isolating post-inference errors."""
        while self._running or not self.queue.is_empty():
            if self.queue.is_empty():
                time.sleep(0.01)
                continue

            state, decision = self._apply_policy()
            batch = self.collector.collect()
            if batch is None:
                continue

            batch_start = time.perf_counter()

            # Only request-starting and inference errors determine whether
            # inference itself failed. Telemetry/learning are best-effort and
            # must never reclassify already-completed requests as failed.
            try:
                for request in batch.requests:
                    request.mark_started()
                self.inference_service.infer_batch_object(batch)
            except Exception:
                for request in batch.requests:
                    if request.status not in ("failed", "completed"):
                        request.mark_failed()
                    try:
                        self._record_telemetry(request, batch.size)
                    except Exception:
                        pass

                with self._metrics_lock:
                    self._failed += sum(
                        1 for request in batch.requests
                        if request.status == "failed"
                    )
                continue

            batch_runtime_ms = (time.perf_counter() - batch_start) * 1000.0
            for request in batch.requests:
                request.mark_completed()

            with self._metrics_lock:
                self._completed += batch.size
                self._batches_executed += 1

            for request in batch.requests:
                try:
                    self._record_telemetry(request, batch.size)
                except Exception:
                    # A telemetry sink failure must not change request status.
                    continue

            try:
                self._learn_from_batch(
                    state=state,
                    decision=decision,
                    batch_runtime_ms=batch_runtime_ms,
                    batch_requests=batch.requests,
                )
            except Exception:
                # Learning/persistence is auxiliary to successful inference.
                pass

    def metrics(self):
        """Return scheduler metrics and most recent AI decision."""
        with self._metrics_lock:
            return {
                "requests_submitted": self._submitted,
                "requests_completed": self._completed,
                "requests_failed": self._failed,
                "batches_executed": self._batches_executed,
                "deadline_misses": self._deadline_misses,
                "telemetry_events": self.telemetry.count(),
                "last_ai_decision": self._last_decision,
                "last_reward": self._last_reward,
                "learning_statistics": (
                    self.learning_agent.statistics()
                ),
                "queue_length": self.queue.size(),
            }
