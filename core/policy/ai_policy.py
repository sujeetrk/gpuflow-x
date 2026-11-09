from ai.workload_predictor.predictor import WorkloadPredictor
from ai.workload_predictor.classifier import WorkloadClassifier
from core.policy.state import WorkloadState


class AIPolicy:
    def __init__(
        self,
        predictor=None,
        min_batch_size: int = 1,
        max_batch_size: int = 8,
        min_delay_ms: float = 1.0,
        max_delay_ms: float = 20.0,
        target_p95_latency_ms: float = 200.0,
        classifier=None,
    ):
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be at least 1.")

        if max_batch_size < min_batch_size:
            raise ValueError(
                "max_batch_size cannot be smaller than min_batch_size."
            )

        if min_delay_ms < 0 or max_delay_ms < min_delay_ms:
            raise ValueError("Invalid batch delay limits.")

        if target_p95_latency_ms <= 0:
            raise ValueError("target_p95_latency_ms must be positive.")

        self.predictor = predictor or WorkloadPredictor()
        self.classifier = classifier or WorkloadClassifier()

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.target_latency_ms = target_p95_latency_ms

    def _clamp_batch_size(self, value):
        return max(
            self.min_batch_size,
            min(self.max_batch_size, int(value)),
        )

    def _clamp_delay(self, value):
        return max(
            0.0,
            min(self.max_delay_ms, float(value)),
        )

    def decide(self, state: WorkloadState) -> dict:
        # Existing latency prediction remains unchanged.
        predicted_latency = self.predictor.predict(
            queue_time_ms=state.average_queue_time_ms,
            inference_time_ms=state.average_inference_time_ms,
            total_latency_ms=state.average_latency_ms,
            batch_size=state.current_batch_size,
        )

        queue_is_high = (
            state.queue_length >= state.current_batch_size * 2
        )

        prediction_is_high = (
            predicted_latency > self.target_latency_ms
        )

        # Existing latency-based scheduling policy.
        if prediction_is_high and not queue_is_high:
            batch_size = max(
                self.min_batch_size,
                state.current_batch_size // 2,
            )

            batch_delay_ms = max(
                self.min_delay_ms,
                state.current_batch_delay_ms / 2,
            )

            reason = "predicted_latency_high_low_queue"

        elif prediction_is_high and queue_is_high:
            batch_size = min(
                self.max_batch_size,
                state.current_batch_size + 1,
            )

            batch_delay_ms = min(
                self.max_delay_ms,
                state.current_batch_delay_ms + 1,
            )

            reason = "predicted_latency_high_queue_pressure"

        elif queue_is_high:
            batch_size = min(
                self.max_batch_size,
                state.current_batch_size + 1,
            )

            batch_delay_ms = min(
                self.max_delay_ms,
                state.current_batch_delay_ms + 1,
            )

            reason = "predicted_latency_safe_high_queue"

        else:
            batch_size = state.current_batch_size
            batch_delay_ms = state.current_batch_delay_ms

            reason = "predicted_latency_safe"

        # Classify the observed workload independently of latency prediction.
        classification = self.classifier.classify(
            state,
            target_latency_ms=self.target_latency_ms,
        )

        workload_class = classification["workload_class"]

        # Apply classification-aware adjustments only to normal policy
        # decisions. Scheduler deadline/backlog bypasses remain outside
        # this method and retain their existing priority.
        if workload_class == "latency_sensitive":
            batch_size = min(
                batch_size,
                max(
                    self.min_batch_size,
                    state.current_batch_size // 2,
                ),
            )

            if state.current_batch_delay_ms > 0:
                batch_delay_ms = min(
                    batch_delay_ms,
                    state.current_batch_delay_ms / 2,
                )

        elif workload_class == "throughput_oriented":
            batch_size = max(
                batch_size,
                min(
                    self.max_batch_size,
                    state.current_batch_size + 1,
                ),
            )

        elif workload_class == "compute_intensive":
            batch_size = max(
                batch_size,
                min(
                    self.max_batch_size,
                    state.current_batch_size + 1,
                ),
            )

        # Balanced workloads retain the latency policy's original action.
        batch_size = self._clamp_batch_size(batch_size)
        batch_delay_ms = self._clamp_delay(batch_delay_ms)

        return {
            "batch_size": batch_size,
            "batch_delay_ms": batch_delay_ms,
            "ai_reason": reason,
            "predicted_latency_ms": float(predicted_latency),
            "workload_class": workload_class,
            "classification_reason": classification[
                "classification_reason"
            ],
            "classification_signals": classification[
                "classification_signals"
            ],
        }
