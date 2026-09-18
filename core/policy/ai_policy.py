from ai.workload_predictor.predictor import WorkloadPredictor
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
    ):
        self.predictor = predictor or WorkloadPredictor()

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.target_latency_ms = target_p95_latency_ms

    def decide(self, state: WorkloadState) -> dict:

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

        # High predicted latency + low queue:
        # prioritize latency.
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

        # High predicted latency + high queue:
        # preserve throughput instead of collapsing to batch size 1.
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

        # Safe prediction + high queue:
        # increase batching to process queued work efficiently.
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

        # Safe prediction + low queue:
        # maintain the current configuration.
        else:

            batch_size = state.current_batch_size
            batch_delay_ms = state.current_batch_delay_ms

            reason = "predicted_latency_safe"

        return {
            "batch_size": batch_size,
            "batch_delay_ms": batch_delay_ms,
            "ai_reason": reason,
            "predicted_latency_ms": predicted_latency,
        }
