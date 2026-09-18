from core.policy.policy import SchedulerPolicy
from core.policy.state import WorkloadState


class AdaptiveBatchPolicy(SchedulerPolicy):

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 8,
        target_p95_latency_ms: float = 200.0,
    ):
        if min_batch_size < 1:
            raise ValueError("Minimum batch size must be at least 1.")

        if max_batch_size < min_batch_size:
            raise ValueError(
                "Maximum batch size must be >= minimum batch size."
            )

        if target_p95_latency_ms <= 0:
            raise ValueError(
                "Target P95 latency must be greater than 0."
            )

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_p95_latency_ms = target_p95_latency_ms

    def decide(self, state: WorkloadState) -> dict:

        current_size = state.current_batch_size

        # Protect latency when the system is experiencing high latency.
        if state.p95_latency_ms > self.target_p95_latency_ms:
            new_batch_size = max(
                self.min_batch_size,
                current_size // 2
            )

            reason = "high_latency"

        # Increase batching when there is enough queued work.
        elif state.queue_length >= current_size * 2:
            new_batch_size = min(
                self.max_batch_size,
                current_size + 1
            )

            reason = "high_queue"

        # Otherwise keep the current configuration.
        else:
            new_batch_size = current_size
            reason = "stable"

        return {
            "batch_size": new_batch_size,
            "reason": reason,
        }
