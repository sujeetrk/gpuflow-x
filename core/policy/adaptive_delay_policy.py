from core.policy.policy import SchedulerPolicy
from core.policy.state import WorkloadState


class AdaptiveDelayPolicy(SchedulerPolicy):

    def __init__(
        self,
        min_delay_ms: float = 1.0,
        max_delay_ms: float = 20.0,
        target_p95_latency_ms: float = 200.0,
    ):
        if min_delay_ms < 0:
            raise ValueError("Minimum delay cannot be negative.")

        if max_delay_ms < min_delay_ms:
            raise ValueError(
                "Maximum delay must be >= minimum delay."
            )

        if target_p95_latency_ms <= 0:
            raise ValueError(
                "Target P95 latency must be greater than 0."
            )

        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.target_p95_latency_ms = target_p95_latency_ms

    def decide(self, state: WorkloadState) -> dict:

        current_delay = state.current_batch_delay_ms

        # Reduce waiting time when latency is too high.
        if state.p95_latency_ms > self.target_p95_latency_ms:
            new_delay = max(
                self.min_delay_ms,
                current_delay / 2
            )

            reason = "high_latency"

        # Allow more batching time when queued work is increasing.
        elif state.queue_length >= state.current_batch_size * 2:
            new_delay = min(
                self.max_delay_ms,
                current_delay + 1
            )

            reason = "high_queue"

        # Keep the current delay under stable conditions.
        else:
            new_delay = current_delay
            reason = "stable"

        return {
            "batch_delay_ms": new_delay,
            "reason": reason,
        }
