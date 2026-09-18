from core.policy.adaptive_batch_policy import AdaptiveBatchPolicy
from core.policy.adaptive_delay_policy import AdaptiveDelayPolicy
from core.policy.state import WorkloadState


class AdaptivePolicy:
    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 8,
        min_delay_ms: float = 1.0,
        max_delay_ms: float = 20.0,
        target_p95_latency_ms: float = 200.0,
    ):
        self.batch_policy = AdaptiveBatchPolicy(
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            target_p95_latency_ms=target_p95_latency_ms,
        )

        self.delay_policy = AdaptiveDelayPolicy(
            min_delay_ms=min_delay_ms,
            max_delay_ms=max_delay_ms,
            target_p95_latency_ms=target_p95_latency_ms,
        )

    def decide(self, state: WorkloadState) -> dict:
        batch_decision = self.batch_policy.decide(state)
        delay_decision = self.delay_policy.decide(state)

        return {
            "batch_size": batch_decision["batch_size"],
            "batch_reason": batch_decision["reason"],
            "batch_delay_ms": delay_decision["batch_delay_ms"],
            "delay_reason": delay_decision["reason"],
        }
