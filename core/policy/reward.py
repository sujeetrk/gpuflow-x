from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    target_latency_ms: float = 200.0
    latency_weight: float = 1.0
    queue_weight: float = 0.5
    throughput_weight: float = 0.2
    sla_violation_penalty: float = 2.0


class SchedulerReward:
    """
    Converts measured scheduler performance into a scalar reward.

    Higher reward means better scheduling behavior.
    Actual SLA misses receive an additional penalty.
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else RewardConfig()
        )

    def calculate(
        self,
        average_latency_ms: float,
        average_queue_time_ms: float,
        throughput_requests_per_sec: float,
        deadline_misses: int = 0,
        request_count: int = 0,
    ) -> float:

        target = max(
            self.config.target_latency_ms,
            0.001,
        )

        latency_score = max(
            0.0,
            1.0 - average_latency_ms / target,
        )

        queue_score = max(
            0.0,
            1.0 - average_queue_time_ms / target,
        )

        throughput_score = min(
            1.0,
            max(0.0, throughput_requests_per_sec) / 100.0,
        )

        reward = (
            self.config.latency_weight * latency_score
            + self.config.queue_weight * queue_score
            + self.config.throughput_weight * throughput_score
        )

        # Preserve the existing latency-target penalty.
        if average_latency_ms > target:
            reward -= self.config.sla_violation_penalty

        # Penalize the fraction of requests that missed their SLA.
        if request_count > 0 and deadline_misses > 0:
            miss_ratio = min(
                1.0,
                deadline_misses / request_count,
            )

            reward -= (
                self.config.sla_violation_penalty
                * miss_ratio
            )

        return float(reward)
