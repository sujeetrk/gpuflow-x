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
    ) -> float:

        latency_score = max(
            0.0,
            1.0
            - (
                average_latency_ms
                / self.config.target_latency_ms
            ),
        )

        queue_score = max(
            0.0,
            1.0
            - (
                average_queue_time_ms
                / self.config.target_latency_ms
            ),
        )

        throughput_score = min(
            1.0,
            throughput_requests_per_sec / 100.0,
        )

        reward = (
            self.config.latency_weight
            * latency_score
            + self.config.queue_weight
            * queue_score
            + self.config.throughput_weight
            * throughput_score
        )

        if (
            average_latency_ms
            > self.config.target_latency_ms
        ):
            reward -= self.config.sla_violation_penalty

        return float(reward)
