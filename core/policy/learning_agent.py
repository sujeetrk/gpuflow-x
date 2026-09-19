from core.policy.reward import SchedulerReward
from core.policy.self_learning import (
    SchedulingAction,
    SelfLearningPolicy,
)


class LearningAgent:

    def __init__(
        self,
        policy=None,
        reward_model=None,
    ):
        self.policy = (
            policy
            if policy is not None
            else SelfLearningPolicy()
        )

        self.reward_model = (
            reward_model
            if reward_model is not None
            else SchedulerReward()
        )

        self.decisions = 0
        self.total_reward = 0.0

    def choose_action(
        self,
        queue_length: int,
        predicted_latency_ms: float,
        current_batch_size: int,
    ) -> SchedulingAction:

        return self.policy.select_action(
            queue_length=queue_length,
            predicted_latency_ms=predicted_latency_ms,
            current_batch_size=current_batch_size,
        )

    def learn(
        self,
        queue_length: int,
        predicted_latency_ms: float,
        current_batch_size: int,
        action: SchedulingAction,
        average_latency_ms: float,
        average_queue_time_ms: float,
        throughput_requests_per_sec: float,
        deadline_misses: int = 0,
        request_count: int = 0,
    ) -> float:

        reward = self.reward_model.calculate(
            average_latency_ms=average_latency_ms,
            average_queue_time_ms=average_queue_time_ms,
            throughput_requests_per_sec=throughput_requests_per_sec,
            deadline_misses=deadline_misses,
            request_count=request_count,
        )

        self.policy.update(
            queue_length=queue_length,
            predicted_latency_ms=predicted_latency_ms,
            current_batch_size=current_batch_size,
            action=action,
            reward=reward,
        )

        self.decisions += 1
        self.total_reward += reward

        return reward

    def statistics(self) -> dict:

        policy_stats = self.policy.statistics()

        average_reward = (
            self.total_reward / self.decisions
            if self.decisions > 0
            else 0.0
        )

        return {
            **policy_stats,
            "decisions": self.decisions,
            "total_reward": self.total_reward,
            "average_reward": average_reward,
        }
