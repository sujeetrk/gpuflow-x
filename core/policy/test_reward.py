from core.policy.reward import (
    RewardConfig,
    SchedulerReward,
)


def test_low_latency_gets_positive_reward():

    reward_model = SchedulerReward()

    reward = reward_model.calculate(
        average_latency_ms=10.0,
        average_queue_time_ms=2.0,
        throughput_requests_per_sec=80.0,
    )

    assert reward > 0


def test_high_latency_gets_sla_penalty():

    reward_model = SchedulerReward(
        RewardConfig(
            target_latency_ms=50.0,
            sla_violation_penalty=2.0,
        )
    )

    reward = reward_model.calculate(
        average_latency_ms=100.0,
        average_queue_time_ms=20.0,
        throughput_requests_per_sec=10.0,
    )

    assert reward < 0


def test_zero_latency_and_queue_are_valid():

    reward_model = SchedulerReward()

    reward = reward_model.calculate(
        average_latency_ms=0.0,
        average_queue_time_ms=0.0,
        throughput_requests_per_sec=0.0,
    )

    assert reward == 1.5


def test_reward_is_float():

    reward_model = SchedulerReward()

    reward = reward_model.calculate(
        average_latency_ms=20.0,
        average_queue_time_ms=5.0,
        throughput_requests_per_sec=50.0,
    )

    assert isinstance(reward, float)
