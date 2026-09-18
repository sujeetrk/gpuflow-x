from core.policy.learning_agent import LearningAgent
from core.policy.self_learning import SchedulingAction


def test_agent_selects_action():

    agent = LearningAgent()

    action = agent.choose_action(
        queue_length=10,
        predicted_latency_ms=20.0,
        current_batch_size=4,
    )

    assert isinstance(
        action,
        SchedulingAction,
    )

    assert action in agent.policy.actions


def test_agent_learns_from_result():

    agent = LearningAgent()

    action = SchedulingAction(
        batch_size=8,
        batch_delay_ms=5,
    )

    reward = agent.learn(
        queue_length=10,
        predicted_latency_ms=20.0,
        current_batch_size=4,
        action=action,
        average_latency_ms=10.0,
        average_queue_time_ms=2.0,
        throughput_requests_per_sec=80.0,
    )

    assert isinstance(reward, float)
    assert agent.decisions == 1
    assert agent.total_reward == reward


def test_agent_statistics():

    agent = LearningAgent()

    stats = agent.statistics()

    assert stats["decisions"] == 0
    assert stats["total_updates"] == 0
    assert stats["average_reward"] == 0.0
