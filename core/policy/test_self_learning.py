from core.policy.self_learning import (
    SchedulingAction,
    SelfLearningPolicy,
)


def test_policy_has_actions():

    policy = SelfLearningPolicy()

    assert len(policy.actions) == 16


def test_policy_selects_valid_action():

    policy = SelfLearningPolicy(
        epsilon=0.0
    )

    action = policy.select_action(
        queue_length=10,
        predicted_latency_ms=20.0,
        current_batch_size=4,
    )

    assert isinstance(
        action,
        SchedulingAction,
    )

    assert action in policy.actions


def test_policy_learns_reward():

    policy = SelfLearningPolicy(
        epsilon=0.0
    )

    action = SchedulingAction(
        batch_size=8,
        batch_delay_ms=5,
    )

    for _ in range(10):

        policy.update(
            queue_length=10,
            predicted_latency_ms=20.0,
            current_batch_size=4,
            action=action,
            reward=10.0,
        )

    value = policy.action_value(
        queue_length=10,
        predicted_latency_ms=20.0,
        current_batch_size=4,
        action=action,
    )

    assert value == 10.0
    assert policy.total_updates == 10


def test_epsilon_decays():

    policy = SelfLearningPolicy(
        epsilon=1.0,
        epsilon_decay=0.5,
        min_epsilon=0.1,
    )

    action = policy.actions[0]

    policy.update(
        queue_length=1,
        predicted_latency_ms=5.0,
        current_batch_size=1,
        action=action,
        reward=1.0,
    )

    assert policy.epsilon == 0.5


def test_statistics():

    policy = SelfLearningPolicy()

    stats = policy.statistics()

    assert stats["actions"] == 16
    assert stats["learned_state_action_pairs"] == 0
    assert stats["total_updates"] == 0
