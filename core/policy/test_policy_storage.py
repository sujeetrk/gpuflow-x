from pathlib import Path

from core.policy.policy_storage import PolicyStorage
from core.policy.self_learning import (
    SchedulingAction,
    SelfLearningPolicy,
)


def test_policy_save_and_load(tmp_path):

    policy = SelfLearningPolicy(
        epsilon=0.2
    )

    action = SchedulingAction(
        batch_size=8,
        batch_delay_ms=5,
    )

    for _ in range(5):

        policy.update(
            queue_length=10,
            predicted_latency_ms=20.0,
            current_batch_size=4,
            action=action,
            reward=5.0,
        )

    storage = PolicyStorage(
        tmp_path / "policy.json"
    )

    storage.save(policy)

    assert Path(
        tmp_path / "policy.json"
    ).exists()

    restored_policy = SelfLearningPolicy()

    loaded = storage.load(
        restored_policy
    )

    assert loaded is True

    assert restored_policy.total_updates == 5

    assert (
        restored_policy.action_value(
            queue_length=10,
            predicted_latency_ms=20.0,
            current_batch_size=4,
            action=action,
        )
        == 5.0
    )


def test_load_missing_policy_returns_false(tmp_path):

    storage = PolicyStorage(
        tmp_path / "missing.json"
    )

    policy = SelfLearningPolicy()

    assert storage.load(policy) is False
