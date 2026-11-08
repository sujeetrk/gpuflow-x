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


def test_policy_load_restores_learning_parameters(tmp_path):
    original_policy = SelfLearningPolicy(
        epsilon=0.15,
        epsilon_decay=0.90,
        min_epsilon=0.02,
    )

    storage = PolicyStorage(tmp_path / "policy.json")
    storage.save(original_policy)

    restored_policy = SelfLearningPolicy()
    assert storage.load(restored_policy) is True

    assert restored_policy.epsilon == 0.15
    assert restored_policy.epsilon_decay == 0.90
    assert restored_policy.min_epsilon == 0.02


def test_failed_atomic_replace_preserves_existing_policy(
    tmp_path,
    monkeypatch,
):
    import json
    import pytest
    import core.policy.policy_storage as storage_module

    policy_path = tmp_path / "policy.json"
    storage = PolicyStorage(policy_path)

    original_policy = SelfLearningPolicy(epsilon=0.20)
    storage.save(original_policy)

    original_content = policy_path.read_text()
    updated_policy = SelfLearningPolicy(epsilon=0.10)

    def fail_replace(source, destination):
        raise OSError("Simulated replacement failure")

    monkeypatch.setattr(storage_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="Simulated replacement failure"):
        storage.save(updated_policy)

    assert policy_path.read_text() == original_content
    assert json.loads(policy_path.read_text())["epsilon"] == 0.20
    assert list(tmp_path.glob(".policy.json.*.tmp")) == []
