import time

from core.batching.batch_config import BatchConfig
from core.policy.ai_policy import AIPolicy
from core.queue.request import InferenceRequest
from core.scheduler.ai_scheduler import AIScheduler
from inference.service import InferenceService


def create_scheduler(
    tmp_path,
    learning_enabled=False,
    deadline_guard_ms=1.0,
):
    return AIScheduler(
        inference_service=InferenceService(),
        policy=AIPolicy(),
        batch_config=BatchConfig(
            max_batch_size=8,
            max_batch_delay_ms=20,
        ),
        max_queue_size=20,
        learning_enabled=learning_enabled,
        policy_path=str(tmp_path / "test_policy.json"),
        deadline_guard_ms=deadline_guard_ms,
    )


def test_urgent_deadline_reduces_batch_size_and_delay(tmp_path):
    # A 1000 ms guard makes a 500 ms deadline
    # reliably urgent without being already expired.
    scheduler = create_scheduler(
        tmp_path,
        deadline_guard_ms=1000.0,
    )

    request = InferenceRequest(
        values=[1.0, 2.0],
        deadline_time=time.perf_counter() + 0.5,
        priority=10,
    )
    scheduler.submit(request)

    _, decision = scheduler._apply_policy()

    assert 0 < decision["deadline_pressure_ms"] <= 1000.0
    assert decision["batch_size"] == 1
    assert decision["batch_delay_ms"] == 0.0
    assert decision["ai_reason"] == "urgent_deadline"


def test_deadline_pressure_is_none_without_deadlines(tmp_path):
    scheduler = create_scheduler(tmp_path)

    scheduler.submit(InferenceRequest(values=[1.0]))

    _, decision = scheduler._apply_policy()

    assert decision["deadline_pressure_ms"] is None
    assert decision["batch_size"] >= 1
    assert decision["batch_delay_ms"] >= 0


def test_deadline_miss_is_recorded_in_telemetry(tmp_path):
    scheduler = create_scheduler(tmp_path)

    request = InferenceRequest(
        values=[1.0],
        deadline_time=time.perf_counter() - 1.0,
    )

    request.mark_enqueued()
    request.mark_started()
    request.mark_completed()

    scheduler._record_telemetry(request, batch_size=1)

    metrics = scheduler.metrics()

    assert metrics["deadline_misses"] == 1
    assert metrics["telemetry_events"] == 1


def test_on_time_request_does_not_count_as_deadline_miss(tmp_path):
    scheduler = create_scheduler(tmp_path)

    request = InferenceRequest(
        values=[1.0],
        deadline_time=time.perf_counter() + 10.0,
    )

    request.mark_enqueued()
    request.mark_started()
    request.mark_completed()

    scheduler._record_telemetry(request, batch_size=1)

    assert scheduler.metrics()["deadline_misses"] == 0


def test_self_learning_policy_compatible_with_deadline_scheduler(
    tmp_path,
):
    scheduler = create_scheduler(
        tmp_path,
        learning_enabled=True,
        deadline_guard_ms=1000.0,
    )

    request = InferenceRequest(
        values=[1.0],
        deadline_time=time.perf_counter() + 0.5,
        priority=5,
    )
    scheduler.submit(request)

    _, decision = scheduler._apply_policy()

    assert decision["learning_enabled"] is True
    assert "proposed_learning_action" in decision
    assert "effective_action" in decision

    assert 0 < decision["deadline_pressure_ms"] <= 1000.0
    assert decision["batch_size"] == 1
    assert decision["batch_delay_ms"] == 0.0

    proposed = decision["proposed_learning_action"]
    effective = decision["effective_action"]
    if proposed == effective:
        assert decision["learning_action"] == effective
    else:
        assert "learning_action" not in decision
        assert decision["learning_skipped_for_action_override"] is True


def test_expired_deadline_drains_using_configured_batch_capacity(tmp_path):
    scheduler = create_scheduler(tmp_path, deadline_guard_ms=1.0)

    for _ in range(4):
        scheduler.submit(
            InferenceRequest(
                values=[1.0],
                deadline_time=time.perf_counter() - 1.0,
            )
        )

    _, decision = scheduler._apply_policy()

    assert decision["ai_reason"] == "expired_deadline_drain"
    assert decision["batch_size"] == 8
    assert decision["batch_delay_ms"] == 0.0
    assert decision["deadline_override"] is True


def test_backlog_uses_batch_capacity_without_running_expensive_policy(tmp_path):
    class ExplodingPolicy:
        def decide(self, state):
            raise AssertionError("AI model should be bypassed for deep backlog")

    scheduler = AIScheduler(
        inference_service=InferenceService(),
        policy=ExplodingPolicy(),
        batch_config=BatchConfig(max_batch_size=4, max_batch_delay_ms=5),
        max_queue_size=20,
        learning_enabled=False,
        policy_path=str(tmp_path / "backlog_policy.json"),
    )

    for _ in range(8):
        scheduler.submit(InferenceRequest(values=[1.0]))

    _, decision = scheduler._apply_policy()

    assert decision["ai_reason"] == "backlog_throughput"
    assert decision["batch_size"] == 4
    assert decision["batch_delay_ms"] == 0.0
    assert decision["policy_bypassed_for_backlog"] is True
