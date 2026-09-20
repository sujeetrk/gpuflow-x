import time

from core.batching.batch_config import BatchConfig
from core.policy.ai_policy import AIPolicy
from core.scheduler.ai_scheduler import AIScheduler
from core.queue.request import InferenceRequest
from inference.service import InferenceService


def test_ai_scheduler_completes_requests():

    config = BatchConfig(
        max_batch_size=4,
        max_batch_delay_ms=5
    )

    policy = AIPolicy(
        target_p95_latency_ms=200
    )

    scheduler = AIScheduler(
        inference_service=InferenceService(),
        policy=policy,
        batch_config=config,
        max_queue_size=20
    )

    scheduler.start()

    requests = [
        InferenceRequest([i] * 10)
        for i in range(1, 9)
    ]

    for request in requests:
        scheduler.submit(request)

    deadline = time.perf_counter() + 10

    while scheduler.metrics()["requests_completed"] < 8:
        if time.perf_counter() >= deadline:
            break

        time.sleep(0.01)

    scheduler.stop()

    metrics = scheduler.metrics()

    assert metrics["requests_submitted"] == 8
    assert metrics["requests_completed"] == 8
    assert metrics["requests_failed"] == 0
    assert metrics["batches_executed"] >= 1
    assert metrics["telemetry_events"] == 8

    assert all(
        request.status == "completed"
        for request in requests
    )


def test_ai_scheduler_records_ai_decision():

    scheduler = AIScheduler(
        inference_service=InferenceService(),
        policy=AIPolicy(),
        batch_config=BatchConfig(
            max_batch_size=4,
            max_batch_delay_ms=5
        ),
        max_queue_size=20
    )

    scheduler.start()

    requests = [
        InferenceRequest([i] * 10)
        for i in range(1, 5)
    ]

    for request in requests:
        scheduler.submit(request)

    deadline = time.perf_counter() + 10

    while scheduler.metrics()["requests_completed"] < 4:
        if time.perf_counter() >= deadline:
            break

        time.sleep(0.01)

    scheduler.stop()

    decision = scheduler.metrics()["last_ai_decision"]

    assert decision is not None
    assert "predicted_latency_ms" in decision
    assert "ai_reason" in decision
    assert decision["predicted_latency_ms"] >= 0


class _FixedPolicy:
    def decide(self, state):
        return {
            "batch_size": 4,
            "batch_delay_ms": 10.0,
            "ai_reason": "test_policy",
            "predicted_latency_ms": 5.0,
        }


class _FixedLearningAgent:
    def choose_action(self, **kwargs):
        from core.policy.self_learning import SchedulingAction
        return SchedulingAction(batch_size=8, batch_delay_ms=20.0)


def test_sla_override_is_not_attributed_to_proposed_learning_action(tmp_path, monkeypatch):
    scheduler = AIScheduler(
        inference_service=InferenceService(),
        policy=_FixedPolicy(),
        batch_config=BatchConfig(max_batch_size=8, max_batch_delay_ms=20.0),
        learning_enabled=True,
        policy_path=str(tmp_path / "attribution-policy.json"),
        deadline_guard_ms=1.0,
    )
    scheduler.learning_agent = _FixedLearningAgent()
    monkeypatch.setattr(scheduler, "_deadline_pressure", lambda: 0.5)

    _, decision = scheduler._apply_policy()

    assert decision["proposed_learning_action"] == {
        "batch_size": 8, "batch_delay_ms": 20.0
    }
    assert decision["effective_action"] == {
        "batch_size": 1, "batch_delay_ms": 0.0
    }
    assert "learning_action" not in decision
    assert decision["learning_skipped_for_action_override"] is True


def test_telemetry_failure_does_not_reclassify_completed_request(tmp_path, monkeypatch):
    from core.batching.batch import InferenceBatch

    class _SuccessfulInference:
        def infer_batch_object(self, batch):
            return {"ok": True}

    class _OneBatchCollector:
        def __init__(self, scheduler):
            self.scheduler = scheduler
            self.used = False

        def collect(self):
            if self.used:
                return None
            self.used = True
            request = self.scheduler.queue.dequeue(timeout=0)
            self.scheduler._running = False
            batch = InferenceBatch()
            batch.add(request)
            return batch

    scheduler = AIScheduler(
        inference_service=_SuccessfulInference(),
        policy=_FixedPolicy(),
        batch_config=BatchConfig(max_batch_size=1, max_batch_delay_ms=0),
        learning_enabled=False,
        policy_path=str(tmp_path / "telemetry-policy.json"),
    )
    request = InferenceRequest(values=[1.0])
    scheduler.submit(request)
    scheduler.collector = _OneBatchCollector(scheduler)
    monkeypatch.setattr(
        scheduler, "_record_telemetry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("sink down")),
    )
    scheduler._running = True
    scheduler._worker_loop()

    metrics = scheduler.metrics()
    assert request.status == "completed"
    assert metrics["requests_completed"] == 1
    assert metrics["requests_failed"] == 0


class _FixedPolicy:
    def decide(self, state):
        return {
            "batch_size": 4,
            "batch_delay_ms": 10.0,
            "ai_reason": "test_policy",
            "predicted_latency_ms": 5.0,
        }


class _FixedLearningAgent:
    def choose_action(self, **kwargs):
        from core.policy.self_learning import SchedulingAction
        return SchedulingAction(batch_size=8, batch_delay_ms=20.0)


def test_sla_override_is_not_attributed_to_proposed_learning_action(
    tmp_path, monkeypatch
):
    scheduler = AIScheduler(
        inference_service=InferenceService(),
        policy=_FixedPolicy(),
        batch_config=BatchConfig(
            max_batch_size=8, max_batch_delay_ms=20.0
        ),
        learning_enabled=True,
        policy_path=str(tmp_path / "attribution-policy.json"),
        deadline_guard_ms=1.0,
    )
    scheduler.learning_agent = _FixedLearningAgent()
    monkeypatch.setattr(scheduler, "_deadline_pressure", lambda: 0.5)

    _, decision = scheduler._apply_policy()

    assert decision["proposed_learning_action"] == {
        "batch_size": 8, "batch_delay_ms": 20.0
    }
    assert decision["effective_action"] == {
        "batch_size": 1, "batch_delay_ms": 0.0
    }
    assert "learning_action" not in decision
    assert decision["learning_skipped_for_action_override"] is True


def test_telemetry_failure_does_not_reclassify_completed_request(
    tmp_path, monkeypatch
):
    from core.batching.batch import InferenceBatch

    class _SuccessfulInference:
        def infer_batch_object(self, batch):
            return {"ok": True}

    class _OneBatchCollector:
        def __init__(self, scheduler):
            self.scheduler = scheduler
            self.used = False

        def collect(self):
            if self.used:
                return None
            self.used = True
            request = self.scheduler.queue.dequeue(timeout=0)
            self.scheduler._running = False
            batch = InferenceBatch()
            batch.add(request)
            return batch

    scheduler = AIScheduler(
        inference_service=_SuccessfulInference(),
        policy=_FixedPolicy(),
        batch_config=BatchConfig(max_batch_size=1, max_batch_delay_ms=0),
        learning_enabled=False,
        policy_path=str(tmp_path / "telemetry-policy.json"),
    )
    request = InferenceRequest(values=[1.0])
    scheduler.submit(request)
    scheduler.collector = _OneBatchCollector(scheduler)
    monkeypatch.setattr(
        scheduler,
        "_record_telemetry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("telemetry sink down")
        ),
    )
    scheduler._running = True
    scheduler._worker_loop()

    metrics = scheduler.metrics()
    assert request.status == "completed"
    assert metrics["requests_completed"] == 1
    assert metrics["requests_failed"] == 0
