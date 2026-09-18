from ai.workload_predictor.predictor import WorkloadPredictor
from core.policy.ai_policy import AIPolicy
from core.policy.state import WorkloadState


def test_ai_policy_returns_prediction():

    predictor = WorkloadPredictor()

    policy = AIPolicy(
        predictor=predictor,
        target_p95_latency_ms=200
    )

    state = WorkloadState(
        queue_length=4,
        arrival_rate_rps=20,
        gpu_utilization=0.5,
        vram_utilization=0.4,
        average_latency_ms=25,
        p95_latency_ms=30,
        current_batch_size=4,
        current_batch_delay_ms=5
    )

    decision = policy.decide(state)

    assert "predicted_latency_ms" in decision
    assert decision["predicted_latency_ms"] >= 0
    assert "ai_reason" in decision


def test_ai_policy_safe_prediction():

    predictor = WorkloadPredictor()

    policy = AIPolicy(
        predictor=predictor,
        target_p95_latency_ms=200
    )

    state = WorkloadState(
        queue_length=2,
        arrival_rate_rps=5,
        gpu_utilization=0.3,
        vram_utilization=0.2,
        average_latency_ms=10,
        p95_latency_ms=15,
        current_batch_size=4,
        current_batch_delay_ms=5
    )

    decision = policy.decide(state)

    assert decision["predicted_latency_ms"] >= 0
    assert decision["ai_reason"] == "predicted_latency_safe"


def test_ai_policy_preserves_batching_under_queue_pressure():

    predictor = WorkloadPredictor()

    policy = AIPolicy(
        predictor=predictor,
        min_batch_size=1,
        max_batch_size=8,
        min_delay_ms=1,
        max_delay_ms=20,
        target_p95_latency_ms=0.001
    )

    state = WorkloadState(
        queue_length=10,
        arrival_rate_rps=50,
        gpu_utilization=0.7,
        vram_utilization=0.5,
        average_latency_ms=25,
        p95_latency_ms=30,
        current_batch_size=4,
        current_batch_delay_ms=5
    )

    decision = policy.decide(state)

    assert decision["predicted_latency_ms"] > 0
    assert decision["ai_reason"] == (
        "predicted_latency_high_queue_pressure"
    )
    assert decision["batch_size"] >= 4
