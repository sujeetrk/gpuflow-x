from core.policy.adaptive_delay_policy import AdaptiveDelayPolicy
from core.policy.state import WorkloadState


def test_high_queue_increases_delay():

    policy = AdaptiveDelayPolicy(
        min_delay_ms=1,
        max_delay_ms=20,
        target_p95_latency_ms=200
    )

    state = WorkloadState(
        queue_length=10,
        arrival_rate_rps=50,
        gpu_utilization=0.7,
        vram_utilization=0.5,
        average_latency_ms=50,
        p95_latency_ms=100,
        current_batch_size=4,
        current_batch_delay_ms=5
    )

    decision = policy.decide(state)

    assert decision["batch_delay_ms"] == 6
    assert decision["reason"] == "high_queue"


def test_high_latency_reduces_delay():

    policy = AdaptiveDelayPolicy(
        min_delay_ms=1,
        max_delay_ms=20,
        target_p95_latency_ms=200
    )

    state = WorkloadState(
        queue_length=10,
        arrival_rate_rps=50,
        gpu_utilization=0.9,
        vram_utilization=0.8,
        average_latency_ms=250,
        p95_latency_ms=300,
        current_batch_size=8,
        current_batch_delay_ms=10
    )

    decision = policy.decide(state)

    assert decision["batch_delay_ms"] == 5
    assert decision["reason"] == "high_latency"


def test_stable_workload_keeps_delay():

    policy = AdaptiveDelayPolicy(
        min_delay_ms=1,
        max_delay_ms=20,
        target_p95_latency_ms=200
    )

    state = WorkloadState(
        queue_length=2,
        arrival_rate_rps=5,
        gpu_utilization=0.4,
        vram_utilization=0.3,
        average_latency_ms=50,
        p95_latency_ms=100,
        current_batch_size=4,
        current_batch_delay_ms=5
    )

    decision = policy.decide(state)

    assert decision["batch_delay_ms"] == 5
    assert decision["reason"] == "stable"
