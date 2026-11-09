from ai.workload_predictor.classifier import WorkloadClassifier
from core.policy.ai_policy import AIPolicy
from core.policy.state import WorkloadState


def make_state(
    queue_length=1,
    arrival_rate_rps=5.0,
    average_latency_ms=20.0,
    p95_latency_ms=30.0,
    current_batch_size=4,
    current_batch_delay_ms=5.0,
    average_queue_time_ms=1.0,
    average_inference_time_ms=2.0,
):
    return WorkloadState(
        queue_length=queue_length,
        arrival_rate_rps=arrival_rate_rps,
        gpu_utilization=0.0,
        vram_utilization=0.0,
        average_latency_ms=average_latency_ms,
        p95_latency_ms=p95_latency_ms,
        current_batch_size=current_batch_size,
        current_batch_delay_ms=current_batch_delay_ms,
        average_queue_time_ms=average_queue_time_ms,
        average_inference_time_ms=average_inference_time_ms,
    )


def test_classifies_latency_sensitive_workload():
    classifier = WorkloadClassifier()

    result = classifier.classify(
        make_state(p95_latency_ms=250.0),
        target_latency_ms=200.0,
    )

    assert result["workload_class"] == "latency_sensitive"
    assert result["classification_reason"] == (
        "queue_or_p95_latency_pressure"
    )


def test_classifies_throughput_oriented_workload():
    classifier = WorkloadClassifier()

    result = classifier.classify(
        make_state(queue_length=10, arrival_rate_rps=10.0)
    )

    assert result["workload_class"] == "throughput_oriented"


def test_classifies_compute_intensive_workload():
    classifier = WorkloadClassifier()

    result = classifier.classify(
        make_state(
            average_queue_time_ms=1.0,
            average_inference_time_ms=20.0,
        )
    )

    assert result["workload_class"] == "compute_intensive"


def test_classifies_balanced_workload():
    classifier = WorkloadClassifier()

    result = classifier.classify(make_state())

    assert result["workload_class"] == "balanced"


def test_classifier_rejects_invalid_target_latency():
    classifier = WorkloadClassifier()

    try:
        classifier.classify(make_state(), target_latency_ms=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


class FixedPredictor:
    def predict(self, **kwargs):
        return 10.0


def test_latency_sensitive_classification_reduces_batching():
    policy = AIPolicy(
        predictor=FixedPredictor(),
        target_p95_latency_ms=200.0,
    )

    state = make_state(
        current_batch_size=4,
        current_batch_delay_ms=5.0,
        p95_latency_ms=250.0,
    )

    decision = policy.decide(state)

    assert decision["workload_class"] == "latency_sensitive"
    assert decision["batch_size"] == 2
    assert decision["batch_delay_ms"] <= 2.5


def test_compute_intensive_classification_increases_batch_size():
    policy = AIPolicy(predictor=FixedPredictor())

    state = make_state(
        current_batch_size=4,
        average_queue_time_ms=1.0,
        average_inference_time_ms=20.0,
    )

    decision = policy.decide(state)

    assert decision["workload_class"] == "compute_intensive"
    assert decision["batch_size"] == 5


def test_policy_exposes_classification_metadata():
    policy = AIPolicy(predictor=FixedPredictor())

    decision = policy.decide(make_state())

    assert decision["workload_class"] == "balanced"
    assert decision["classification_reason"]
    assert "classification_signals" in decision
    assert decision["predicted_latency_ms"] == 10.0
