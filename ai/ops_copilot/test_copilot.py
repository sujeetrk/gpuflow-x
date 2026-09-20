from ai.ops_copilot.copilot import AIOpsCopilot


def test_explains_backlog_throughput_decision():
    copilot = AIOpsCopilot()
    explanation = copilot.explain_decision({
        "ai_reason": "backlog_throughput",
        "batch_size": 8,
        "batch_delay_ms": 0,
    })

    assert "queue reached" in explanation
    assert "batch size: 8" in explanation


def test_explains_urgent_deadline():
    explanation = AIOpsCopilot().explain_decision({
        "ai_reason": "urgent_deadline",
        "batch_size": 1,
        "batch_delay_ms": 0,
    })

    assert "deadline" in explanation.lower()
    assert "singleton" in explanation


def test_detects_queue_and_deadline_bottlenecks():
    report = AIOpsCopilot().analyze({
        "requests_submitted": 20,
        "requests_completed": 15,
        "requests_failed": 2,
        "queue_length": 3,
        "deadline_misses": 1,
        "last_ai_decision": {
            "ai_reason": "backlog_throughput",
            "batch_size": 8,
            "batch_delay_ms": 0,
        },
    })

    assert any("Queue backlog" in item for item in report["bottlenecks"])
    assert any("deadline misses" in item for item in report["bottlenecks"])
    assert report["observed_metrics"]["requests_failed"] == 2
    assert "backlog_throughput" in report["decision_explanation"]


def test_detects_queue_wait_bottleneck_from_telemetry():
    events = [
        {
            "status": "completed",
            "total_latency_ms": 30,
            "queue_time_ms": 20,
            "inference_time_ms": 5,
        },
        {
            "status": "completed",
            "total_latency_ms": 40,
            "queue_time_ms": 25,
            "inference_time_ms": 5,
        },
    ]

    report = AIOpsCopilot().analyze({}, events)

    assert report["observed_metrics"]["average_queue_time_ms"] == 22.5
    assert any(
        "queue wait exceeds" in item.lower()
        for item in report["bottlenecks"]
    )


def test_handles_missing_metrics_and_decision():
    report = AIOpsCopilot().analyze({})

    assert report["observed_metrics"]["queue_length"] == 0
    assert "No scheduler decision" in report["decision_explanation"]
    assert report["recommendations"]


def test_accepts_telemetry_objects():
    class Event:
        status = "completed"
        total_latency_ms = 12.0
        queue_time_ms = 3.0
        inference_time_ms = 8.0

    report = AIOpsCopilot().analyze({}, [Event()])

    assert report["observed_metrics"]["average_latency_ms"] == 12.0
