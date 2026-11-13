from core.telemetry.aggregator import MetricsAggregator
from core.telemetry.event import TelemetryEvent


def make_event(
    request_id,
    status="completed",
    device="cpu",
    latency=100.0,
    sla_violated=False,
    arrival=1.0,
):
    return TelemetryEvent(
        request_id=request_id,
        arrival_time=arrival,
        enqueue_time=arrival + 0.01,
        start_time=arrival + 0.02,
        end_time=arrival + latency / 1000,
        queue_time_ms=10.0,
        inference_time_ms=latency - 10.0,
        total_latency_ms=latency,
        batch_size=2,
        status=status,
        device=device,
        sla_violated=sla_violated,
    )


def test_metrics_include_completed_and_failed_requests():
    events = [
        make_event("r1", status="completed"),
        make_event("r2", status="failed"),
    ]

    metrics = MetricsAggregator(events).calculate()

    assert metrics["total_request_count"] == 2
    assert metrics["completed_request_count"] == 1
    assert metrics["failed_request_count"] == 1
    assert metrics["failure_rate"] == 0.5
    assert metrics["success_rate"] == 0.5


def test_metrics_track_cpu_and_gpu_execution():
    events = [
        make_event("r1", device="cpu"),
        make_event("r2", device="cuda:0"),
        make_event("r3", device="cpu"),
    ]

    metrics = MetricsAggregator(events).calculate()

    assert metrics["cpu_request_count"] == 2
    assert metrics["gpu_request_count"] == 1
    assert metrics["cpu_fallback_count"] == 0


def test_metrics_count_actual_sla_violations():
    events = [
        make_event("r1", sla_violated=True),
        make_event("r2", sla_violated=False),
        make_event("r3", status="failed", sla_violated=True),
    ]

    metrics = MetricsAggregator(events).calculate()

    assert metrics["sla_violations"] == 2


def test_empty_telemetry_returns_safe_zero_metrics():
    metrics = MetricsAggregator([]).calculate()

    assert metrics["total_request_count"] == 0
    assert metrics["completed_request_count"] == 0
    assert metrics["failed_request_count"] == 0
    assert metrics["failure_rate"] == 0.0
    assert metrics["success_rate"] == 0.0
    assert metrics["sla_violations"] == 0
    assert metrics["cpu_request_count"] == 0
    assert metrics["gpu_request_count"] == 0


def test_latency_metrics_exclude_failed_requests():
    events = [
        make_event("r1", latency=100.0),
        make_event("r2", latency=300.0),
        make_event("r3", status="failed", latency=900.0),
    ]

    metrics = MetricsAggregator(events).calculate()

    assert metrics["request_count"] == 2
    assert metrics["average_latency_ms"] == 200.0
    assert metrics["p50_latency_ms"] == 200.0



def test_cpu_fallback_count_only_counts_gpu_failure_retries():
    normal_cpu = make_event("r1", device="cpu")
    actual_fallback = make_event("r2", device="cpu")
    actual_fallback.cpu_fallback = True
    gpu = make_event("r3", device="cuda:0")

    metrics = MetricsAggregator(
        [normal_cpu, actual_fallback, gpu]
    ).calculate()

    assert metrics["cpu_request_count"] == 2
    assert metrics["gpu_request_count"] == 1
    assert metrics["cpu_fallback_count"] == 1
