class WorkloadClassifier:
    """
    Classifies the current runtime workload using scheduler telemetry.

    This is a deterministic, rule-based classifier. It does not claim
    to be a trained ML model.
    """

    CLASSES = (
        "latency_sensitive",
        "throughput_oriented",
        "compute_intensive",
        "balanced",
    )

    def __init__(self, high_arrival_rate_rps=20.0):
        if high_arrival_rate_rps <= 0:
            raise ValueError("high_arrival_rate_rps must be positive.")
        self.high_arrival_rate_rps = float(high_arrival_rate_rps)

    def classify(self, state, target_latency_ms=200.0):
        if target_latency_ms <= 0:
            raise ValueError("target_latency_ms must be positive.")

        queue_time = max(float(state.average_queue_time_ms), 0.0)
        inference_time = max(float(state.average_inference_time_ms), 0.0)
        p95_latency = max(float(state.p95_latency_ms), 0.0)
        arrival_rate = max(float(state.arrival_rate_rps), 0.0)

        queue_pressure = (
            state.queue_length >= max(2, state.current_batch_size * 2)
        )

        latency_pressure = (
            p95_latency > target_latency_ms
            or (
                queue_time >= 5.0
                and queue_time > max(inference_time * 1.5, 1.0)
            )
        )

        compute_pressure = (
            inference_time >= 5.0
            and inference_time > max(queue_time * 2.0, 1.0)
        )

        # Queue pressure takes priority so batching is not reduced
        # when the system needs higher throughput.
        if queue_pressure or arrival_rate >= self.high_arrival_rate_rps:
            workload_class = "throughput_oriented"
            reason = "queue_or_arrival_rate_pressure"

        elif latency_pressure:
            workload_class = "latency_sensitive"
            reason = "queue_or_p95_latency_pressure"

        elif compute_pressure:
            workload_class = "compute_intensive"
            reason = "inference_time_dominates_queue_time"

        else:
            workload_class = "balanced"
            reason = "no_dominant_workload_pressure"

        return {
            "workload_class": workload_class,
            "classification_reason": reason,
            "classification_signals": {
                "queue_length": state.queue_length,
                "arrival_rate_rps": arrival_rate,
                "average_queue_time_ms": queue_time,
                "average_inference_time_ms": inference_time,
                "p95_latency_ms": p95_latency,
            },
        }
