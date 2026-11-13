"""Aggregate operational metrics from inference telemetry."""

import math


class MetricsAggregator:
    """Calculate latency, throughput, failure, device and SLA metrics."""

    def __init__(self, events):
        self.all_events = list(events)
        self.events = [
            event for event in self.all_events
            if event.status == "completed"
        ]
        self.failed_events = [
            event for event in self.all_events
            if event.status == "failed"
        ]

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return 0.0

        values = sorted(values)
        index = (len(values) - 1) * percentile
        lower = math.floor(index)
        upper = math.ceil(index)

        if lower == upper:
            return values[lower]

        return (
            values[lower]
            + (values[upper] - values[lower]) * (index - lower)
        )

    def calculate(self):
        latencies = [
            e.total_latency_ms for e in self.events
            if e.total_latency_ms is not None
        ]
        queue_times = [
            e.queue_time_ms for e in self.events
            if e.queue_time_ms is not None
        ]
        inference_times = [
            e.inference_time_ms for e in self.events
            if e.inference_time_ms is not None
        ]
        batch_sizes = [e.batch_size for e in self.events]

        total = len(self.all_events)
        completed = len(self.events)
        failed = len(self.failed_events)

        starts = [e.arrival_time for e in self.events]
        ends = [
            e.end_time for e in self.events
            if e.end_time is not None
        ]
        duration = max(ends) - min(starts) if starts and ends else 0.0
        throughput = completed / duration if duration > 0 else 0.0

        cpu_count = sum(
            1 for e in self.all_events
            if "cpu" in (e.device or "").lower()
        )
        cpu_fallback_count = sum(
            1 for e in self.all_events
            if e.cpu_fallback
        )
        gpu_count = sum(
            1 for e in self.all_events
            if "cuda" in (e.device or "").lower()
            or "gpu" in (e.device or "").lower()
        )
        sla_violations = sum(
            1 for e in self.all_events
            if e.sla_violated
        )

        average = lambda values: (
            sum(values) / len(values) if values else 0.0
        )

        return {
            "request_count": completed,
            "total_request_count": total,
            "completed_request_count": completed,
            "failed_request_count": failed,
            "failure_rate": failed / total if total else 0.0,
            "average_latency_ms": average(latencies),
            "p50_latency_ms": self._percentile(latencies, 0.50),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "p99_latency_ms": self._percentile(latencies, 0.99),
            "average_queue_time_ms": average(queue_times),
            "average_inference_time_ms": average(inference_times),
            "average_batch_size": average(batch_sizes),
            "throughput_requests_per_second": throughput,
            "success_rate": completed / total if total else 0.0,
            "sla_violations": sla_violations,
            "cpu_request_count": cpu_count,
            "gpu_request_count": gpu_count,
            "cpu_fallback_count": cpu_fallback_count,
        }
