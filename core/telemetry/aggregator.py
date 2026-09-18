import math
import time
from typing import List

from core.telemetry.event import TelemetryEvent


class MetricsAggregator:
    """
    Calculates performance metrics from telemetry events.
    """

    def __init__(self, events: List[TelemetryEvent]):
        self.events = [
            event
            for event in events
            if event.status == "completed"
        ]

    @staticmethod
    def _percentile(
        values: List[float],
        percentile: float
    ) -> float:

        if not values:
            return 0.0

        values = sorted(values)

        index = (
            (len(values) - 1)
            * percentile
        )

        lower = math.floor(index)
        upper = math.ceil(index)

        if lower == upper:
            return values[lower]

        return (
            values[lower]
            + (values[upper] - values[lower])
            * (index - lower)
        )

    def calculate(self) -> dict:

        if not self.events:
            return {
                "request_count": 0,
                "average_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "average_queue_time_ms": 0.0,
                "average_inference_time_ms": 0.0,
                "average_batch_size": 0.0,
                "throughput_requests_per_second": 0.0,
                "success_rate": 0.0
            }

        latencies = [
            event.total_latency_ms
            for event in self.events
            if event.total_latency_ms is not None
        ]

        queue_times = [
            event.queue_time_ms
            for event in self.events
            if event.queue_time_ms is not None
        ]

        inference_times = [
            event.inference_time_ms
            for event in self.events
            if event.inference_time_ms is not None
        ]

        batch_sizes = [
            event.batch_size
            for event in self.events
        ]

        average_latency = (
            sum(latencies) / len(latencies)
            if latencies else 0.0
        )

        average_queue_time = (
            sum(queue_times) / len(queue_times)
            if queue_times else 0.0
        )

        average_inference_time = (
            sum(inference_times) / len(inference_times)
            if inference_times else 0.0
        )

        average_batch_size = (
            sum(batch_sizes) / len(batch_sizes)
            if batch_sizes else 0.0
        )

        start_times = [
            event.arrival_time
            for event in self.events
        ]

        end_times = [
            event.end_time
            for event in self.events
            if event.end_time is not None
        ]

        if start_times and end_times:

            duration = max(end_times) - min(start_times)

            throughput = (
                len(self.events) / duration
                if duration > 0
                else 0.0
            )

        else:
            throughput = 0.0

        return {
            "request_count": len(self.events),
            "average_latency_ms": average_latency,
            "p50_latency_ms": self._percentile(
                latencies,
                0.50
            ),
            "p95_latency_ms": self._percentile(
                latencies,
                0.95
            ),
            "p99_latency_ms": self._percentile(
                latencies,
                0.99
            ),
            "average_queue_time_ms": average_queue_time,
            "average_inference_time_ms": average_inference_time,
            "average_batch_size": average_batch_size,
            "throughput_requests_per_second": throughput,
            "success_rate": 1.0
        }
