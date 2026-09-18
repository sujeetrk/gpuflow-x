from dataclasses import asdict
from typing import List

from core.telemetry.event import TelemetryEvent


class WorkloadDatasetBuilder:

    FEATURES = [
        "queue_time_ms",
        "inference_time_ms",
        "total_latency_ms",
        "batch_size",
    ]

    TARGET = "next_latency_ms"

    def build_rows(
        self,
        events: List[TelemetryEvent],
    ) -> List[dict]:

        completed = [
            event
            for event in events
            if event.status == "completed"
        ]

        completed.sort(key=lambda event: event.arrival_time)

        rows = []

        for index in range(len(completed) - 1):

            current = completed[index]
            next_event = completed[index + 1]

            rows.append(
                {
                    "queue_time_ms": current.queue_time_ms or 0.0,
                    "inference_time_ms": current.inference_time_ms or 0.0,
                    "total_latency_ms": current.total_latency_ms or 0.0,
                    "batch_size": current.batch_size,
                    "next_latency_ms": (
                        next_event.total_latency_ms or 0.0
                    ),
                }
            )

        return rows
