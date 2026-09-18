from dataclasses import dataclass
from typing import Optional


@dataclass
class TelemetryEvent:
    """
    Telemetry record for a single inference request.
    """

    request_id: str

    arrival_time: float
    enqueue_time: Optional[float]
    start_time: Optional[float]
    end_time: Optional[float]

    queue_time_ms: Optional[float]
    inference_time_ms: Optional[float]
    total_latency_ms: Optional[float]

    batch_size: int

    status: str

    device: str
