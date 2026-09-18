import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceRequest:
    """
    Represents a single inference request moving through GPUFlow-X.
    """

    values: list[float]

    request_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    priority: int = 0

    arrival_time: float = field(
        default_factory=time.perf_counter
    )

    enqueue_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    status: str = "created"

    def mark_enqueued(self) -> None:
        self.enqueue_time = time.perf_counter()
        self.status = "queued"

    def mark_started(self) -> None:
        self.start_time = time.perf_counter()
        self.status = "processing"

    def mark_completed(self) -> None:
        self.end_time = time.perf_counter()
        self.status = "completed"

    def mark_failed(self) -> None:
        self.end_time = time.perf_counter()
        self.status = "failed"

    @property
    def queue_time_ms(self) -> Optional[float]:
        if self.enqueue_time is None or self.start_time is None:
            return None

        return (self.start_time - self.enqueue_time) * 1000

    @property
    def inference_time_ms(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None

        return (self.end_time - self.start_time) * 1000

    @property
    def total_latency_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None

        return (self.end_time - self.arrival_time) * 1000
