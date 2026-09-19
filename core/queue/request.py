import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceRequest:
    """
    Represents a single inference request in GPUFlow-X.

    Higher priority values indicate more important requests.
    SLA budget is measured in milliseconds.
    Deadline timestamps use time.perf_counter().
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

    # Phase 9: SLA and deadline metadata
    sla_budget_ms: Optional[float] = None
    deadline_time: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sla_budget_ms is not None:
            if self.sla_budget_ms <= 0:
                raise ValueError(
                    "SLA budget must be greater than zero."
                )

            # Derive deadline from arrival time if not provided.
            if self.deadline_time is None:
                self.deadline_time = (
                    self.arrival_time
                    + self.sla_budget_ms / 1000
                )

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

        return (
            self.start_time - self.enqueue_time
        ) * 1000

    @property
    def inference_time_ms(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None

        return (
            self.end_time - self.start_time
        ) * 1000

    @property
    def total_latency_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None

        return (
            self.end_time - self.arrival_time
        ) * 1000

    @property
    def deadline_missed(self) -> bool:
        """
        True when a completed or processing request
        has passed its deadline.
        """
        if self.deadline_time is None:
            return False

        check_time = (
            self.end_time
            if self.end_time is not None
            else time.perf_counter()
        )

        return check_time > self.deadline_time

    @property
    def remaining_sla_ms(self) -> Optional[float]:
        """
        Remaining time before the deadline.
        Negative means the deadline has passed.
        """
        if self.deadline_time is None:
            return None

        return (
            self.deadline_time - time.perf_counter()
        ) * 1000
