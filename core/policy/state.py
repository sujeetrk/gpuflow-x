from dataclasses import dataclass


@dataclass
class WorkloadState:
    queue_length: int
    arrival_rate_rps: float
    gpu_utilization: float
    vram_utilization: float
    average_latency_ms: float
    p95_latency_ms: float
    current_batch_size: int
    current_batch_delay_ms: float
    average_queue_time_ms: float = 0.0
    average_inference_time_ms: float = 0.0

    def __post_init__(self):
        if self.queue_length < 0:
            raise ValueError("Queue length cannot be negative.")

        if self.arrival_rate_rps < 0:
            raise ValueError("Arrival rate cannot be negative.")

        if not 0.0 <= self.gpu_utilization <= 1.0:
            raise ValueError(
                "GPU utilization must be between 0 and 1."
            )

        if not 0.0 <= self.vram_utilization <= 1.0:
            raise ValueError(
                "VRAM utilization must be between 0 and 1."
            )

        if self.average_latency_ms < 0:
            raise ValueError(
                "Average latency cannot be negative."
            )

        if self.p95_latency_ms < 0:
            raise ValueError(
                "P95 latency cannot be negative."
            )

        if self.current_batch_size < 1:
            raise ValueError(
                "Batch size must be at least 1."
            )

        if self.current_batch_delay_ms < 0:
            raise ValueError(
                "Batch delay cannot be negative."
            )

        if self.average_queue_time_ms < 0:
            raise ValueError(
                "Average queue time cannot be negative."
            )

        if self.average_inference_time_ms < 0:
            raise ValueError(
                "Average inference time cannot be negative."
            )
