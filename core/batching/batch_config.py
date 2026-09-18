from dataclasses import dataclass


@dataclass
class BatchConfig:
    """
    Configuration for dynamic inference batching.
    """

    max_batch_size: int = 8
    max_batch_delay_ms: float = 10.0

    def __post_init__(self):
        if self.max_batch_size < 1:
            raise ValueError(
                "max_batch_size must be at least 1"
            )

        if self.max_batch_delay_ms < 0:
            raise ValueError(
                "max_batch_delay_ms cannot be negative"
            )
