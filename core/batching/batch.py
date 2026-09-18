from dataclasses import dataclass, field
from typing import List

from core.queue.request import InferenceRequest


@dataclass
class InferenceBatch:
    """
    A collection of inference requests that will be
    executed together as a single model batch.
    """

    requests: List[InferenceRequest] = field(
        default_factory=list
    )

    created_time: float = 0.0

    def add(self, request: InferenceRequest) -> None:
        """Add a request to the batch."""
        self.requests.append(request)

    @property
    def size(self) -> int:
        """Return the number of requests in the batch."""
        return len(self.requests)

    @property
    def inputs(self) -> list[list[float]]:
        """Return model inputs for all requests."""
        return [
            request.values
            for request in self.requests
        ]

    def is_full(self, max_size: int) -> bool:
        """Check whether the batch reached its maximum size."""
        return self.size >= max_size

    def is_empty(self) -> bool:
        """Check whether the batch contains no requests."""
        return self.size == 0
