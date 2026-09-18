from abc import ABC, abstractmethod
from typing import Any


class InferenceBackend(ABC):
    """
    Common interface for all inference backends.

    The scheduler interacts only with this interface,
    allowing CPU and CUDA implementations to be swapped.
    """

    @abstractmethod
    def load_model(self, model: Any) -> None:
        """Load an inference model."""
        pass

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Run inference on the supplied inputs."""
        pass

    @abstractmethod
    def device(self) -> str:
        """Return the execution device."""
        pass
