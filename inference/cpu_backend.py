import time
import torch

from inference.backend import InferenceBackend


class CPUBackend(InferenceBackend):
    """
    CPU-based inference backend for local development.

    This backend will later have a CUDA equivalent.
    """

    def __init__(self):
        self.model = None
        self._device = torch.device("cpu")

    def load_model(self, model) -> None:
        """Load a PyTorch model onto the CPU."""
        self.model = model.to(self._device)
        self.model.eval()

    def predict(self, inputs):
        """
        Run inference and return prediction + latency.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        start_time = time.perf_counter()

        with torch.no_grad():
            output = self.model(inputs)

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        return {
            "output": output,
            "latency_ms": latency_ms,
            "device": self.device(),
        }

    def device(self) -> str:
        """Return the execution device."""
        return str(self._device)
