"""CUDA inference backend for GPUFlow-X."""

import time

import torch

from inference.backend import InferenceBackend


class CUDABackend(InferenceBackend):
    """Run model inference on a specific CUDA device."""

    def __init__(self, device):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        self._device = torch.device(device)
        self.model = None

    def load_model(self, model):
        self.model = model.to(self._device)
        self.model.eval()

    def predict(self, inputs):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        inputs = inputs.to(self._device)
        start = time.perf_counter()

        with torch.no_grad():
            output = self.model(inputs)

        # Ensure GPU work is complete before measuring latency.
        torch.cuda.synchronize(self._device)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "output": output,
            "latency_ms": latency_ms,
            "device": self.device(),
        }

    def device(self):
        return str(self._device)
