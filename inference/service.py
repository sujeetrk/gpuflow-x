import time
import uuid

import torch
import torch.nn as nn

from core.batching.batch import InferenceBatch
from core.scheduler.device_manager import GPUDeviceManager
from inference.cpu_backend import CPUBackend
from inference.cuda_backend import CUDABackend


class InferenceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


class InferenceService:
    """Multi-device inference with load-balanced GPU placement."""

    def __init__(self):
        self.device_manager = GPUDeviceManager()
        self.backends = {}

        # Keep CPU available as the safe fallback.
        cpu_backend = CPUBackend()
        cpu_backend.load_model(InferenceModel())
        self.backends["cpu"] = cpu_backend

        # Each detected GPU gets its own model/backend.
        for device in self.device_manager.devices:
            backend = CUDABackend(device)
            backend.load_model(InferenceModel())
            self.backends[device] = backend

    def infer_batch(self, batch_values):
        """Run a batch on the least-loaded available GPU or CPU."""
        if not batch_values:
            raise ValueError("Cannot infer an empty batch.")

        device = self.device_manager.select_device()

        try:
            backend = self.backends[device]
            inputs = torch.tensor(batch_values, dtype=torch.float32)

            arrival_time = time.perf_counter()
            result = backend.predict(inputs)
            outputs = result["output"].cpu().tolist()
            end_time = time.perf_counter()

            return {
                "batch_size": len(batch_values),
                "request_ids": [
                    str(uuid.uuid4()) for _ in batch_values
                ],
                "output": outputs,
                "inference_time_ms": result["latency_ms"],
                "total_latency_ms": (
                    end_time - arrival_time
                ) * 1000,
                "device": result["device"],
            }
        finally:
            self.device_manager.release_device(device)

    def infer_batch_object(self, batch: InferenceBatch):
        """Execute an InferenceBatch object."""
        if batch.is_empty():
            raise ValueError("Cannot execute an empty batch.")

        return self.infer_batch(batch.inputs)

    def infer(self, values, arrival_time=None):
        """Run inference for a single request."""
        if arrival_time is None:
            arrival_time = time.perf_counter()

        result = self.infer_batch([values])

        return {
            "request_id": result["request_ids"][0],
            "arrival_time": arrival_time,
            "start_time": arrival_time,
            "end_time": (
                arrival_time + result["total_latency_ms"] / 1000
            ),
            "queue_time_ms": 0.0,
            "inference_time_ms": result["inference_time_ms"],
            "total_latency_ms": result["total_latency_ms"],
            "output": result["output"][0],
            "device": result["device"],
        }
