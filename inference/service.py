import time
import uuid

import torch
import torch.nn as nn

from core.batching.batch import InferenceBatch
from core.scheduler.device_manager import GPUDeviceManager
from inference.cpu_backend import CPUBackend
from inference.cuda_backend import CUDABackend, GPUInferenceError


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

    def infer_batch(self, batch_values, request_ids=None):
        """Run a batch, retrying on CPU after a CUDA execution failure."""
        if not batch_values:
            raise ValueError("Cannot infer an empty batch.")

        if request_ids is None:
            request_ids = [
                str(uuid.uuid4()) for _ in batch_values
            ]
        else:
            request_ids = list(request_ids)

            if len(request_ids) != len(batch_values):
                raise ValueError(
                    "The number of request IDs must match "
                    "the number of batch inputs."
                )

            if any(not request_id for request_id in request_ids):
                raise ValueError("Request IDs cannot be empty.")

        # Validate inputs before selecting or quarantining any GPU.
        inputs = torch.tensor(batch_values, dtype=torch.float32)
        start_time = time.perf_counter()
        device = self.device_manager.select_device()

        try:
            backend = self.backends[device]

            try:
                result = backend.predict(inputs)
            except GPUInferenceError as exc:
                if device == "cpu":
                    raise

                self.device_manager.mark_unhealthy(device, str(exc))

                # Retry the same inputs using the existing CPU backend.
                result = self.backends["cpu"].predict(inputs)

            outputs = result["output"].cpu().tolist()
            end_time = time.perf_counter()

            return {
                "batch_size": len(batch_values),
                "request_ids": request_ids,
                "output": outputs,
                "inference_time_ms": result["latency_ms"],
                "total_latency_ms": (
                    end_time - start_time
                ) * 1000,
                "device": result["device"],
            }

        finally:
            self.device_manager.release_device(device)

    def infer_batch_object(self, batch: InferenceBatch):
        """Execute an InferenceBatch while preserving request IDs."""
        if batch.is_empty():
            raise ValueError("Cannot execute an empty batch.")

        request_ids = [
            request.request_id
            for request in batch.requests
        ]

        return self.infer_batch(
            batch.inputs,
            request_ids=request_ids
        )

    def infer(
        self,
        values,
        arrival_time=None,
        request_id=None
    ):
        """Run inference for a single request."""

        if arrival_time is None:
            arrival_time = time.perf_counter()

        result = self.infer_batch(
            [values],
            request_ids=(
                [request_id] if request_id is not None else None
            )
        )

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
