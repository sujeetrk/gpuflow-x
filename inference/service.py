import time
import uuid

import torch
import torch.nn as nn

from inference.cpu_backend import CPUBackend


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
    def __init__(self):
        self.backend = CPUBackend()

        model = InferenceModel()
        self.backend.load_model(model)

    def infer_batch(self, batch_values):
        """
        Run inference on multiple requests as one batch.
        """

        batch_size = len(batch_values)

        request_ids = [
            str(uuid.uuid4())
            for _ in range(batch_size)
        ]

        arrival_time = time.perf_counter()

        inputs = torch.tensor(
            batch_values,
            dtype=torch.float32
        )

        start_time = time.perf_counter()

        result = self.backend.predict(inputs)

        end_time = time.perf_counter()

        outputs = result["output"].tolist()

        total_latency_ms = (
            end_time - arrival_time
        ) * 1000

        return {
            "batch_size": batch_size,
            "request_ids": request_ids,
            "output": outputs,
            "inference_time_ms": result["latency_ms"],
            "total_latency_ms": total_latency_ms,
            "device": result["device"]
        }

    def infer(self, values, arrival_time=None):
        """
        Run inference for a single request.
        """

        if arrival_time is None:
            arrival_time = time.perf_counter()

        result = self.infer_batch([values])

        return {
            "request_id": result["request_ids"][0],
            "arrival_time": arrival_time,
            "start_time": arrival_time,
            "end_time": arrival_time
                + result["total_latency_ms"] / 1000,
            "queue_time_ms": 0.0,
            "inference_time_ms": result["inference_time_ms"],
            "total_latency_ms": result["total_latency_ms"],
            "output": result["output"][0],
            "device": result["device"]
        }
