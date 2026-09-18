import torch
import torch.nn as nn

from inference.cpu_backend import CPUBackend


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


model = TestModel()

backend = CPUBackend()

backend.load_model(model)

inputs = torch.randn(1, 10)

result = backend.predict(inputs)

print("Device:", result["device"])
print("Latency (ms):", result["latency_ms"])
print("Output shape:", tuple(result["output"].shape))
