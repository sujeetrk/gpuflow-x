import torch
import torch.nn as nn

from inference.cpu_backend import CPUBackend


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


def test_cpu_backend_prediction():

    model = DummyModel()
    backend = CPUBackend()

    backend.load_model(model)

    inputs = torch.randn(1, 10)

    result = backend.predict(inputs)

    assert result["device"] == "cpu"
    assert result["latency_ms"] >= 0
    assert tuple(result["output"].shape) == (1, 2)
