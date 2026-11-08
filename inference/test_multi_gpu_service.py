import torch
import pytest

from inference.service import InferenceService
from core.scheduler.device_manager import GPUDeviceManager


class FakeBackend:
    def __init__(self, device):
        self._device = device

    def predict(self, inputs):
        return {
            "output": inputs[:, :2],
            "latency_ms": 1.0,
            "device": self._device,
        }


class FailingBackend:
    def predict(self, inputs):
        raise RuntimeError("Simulated inference failure")


def make_multi_gpu_service():
    service = InferenceService()

    # Simulate two GPUs without requiring CUDA hardware.
    service.device_manager = GPUDeviceManager(device_count=2)
    service.backends["cuda:0"] = FakeBackend("cuda:0")
    service.backends["cuda:1"] = FakeBackend("cuda:1")

    return service


def test_inference_distributes_batches_across_gpus():
    service = make_multi_gpu_service()

    first = service.infer_batch([[0.1] * 10])
    second = service.infer_batch([[0.2] * 10])

    assert first["device"] == "cuda:0"
    assert second["device"] == "cuda:1"


def test_inference_releases_gpu_assignment():
    service = make_multi_gpu_service()

    service.infer_batch([[0.1] * 10])

    assert service.device_manager.load_snapshot() == {
        "cuda:0": 0,
        "cuda:1": 0,
    }


def test_failed_inference_still_releases_gpu():
    service = make_multi_gpu_service()
    service.backends["cuda:0"] = FailingBackend()

    with pytest.raises(RuntimeError, match="Simulated inference failure"):
        service.infer_batch([[0.1] * 10])

    assert service.device_manager.load_snapshot() == {
        "cuda:0": 0,
        "cuda:1": 0,
    }


def test_inference_output_shape_with_simulated_gpu():
    service = make_multi_gpu_service()

    result = service.infer_batch([
        [0.1] * 10,
        [0.2] * 10,
        [0.3] * 10,
    ])

    assert result["batch_size"] == 3
    assert len(result["output"]) == 3
    assert result["device"].startswith("cuda:")


def test_inference_batch_object_preserves_original_request_ids():
    from core.batching.batch import InferenceBatch
    from core.queue.request import InferenceRequest

    service = make_multi_gpu_service()

    first = InferenceRequest(values=[0.1] * 10)
    second = InferenceRequest(values=[0.2] * 10)

    batch = InferenceBatch(requests=[first, second])

    result = service.infer_batch_object(batch)

    assert result["request_ids"] == [
        first.request_id,
        second.request_id,
    ]

    assert len(result["output"]) == 2
    assert result["batch_size"] == 2
    assert result["device"] in ("cuda:0", "cuda:1")
