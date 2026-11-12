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


def test_gpu_failure_quarantines_device_and_retries_on_cpu():
    from inference.cuda_backend import GPUInferenceError

    class GPUFailureBackend:
        def predict(self, inputs):
            raise GPUInferenceError("Simulated CUDA device failure")

    service = make_multi_gpu_service()
    service.backends["cuda:0"] = GPUFailureBackend()

    result = service.infer_batch(
        [[0.1] * 10],
        request_ids=["request-fallback-1"],
    )

    assert result["device"] == "cpu"
    assert result["request_ids"] == ["request-fallback-1"]
    assert result["batch_size"] == 1
    assert service.device_manager.metrics()["unhealthy_devices"] == {
        "cuda:0": "Simulated CUDA device failure"
    }
    assert service.device_manager.load_snapshot() == {
        "cuda:0": 0,
        "cuda:1": 0,
    }

    # The remaining healthy GPU should still receive work.
    next_result = service.infer_batch([[0.2] * 10])
    assert next_result["device"] == "cuda:1"


def test_all_failed_gpus_result_in_cpu_fallback():
    from inference.cuda_backend import GPUInferenceError

    class GPUFailureBackend:
        def predict(self, inputs):
            raise GPUInferenceError("Simulated GPU failure")

    service = make_multi_gpu_service()
    service.backends["cuda:0"] = GPUFailureBackend()
    service.backends["cuda:1"] = GPUFailureBackend()

    first = service.infer_batch([[0.1] * 10])
    second = service.infer_batch([[0.2] * 10])
    third = service.infer_batch([[0.3] * 10])

    assert first["device"] == "cpu"
    assert second["device"] == "cpu"
    assert third["device"] == "cpu"
    assert service.device_manager.metrics()["cpu_fallback"] is True
    assert service.device_manager.metrics()["available_gpus"] == 0
