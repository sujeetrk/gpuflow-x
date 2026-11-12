from fastapi.testclient import TestClient

import server.api.main as api_module


def test_gpu_health_endpoint_returns_cpu_fallback(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    response = TestClient(api_module.app).get("/gpu/health")

    assert response.status_code == 200

    data = response.json()

    assert data["cuda_available"] is False
    assert data["status"] == "cpu_fallback"
    assert data["gpu_count"] == 0
    assert data["cpu_fallback"] is True


def test_gpu_health_endpoint_includes_device_assignments():
    response = TestClient(api_module.app).get("/gpu/health")

    assert response.status_code == 200

    data = response.json()

    assert "active_assignments" in data
    assert "device_loads" in data
    assert "gpus" in data
