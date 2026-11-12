from unittest.mock import Mock

from inference.gpu_monitor import GPUHealthMonitor


def make_device_manager(devices=None):
    devices = devices or []

    manager = Mock()
    manager.devices = devices
    manager.metrics.return_value = {
        "available_gpus": len(devices),
        "active_assignments": 0,
        "device_loads": {
            device: 0 for device in devices
        },
        "cpu_fallback": not bool(devices),
    }

    return manager


def test_monitor_reports_cpu_fallback(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    monitor = GPUHealthMonitor(
        make_device_manager()
    )

    result = monitor.snapshot()

    assert result["status"] == "cpu_fallback"
    assert result["cuda_available"] is False
    assert result["gpu_count"] == 0
    assert result["cpu_fallback"] is True
    assert result["gpus"] == []


def test_monitor_reports_device_loads_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    manager = make_device_manager(
        ["cuda:0", "cuda:1"]
    )

    manager.metrics.return_value.update({
        "active_assignments": 2,
        "device_loads": {
            "cuda:0": 1,
            "cuda:1": 1,
        },
        "cpu_fallback": False,
    })

    result = GPUHealthMonitor(manager).snapshot()

    assert result["gpu_count"] == 2
    assert result["active_assignments"] == 2
    assert result["device_loads"] == {
        "cuda:0": 1,
        "cuda:1": 1,
    }

    assert result["gpus"] == []
    assert result["status"] == "cpu_fallback"


def test_monitor_does_not_invent_utilization(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    result = GPUHealthMonitor(
        make_device_manager()
    ).snapshot()

    assert result["gpus"] == []
    assert "monitoring_note" in result
