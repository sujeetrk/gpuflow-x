import pytest

from core.scheduler.device_manager import GPUDeviceManager


def test_cpu_fallback_when_no_gpu():
    manager = GPUDeviceManager(device_count=0)

    assert manager.select_device() == "cpu"
    assert manager.metrics()["cpu_fallback"] is True


def test_gpu_assignments_are_balanced():
    manager = GPUDeviceManager(device_count=2)

    first = manager.select_device()
    second = manager.select_device()

    assert first == "cuda:0"
    assert second == "cuda:1"
    assert manager.load_snapshot() == {
        "cuda:0": 1,
        "cuda:1": 1,
    }


def test_least_loaded_gpu_is_selected():
    manager = GPUDeviceManager(device_count=2)

    manager.select_device()
    manager.select_device()
    manager.select_device()

    loads = manager.load_snapshot()
    assert sorted(loads.values()) == [1, 2]


def test_release_decreases_load():
    manager = GPUDeviceManager(device_count=1)

    device = manager.select_device()
    manager.release_device(device)

    assert manager.load_snapshot()["cuda:0"] == 0


def test_release_unknown_device_fails():
    manager = GPUDeviceManager(device_count=1)

    with pytest.raises(ValueError):
        manager.release_device("cuda:99")


def test_release_unassigned_device_fails():
    manager = GPUDeviceManager(device_count=1)

    with pytest.raises(ValueError):
        manager.release_device("cuda:0")


def test_negative_device_count_fails():
    with pytest.raises(ValueError):
        GPUDeviceManager(device_count=-1)


def test_cpu_release_is_safe():
    manager = GPUDeviceManager(device_count=0)
    manager.release_device("cpu")
