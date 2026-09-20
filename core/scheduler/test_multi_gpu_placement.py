from core.scheduler.device_manager import GPUDeviceManager


def test_round_robin_when_gpu_loads_are_equal():
    manager = GPUDeviceManager(device_count=3)

    devices = [manager.select_device() for _ in range(6)]

    assert devices == [
        "cuda:0", "cuda:1", "cuda:2",
        "cuda:0", "cuda:1", "cuda:2",
    ]


def test_least_loaded_device_receives_next_assignment():
    manager = GPUDeviceManager(device_count=2)

    manager.select_device()
    manager.select_device()
    manager.select_device()

    loads = manager.load_snapshot()
    assert loads == {"cuda:0": 2, "cuda:1": 1}


def test_released_device_can_be_reused():
    manager = GPUDeviceManager(device_count=2)

    first = manager.select_device()
    manager.select_device()
    manager.release_device(first)

    assert manager.select_device() == first


def test_load_returns_to_zero_after_release():
    manager = GPUDeviceManager(device_count=2)

    device = manager.select_device()
    manager.release_device(device)

    assert sum(manager.load_snapshot().values()) == 0


def test_cpu_fallback_reports_no_gpu_assignments():
    manager = GPUDeviceManager(device_count=0)

    assert manager.select_device() == "cpu"
    assert manager.metrics()["active_assignments"] == 0
    assert manager.metrics()["cpu_fallback"] is True
