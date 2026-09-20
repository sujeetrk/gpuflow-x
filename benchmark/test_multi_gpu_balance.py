import time

from core.scheduler.device_manager import GPUDeviceManager


def run_benchmark(gpu_count=4, request_count=10000):
    manager = GPUDeviceManager(device_count=gpu_count)
    assignments = []

    start = time.perf_counter()

    for _ in range(request_count):
        assignments.append(manager.select_device())

    assignment_time = time.perf_counter() - start
    loads = manager.load_snapshot()

    print("\n--- Multi-GPU Load-Balancing Benchmark ---")
    print(f"Simulated GPUs: {gpu_count}")
    print(f"Requests assigned: {request_count}")
    print(f"Assignment time: {assignment_time * 1000:.2f} ms")
    print(f"Assignment throughput: {request_count / assignment_time:.0f} req/s")
    print(f"Per-GPU assignments: {loads}")

    assert sum(loads.values()) == request_count
    assert max(loads.values()) - min(loads.values()) <= 1

    for device in assignments:
        manager.release_device(device)

    assert sum(manager.load_snapshot().values()) == 0

    print("Balance and release validation: PASSED")


if __name__ == "__main__":
    run_benchmark()
