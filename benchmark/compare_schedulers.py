import time

from core.batching.batch_config import BatchConfig
from core.policy.adaptive_policy import AdaptivePolicy
from core.policy.ai_policy import AIPolicy
from core.queue.request import InferenceRequest
from core.scheduler.adaptive_scheduler import AdaptiveScheduler
from core.scheduler.ai_scheduler import AIScheduler
from inference.service import InferenceService


REQUEST_COUNT = 100


def run_scheduler(scheduler, requests):
    start_time = time.perf_counter()

    scheduler.start()

    for request in requests:
        scheduler.submit(request)

    deadline = time.perf_counter() + 15

    while scheduler.metrics()["requests_completed"] < len(requests):
        if time.perf_counter() >= deadline:
            break

        time.sleep(0.01)

    end_time = time.perf_counter()

    scheduler.stop()

    metrics = scheduler.metrics()
    metrics["runtime_ms"] = (end_time - start_time) * 1000

    return metrics


def create_requests():
    return [
        InferenceRequest(
            [(i + j) % 10 for j in range(10)]
        )
        for i in range(REQUEST_COUNT)
    ]


def main():
    adaptive_scheduler = AdaptiveScheduler(
        inference_service=InferenceService(),
        policy=AdaptivePolicy(),
        batch_config=BatchConfig(
            max_batch_size=4,
            max_batch_delay_ms=5
        ),
        max_queue_size=REQUEST_COUNT + 20,
    )

    ai_scheduler = AIScheduler(
        inference_service=InferenceService(),
        policy=AIPolicy(),
        batch_config=BatchConfig(
            max_batch_size=4,
            max_batch_delay_ms=5
        ),
        max_queue_size=REQUEST_COUNT + 20,
    )

    adaptive_requests = create_requests()
    ai_requests = create_requests()

    adaptive_results = run_scheduler(
        adaptive_scheduler,
        adaptive_requests,
    )

    ai_results = run_scheduler(
        ai_scheduler,
        ai_requests,
    )

    print("=== GPUFlow-X Scheduler Comparison ===")

    print("\nRule-Based Adaptive Scheduler")
    print("Requests:", adaptive_results["requests_submitted"])
    print("Completed:", adaptive_results["requests_completed"])
    print("Failed:", adaptive_results["requests_failed"])
    print("Batches:", adaptive_results["batches_executed"])
    print("Runtime (ms):", adaptive_results["runtime_ms"])
    print(
        "Final batch size:",
        adaptive_results["max_batch_size"],
    )
    print(
        "Final batch delay (ms):",
        adaptive_results["max_batch_delay_ms"],
    )

    print("\nAI Scheduler")
    print("Requests:", ai_results["requests_submitted"])
    print("Completed:", ai_results["requests_completed"])
    print("Failed:", ai_results["requests_failed"])
    print("Batches:", ai_results["batches_executed"])
    print("Runtime (ms):", ai_results["runtime_ms"])
    print(
        "Final batch size:",
        ai_results["max_batch_size"],
    )
    print(
        "Final batch delay (ms):",
        ai_results["max_batch_delay_ms"],
    )

    print("\nAI Decision:")
    print(ai_results["last_ai_decision"])


if __name__ == "__main__":
    main()
