
import time
from pathlib import Path
from statistics import mean

from core.batching.batch_config import BatchConfig
from core.policy.adaptive_policy import AdaptivePolicy
from core.policy.ai_policy import AIPolicy

from core.queue.request import InferenceRequest

from core.scheduler.fifo_scheduler import FIFOScheduler
from core.scheduler.adaptive_scheduler import AdaptiveScheduler
from core.scheduler.ai_scheduler import AIScheduler

from inference.service import InferenceService


REQUEST_COUNT = 100
MAX_WAIT_SECONDS = 30
SUBMISSION_INTERVAL_MS = 1.0

# Same workload profile for each scheduler.
SLA_BUDGETS_MS = [50, 100, 250, 500]


def create_requests():
    requests = []

    for i in range(REQUEST_COUNT):
        sla = SLA_BUDGETS_MS[i % len(SLA_BUDGETS_MS)]

        requests.append(
            InferenceRequest(
                values=[(i + j) % 10 for j in range(10)],
                priority=(i % 5) + 1,
                sla_budget_ms=sla,
            )
        )

    return requests


def run_scheduler(name, scheduler, requests):
    scheduler.start()
    start_time = time.perf_counter()

    for request in requests:
        scheduler.submit(request)
        if SUBMISSION_INTERVAL_MS > 0:
            time.sleep(SUBMISSION_INTERVAL_MS / 1000.0)

    deadline = time.perf_counter() + MAX_WAIT_SECONDS

    while time.perf_counter() < deadline:
        metrics = scheduler.metrics()

        finished = (
            metrics["requests_completed"]
            + metrics["requests_failed"]
        )

        if finished >= len(requests):
            break

        time.sleep(0.005)

    end_time = time.perf_counter()
    scheduler.stop()

    metrics = scheduler.metrics()

    completed_requests = [
        request for request in requests
        if request.status == "completed"
    ]

    latencies = [
        request.total_latency_ms
        for request in completed_requests
        if request.total_latency_ms is not None
    ]

    sorted_latencies = sorted(latencies)

    p95_latency = (
        sorted_latencies[
            max(0, int(len(sorted_latencies) * 0.95) - 1)
        ]
        if sorted_latencies else 0.0
    )

    runtime_seconds = max(end_time - start_time, 0.001)

    deadline_misses = sum(
        1 for request in completed_requests
        if request.deadline_missed
    )

    results = {
        "scheduler": name,
        "submitted": metrics["requests_submitted"],
        "completed": metrics["requests_completed"],
        "failed": metrics["requests_failed"],
        "runtime_ms": runtime_seconds * 1000,
        "throughput_rps": (
            metrics["requests_completed"] / runtime_seconds
        ),
        "avg_latency_ms": mean(latencies) if latencies else 0.0,
        "p95_latency_ms": p95_latency,
        "deadline_misses": deadline_misses,
        "batches": metrics.get("batches_executed", "N/A"),
        "learning_updates": metrics.get("learning_statistics", {}).get(
            "decisions", 0
        ),
    }

    return results


def print_results(results):
    print(f"\n=== {results['scheduler']} ===")
    print("Submitted:", results["submitted"])
    print("Completed:", results["completed"])
    print("Failed:", results["failed"])
    print("Runtime (ms):", round(results["runtime_ms"], 2))
    print("Throughput (req/s):", round(results["throughput_rps"], 2))
    print("Average latency (ms):", round(results["avg_latency_ms"], 2))
    print("P95 latency (ms):", round(results["p95_latency_ms"], 2))
    print("Deadline misses:", results["deadline_misses"])
    print("Batches:", results["batches"])
    if "learning_updates" in results:
        print("Learning updates:", results["learning_updates"])


def main():
    fifo_scheduler = FIFOScheduler(
        inference_service=InferenceService(),
        max_queue_size=REQUEST_COUNT + 20,
    )

    adaptive_scheduler = AdaptiveScheduler(
        inference_service=InferenceService(),
        policy=AdaptivePolicy(),
        batch_config=BatchConfig(
            max_batch_size=4,
            max_batch_delay_ms=5,
        ),
        max_queue_size=REQUEST_COUNT + 20,
    )

    ai_scheduler = AIScheduler(
        inference_service=InferenceService(),
        policy=AIPolicy(),
        batch_config=BatchConfig(
            max_batch_size=4,
            max_batch_delay_ms=5,
        ),
        max_queue_size=REQUEST_COUNT + 20,
        learning_enabled=False,
        policy_path="benchmark/phase9_policy.json",
    )

    # Start each learning benchmark from a clean policy file so repeated runs
    # do not silently compare different persisted training histories.
    learned_policy_path = Path("/tmp/gpuflow_x_phase9_learned_policy.json")
    learned_policy_path.unlink(missing_ok=True)
    learning_ai_scheduler = AIScheduler(
        inference_service=InferenceService(),
        policy=AIPolicy(),
        batch_config=BatchConfig(
            max_batch_size=4,
            max_batch_delay_ms=5,
        ),
        max_queue_size=REQUEST_COUNT + 20,
        learning_enabled=True,
        policy_path=str(learned_policy_path),
    )

    schedulers = [
        ("FIFO Scheduler", fifo_scheduler),
        ("Adaptive Scheduler", adaptive_scheduler),
        ("AI Priority + SLA (fixed)", ai_scheduler),
        ("AI Priority + SLA (learning)", learning_ai_scheduler),
    ]

    all_results = []

    print("GPUFlow-X Phase 9 Scheduler Benchmark")
    print("Learning-enabled AI starts from a clean policy file each run.")
    print("Workload:", REQUEST_COUNT, "requests per scheduler")
    print("SLA budgets (ms):", SLA_BUDGETS_MS)
    print("Submission pacing (ms/request):", SUBMISSION_INTERVAL_MS)
    print("Device: CPU")
    print("Note: Each scheduler receives a fresh equivalent workload.")

    for name, scheduler in schedulers:
        requests = create_requests()
        results = run_scheduler(name, scheduler, requests)
        all_results.append(results)
        print_results(results)

    print("\n=== PHASE 9 COMPARISON SUMMARY ===")
    print(
        f"{'Scheduler':<28}"
        f"{'Completed':>10}"
        f"{'Throughput':>14}"
        f"{'Avg Latency':>14}"
        f"{'P95 Latency':>14}"
        f"{'Misses':>10}"
        f"{'LearnUpdates':>14}"
    )

    for result in all_results:
        print(
            f"{result['scheduler']:<28}"
            f"{result['completed']:>10}"
            f"{result['throughput_rps']:>14.2f}"
            f"{result['avg_latency_ms']:>14.2f}"
            f"{result['p95_latency_ms']:>14.2f}"
            f"{result['deadline_misses']:>10}"
            f"{result['learning_updates']:>14}"
        )


if __name__ == "__main__":
    main()
