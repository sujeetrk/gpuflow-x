import csv
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


SLA_BUDGETS_MS = [50, 100, 250, 500]

WORKLOADS = [
    ("Low Load", 100, 5.0),
    ("Medium Load", 100, 1.0),
    ("Burst Load", 100, 0.0),
]

MAX_WAIT_SECONDS = 30


def create_requests(request_count):
    return [
        InferenceRequest(
            values=[(i + j) % 10 for j in range(10)],
            priority=(i % 5) + 1,
            sla_budget_ms=SLA_BUDGETS_MS[i % len(SLA_BUDGETS_MS)],
        )
        for i in range(request_count)
    ]


def create_schedulers(request_count):
    return [
        (
            "FIFO Scheduler",
            FIFOScheduler(
                inference_service=InferenceService(),
                max_queue_size=request_count + 20,
            ),
        ),
        (
            "Adaptive Scheduler",
            AdaptiveScheduler(
                inference_service=InferenceService(),
                policy=AdaptivePolicy(),
                batch_config=BatchConfig(
                    max_batch_size=4,
                    max_batch_delay_ms=5,
                ),
                max_queue_size=request_count + 20,
            ),
        ),
        (
            "AI Priority + SLA (fixed)",
            AIScheduler(
                inference_service=InferenceService(),
                policy=AIPolicy(),
                batch_config=BatchConfig(
                    max_batch_size=4,
                    max_batch_delay_ms=5,
                ),
                max_queue_size=request_count + 20,
                learning_enabled=False,
                policy_path="benchmark/phase9_policy.json",
            ),
        ),
        (
            "AI Priority + SLA (learning)",
            AIScheduler(
                inference_service=InferenceService(),
                policy=AIPolicy(),
                batch_config=BatchConfig(
                    max_batch_size=4,
                    max_batch_delay_ms=5,
                ),
                max_queue_size=request_count + 20,
                learning_enabled=True,
                policy_path="/tmp/gpuflow_x_phase14_learned_policy.json",
            ),
        ),
    ]


def run_scheduler(name, scheduler, requests, submission_interval_ms):
    scheduler.start()
    start_time = time.perf_counter()

    for request in requests:
        scheduler.submit(request)
        if submission_interval_ms > 0:
            time.sleep(submission_interval_ms / 1000.0)

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
    metrics = scheduler.metrics()

    completed_requests = [
        request
        for request in requests
        if request.status == "completed"
    ]

    latencies = [
        request.total_latency_ms
        for request in completed_requests
        if request.total_latency_ms is not None
    ]

    sorted_latencies = sorted(latencies)

    def percentile(values, p):
        if not values:
            return 0.0
        index = max(0, int(len(values) * p) - 1)
        return values[index]

    runtime_seconds = max(end_time - start_time, 0.001)

    deadline_misses = sum(
        1
        for request in completed_requests
        if request.deadline_missed
    )

    result = {
        "scheduler": name,
        "submitted": metrics["requests_submitted"],
        "completed": metrics["requests_completed"],
        "failed": metrics["requests_failed"],
        "runtime_ms": runtime_seconds * 1000,
        "throughput_rps": metrics["requests_completed"] / runtime_seconds,
        "avg_latency_ms": mean(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(sorted_latencies, 0.95),
        "p99_latency_ms": percentile(sorted_latencies, 0.99),
        "deadline_misses": deadline_misses,
        "batches": metrics.get("batches_executed", "N/A"),
        "learning_updates": metrics.get(
            "learning_statistics", {}
        ).get("decisions", 0),
    }

    scheduler.stop()
    return result


def main():
    output_path = Path("benchmark/phase14_results.csv")

    fieldnames = [
        "workload",
        "request_count",
        "submission_interval_ms",
        "scheduler",
        "submitted",
        "completed",
        "failed",
        "runtime_ms",
        "throughput_rps",
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "deadline_misses",
        "batches",
        "learning_updates",
    ]

    all_results = []

    print("=== GPUFlow-X Phase 14 Final Workload Evaluation ===")
    print("Device: CPU")
    print("SLA budgets (ms):", SLA_BUDGETS_MS)
    print()

    for workload_name, request_count, interval_ms in WORKLOADS:
        print("=" * 80)
        print(
            f"{workload_name}: "
            f"{request_count} requests, "
            f"{interval_ms} ms/request"
        )
        print("=" * 80)

        for name, scheduler in create_schedulers(request_count):
            if "learning" in name.lower():
                Path(
                    "/tmp/gpuflow_x_phase14_learned_policy.json"
                ).unlink(missing_ok=True)

            requests = create_requests(request_count)

            result = run_scheduler(
                name,
                scheduler,
                requests,
                interval_ms,
            )

            result["workload"] = workload_name
            result["request_count"] = request_count
            result["submission_interval_ms"] = interval_ms

            all_results.append(result)

            print(
                f"{name:<32} "
                f"Throughput={result['throughput_rps']:>8.2f} "
                f"Avg={result['avg_latency_ms']:>8.2f} ms "
                f"P95={result['p95_latency_ms']:>8.2f} ms "
                f"P99={result['p99_latency_ms']:>8.2f} ms "
                f"Misses={result['deadline_misses']:>3}"
            )

        print()

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print(f"Total experiment records: {len(all_results)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
