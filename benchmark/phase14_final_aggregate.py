import csv
import statistics
from pathlib import Path

from benchmark.phase14_experiments import (
    WORKLOADS,
    create_requests,
    create_schedulers,
    run_scheduler,
)


REPETITIONS = 3

RAW_PATH = Path("benchmark/phase14_results_all.csv")
SUMMARY_PATH = Path("benchmark/phase14_summary.csv")


def main():
    raw_results = []

    print("=== GPUFlow-X Phase 14 FINAL AGGREGATED EVALUATION ===")
    print("Device: CPU")
    print(f"Repetitions: {REPETITIONS}")
    print(f"Workloads: {len(WORKLOADS)}")
    print("Schedulers: 4")
    print("Expected records:", REPETITIONS * len(WORKLOADS) * 4)
    print()

    for repetition in range(1, REPETITIONS + 1):
        print("=" * 90)
        print(f"REPETITION {repetition}/{REPETITIONS}")
        print("=" * 90)

        for workload_name, request_count, interval_ms in WORKLOADS:
            print(
                f"\n--- {workload_name} | "
                f"{request_count} requests | "
                f"{interval_ms} ms/request ---"
            )

            for name, scheduler in create_schedulers(request_count):
                requests = create_requests(request_count)

                result = run_scheduler(
                    name,
                    scheduler,
                    requests,
                    interval_ms,
                )

                result["repetition"] = repetition
                result["workload"] = workload_name
                result["request_count"] = request_count
                result["submission_interval_ms"] = interval_ms

                raw_results.append(result)

                print(
                    f"{name:<32} "
                    f"Throughput={result['throughput_rps']:>9.2f} "
                    f"Avg={result['avg_latency_ms']:>8.2f} ms "
                    f"P95={result['p95_latency_ms']:>8.2f} ms "
                    f"P99={result['p99_latency_ms']:>8.2f} ms "
                    f"Misses={result['deadline_misses']:>3}"
                )

    raw_fields = [
        "repetition",
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

    with RAW_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_results)

    groups = {}

    for row in raw_results:
        key = (
            row["workload"],
            row["submission_interval_ms"],
            row["scheduler"],
        )
        groups.setdefault(key, []).append(row)

    summary = []

    metrics = [
        "throughput_rps",
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "deadline_misses",
        "completed",
        "failed",
        "batches",
        "learning_updates",
    ]

    for (workload, interval, scheduler), rows in groups.items():
        result = {
            "workload": workload,
            "submission_interval_ms": interval,
            "scheduler": scheduler,
            "runs": len(rows),
        }

        for metric in metrics:
            numeric_values = []

            for row in rows:
                try:
                    numeric_values.append(float(row[metric]))
                except (ValueError, TypeError):
                    # Some schedulers legitimately report N/A
                    # for metrics they do not expose.
                    continue

            if numeric_values:
                result[f"{metric}_mean"] = statistics.mean(numeric_values)
                result[f"{metric}_std"] = (
                    statistics.stdev(numeric_values)
                    if len(numeric_values) > 1
                    else 0.0
                )
            else:
                result[f"{metric}_mean"] = "N/A"
                result[f"{metric}_std"] = "N/A"

        summary.append(result)

    summary_fields = ["workload", "submission_interval_ms", "scheduler", "runs"]

    for metric in metrics:
        summary_fields.append(f"{metric}_mean")
        summary_fields.append(f"{metric}_std")

    with SUMMARY_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    print()
    print("=" * 90)
    print("FINAL AGGREGATED SUMMARY")
    print("=" * 90)

    for row in summary:
        print(
            f"{row['workload']:<12} | "
            f"{row['scheduler']:<32} | "
            f"Throughput: "
            f"{row['throughput_rps_mean']:.2f} ± "
            f"{row['throughput_rps_std']:.2f} | "
            f"Avg latency: "
            f"{row['avg_latency_ms_mean']:.2f} ± "
            f"{row['avg_latency_ms_std']:.2f} ms | "
            f"P95: "
            f"{row['p95_latency_ms_mean']:.2f} ± "
            f"{row['p95_latency_ms_std']:.2f} ms"
        )

    print()
    print("Raw results:", RAW_PATH)
    print("Aggregated results:", SUMMARY_PATH)
    print("Total raw records:", len(raw_results))


if __name__ == "__main__":
    main()
