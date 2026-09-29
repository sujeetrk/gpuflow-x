import csv
from pathlib import Path

import matplotlib.pyplot as plt


SUMMARY = Path("benchmark/phase14_summary.csv")
OUT = Path("benchmark/phase14_artifacts")
OUT.mkdir(exist_ok=True)


def load_data():
    with SUMMARY.open() as f:
        return list(csv.DictReader(f))


def plot_metric(rows, metric, std_metric, ylabel, filename, title):
    workloads = ["Low Load", "Medium Load", "Burst Load"]
    schedulers = [
        "FIFO Scheduler",
        "Adaptive Scheduler",
        "AI Priority + SLA (fixed)",
        "AI Priority + SLA (learning)",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = list(range(len(workloads)))
    width = 0.18

    for i, scheduler in enumerate(schedulers):
        values = []
        errors = []

        for workload in workloads:
            row = next(
                r for r in rows
                if r["workload"] == workload
                and r["scheduler"] == scheduler
            )

            values.append(float(row[f"{metric}_mean"]))
            errors.append(float(row[f"{std_metric}_std"]))

        positions = [
            value + (i - 1.5) * width
            for value in x
        ]

        ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=scheduler,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


def write_report(rows):
    report = OUT / "PHASE14_EVALUATION.md"

    with report.open("w") as f:
        f.write("# GPUFlow-X — Phase 14 Final Experiments & Evaluation\n\n")

        f.write("## Experimental Setup\n\n")
        f.write("- Device: CPU development environment\n")
        f.write("- Workloads: Low Load, Medium Load, Burst Load\n")
        f.write("- Requests per workload/scheduler: 100\n")
        f.write("- Repetitions: 3\n")
        f.write("- SLA budgets: 50, 100, 250, 500 ms\n")
        f.write("- Schedulers: FIFO, Adaptive, AI Fixed, AI Learning\n")
        f.write("- Total measurements: 36\n\n")

        f.write("## Results\n\n")
        f.write("| Workload | Scheduler | Throughput (req/s) | Avg Latency (ms) | P95 (ms) | P99 (ms) | Deadline Misses |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")

        for row in rows:
            f.write(
                f"| {row['workload']} | "
                f"{row['scheduler']} | "
                f"{float(row['throughput_rps_mean']):.2f} ± {float(row['throughput_rps_std']):.2f} | "
                f"{float(row['avg_latency_ms_mean']):.2f} ± {float(row['avg_latency_ms_std']):.2f} | "
                f"{float(row['p95_latency_ms_mean']):.2f} ± {float(row['p95_latency_ms_std']):.2f} | "
                f"{float(row['p99_latency_ms_mean']):.2f} ± {float(row['p99_latency_ms_std']):.2f} | "
                f"{float(row['deadline_misses_mean']):.2f} ± {float(row['deadline_misses_std']):.2f} |\n"
            )

        f.write("\n## Observations\n\n")

        f.write(
            "1. The experiments show workload-dependent scheduler behavior rather "
            "than a single policy dominating every metric.\n\n"
        )

        f.write(
            "2. Under the medium-load workload, FIFO produced the highest measured "
            "throughput and lowest average latency in this CPU experiment.\n\n"
        )

        f.write(
            "3. Under burst load, FIFO produced substantially higher measured "
            "throughput, while the AI policies operated at lower throughput with "
            "different latency characteristics.\n\n"
        )

        f.write(
            "4. The learning-enabled AI scheduler did not consistently outperform "
            "the fixed AI policy. Under burst load it showed slightly lower average "
            "and P95 latency, while under medium load its measured throughput was lower.\n\n"
        )

        f.write(
            "5. Low-load execution produced substantially higher latency and many "
            "deadline misses under the selected SLA configuration, demonstrating "
            "that workload arrival characteristics strongly affect scheduler behavior.\n\n"
        )

        f.write("## Limitations\n\n")
        f.write(
            "These measurements were collected on a CPU development environment. "
            "They should not be interpreted as NVIDIA GPU performance benchmarks. "
            "The project contains an NVIDIA CUDA deployment configuration, but "
            "physical NVIDIA GPU execution was not available on the development laptop.\n\n"
        )

        f.write(
            "The benchmark therefore evaluates scheduler behavior, batching, "
            "latency, throughput and learning behavior under controlled CPU "
            "experiments rather than claiming physical GPU acceleration results.\n"
        )

        print(f"Report written to {report}")


def main():
    rows = load_data()

    plot_metric(
        rows,
        "throughput_rps",
        "throughput_rps",
        "Throughput (requests/s)",
        "throughput_comparison.png",
        "GPUFlow-X Throughput Comparison",
    )

    plot_metric(
        rows,
        "avg_latency_ms",
        "avg_latency_ms",
        "Average Latency (ms)",
        "average_latency_comparison.png",
        "GPUFlow-X Average Latency Comparison",
    )

    plot_metric(
        rows,
        "p95_latency_ms",
        "p95_latency_ms",
        "P95 Latency (ms)",
        "p95_latency_comparison.png",
        "GPUFlow-X P95 Latency Comparison",
    )

    write_report(rows)

    print()
    print("Phase 14 artifacts:")
    for path in sorted(OUT.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
