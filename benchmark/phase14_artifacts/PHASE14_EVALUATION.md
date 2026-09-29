# GPUFlow-X — Phase 14 Final Experiments & Evaluation

## Experimental Setup

- Device: CPU development environment
- Workloads: Low Load, Medium Load, Burst Load
- Requests per workload/scheduler: 100
- Repetitions: 3
- SLA budgets: 50, 100, 250, 500 ms
- Schedulers: FIFO, Adaptive, AI Fixed, AI Learning
- Total measurements: 36

## Results

| Workload | Scheduler | Throughput (req/s) | Avg Latency (ms) | P95 (ms) | P99 (ms) | Deadline Misses |
|---|---|---:|---:|---:|---:|---:|
| Low Load | FIFO Scheduler | 196.46 ± 0.07 | 253.14 ± 0.72 | 479.06 ± 0.18 | 499.43 ± 0.20 | 56.00 ± 0.00 |
| Low Load | Adaptive Scheduler | 195.95 ± 0.27 | 254.49 ± 0.30 | 480.45 ± 0.62 | 500.87 ± 0.69 | 56.00 ± 0.00 |
| Low Load | AI Priority + SLA (fixed) | 192.94 ± 0.28 | 266.61 ± 0.31 | 487.65 ± 4.81 | 504.97 ± 1.81 | 58.00 ± 1.73 |
| Low Load | AI Priority + SLA (learning) | 192.84 ± 1.35 | 269.10 ± 2.31 | 492.72 ± 1.92 | 513.36 ± 2.02 | 58.33 ± 2.31 |
| Medium Load | FIFO Scheduler | 921.01 ± 0.80 | 54.55 ± 0.43 | 102.60 ± 0.08 | 106.88 ± 0.06 | 15.00 ± 0.00 |
| Medium Load | Adaptive Scheduler | 871.19 ± 1.08 | 57.13 ± 0.16 | 106.10 ± 0.08 | 113.30 ± 0.13 | 15.00 ± 0.00 |
| Medium Load | AI Priority + SLA (fixed) | 780.81 ± 50.86 | 71.45 ± 4.01 | 124.26 ± 8.56 | 124.50 ± 8.83 | 22.00 ± 2.00 |
| Medium Load | AI Priority + SLA (learning) | 675.54 ± 95.52 | 69.70 ± 0.97 | 123.37 ± 0.88 | 140.58 ± 16.76 | 21.33 ± 0.58 |
| Burst Load | FIFO Scheduler | 16135.55 ± 5642.46 | 2.64 ± 1.08 | 4.01 ± 1.77 | 4.10 ± 1.79 | 0.00 ± 0.00 |
| Burst Load | Adaptive Scheduler | 4510.96 ± 556.91 | 4.57 ± 0.38 | 20.86 ± 0.49 | 20.96 ± 0.46 | 0.00 ± 0.00 |
| Burst Load | AI Priority + SLA (fixed) | 6397.39 ± 132.57 | 13.44 ± 0.27 | 14.76 ± 0.35 | 14.79 ± 0.36 | 0.00 ± 0.00 |
| Burst Load | AI Priority + SLA (learning) | 6440.01 ± 28.51 | 12.81 ± 0.25 | 13.86 ± 0.38 | 13.89 ± 0.39 | 0.00 ± 0.00 |

## Observations

1. The experiments show workload-dependent scheduler behavior rather than a single policy dominating every metric.

2. Under the medium-load workload, FIFO produced the highest measured throughput and lowest average latency in this CPU experiment.

3. Under burst load, FIFO produced substantially higher measured throughput, while the AI policies operated at lower throughput with different latency characteristics.

4. The learning-enabled AI scheduler did not consistently outperform the fixed AI policy. Under burst load it showed slightly lower average and P95 latency, while under medium load its measured throughput was lower.

5. Low-load execution produced substantially higher latency and many deadline misses under the selected SLA configuration, demonstrating that workload arrival characteristics strongly affect scheduler behavior.

## Limitations

These measurements were collected on a CPU development environment. They should not be interpreted as NVIDIA GPU performance benchmarks. The project contains an NVIDIA CUDA deployment configuration, but physical NVIDIA GPU execution was not available on the development laptop.

The benchmark therefore evaluates scheduler behavior, batching, latency, throughput and learning behavior under controlled CPU experiments rather than claiming physical GPU acceleration results.
