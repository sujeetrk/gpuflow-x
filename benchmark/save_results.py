import csv
import json
from pathlib import Path


def save_results(results):
    output_dir = Path("benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"

    # Save JSON
    with open(json_path, "w") as file:
        json.dump(results, file, indent=4)

    # Save CSV
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(request_count=100)
    results = runner.run()

    save_results(results)
