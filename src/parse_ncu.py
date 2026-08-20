import csv
import sys
from collections import defaultdict
import statistics

METRICS = [
    "Compute (SM) Throughput",
    "Memory Throughput",
    "DRAM Throughput",
    "Achieved Occupancy",
    "Warp Cycles Per Issued Instruction",
    "L1/TEX Hit Rate",
    "L2 Hit Rate",
]

def parse_ncu_csv(filepath):
    # metric_name -> list of float values per kernel run
    data = defaultdict(list)

    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ID", "").strip() == "0":
                continue

            metric = row.get("Metric Name", "").strip()
            section = row.get("Section Name", "").strip()
            unit = row.get("Metric Unit", "").strip()
            value_str = row.get("Metric Value", "").strip()

            if metric in METRICS and value_str:
                # NCU reports Memory Throughput both as % and Gbyte/s.
                if metric == "Memory Throughput" and (
                    section != "GPU Speed Of Light Throughput" or unit != "%"
                ):
                    continue

                try:
                    # NCU sometimes uses commas as thousand separators
                    value = float(value_str.replace(",", ""))
                    data[metric].append(value)
                except ValueError:
                    pass

    return data

def print_summary(data):
    print(f"{'Metric':<40} | {'Runs':>4} | {'Min':>8} | {'Max':>8} | {'Avg':>8} | {'StdDev':>8}")
    print("-" * 90)

    for metric in METRICS:
        values = data.get(metric, [])
        if not values:
            print(f"{metric:<40} | {'n/a':>4}")
            continue

        avg = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{metric:<40} | {len(values):>4} | {min(values):>8.2f} | {max(values):>8.2f} | {avg:>8.2f} | {std:>8.3f}")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "ncu.csv"
    data = parse_ncu_csv(filepath)
    print(f"\nNCU Report: {filepath}")
    print_summary(data)
    print()
