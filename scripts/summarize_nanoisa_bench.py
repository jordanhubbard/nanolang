#!/usr/bin/env python3
"""Summarize repeated NanoISA benchmark samples without external packages."""

from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def command_output(*command: str) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_nanoisa_bench.py RESULTS_DIR", file=sys.stderr)
        return 2

    output = Path(sys.argv[1]).resolve()
    samples: dict[str, list[int]] = {}
    exit_codes: dict[str, set[int]] = {}
    profiles: dict[str, Path] = {}
    with (output / "manifest.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            samples.setdefault(row["workload"], []).append(int(row["elapsed_ns"]))
            exit_codes.setdefault(row["workload"], set()).add(
                int(row.get("exit_code") or 0)
            )
            profiles.setdefault(row["workload"], Path(row["profile"]))

    workloads = []
    for name, elapsed in sorted(samples.items()):
        profile = json.loads(profiles[name].read_text())
        median = statistics.median(elapsed)
        retired = int(profile.get("retired", 0))
        workloads.append(
            {
                "name": name,
                "samples": len(elapsed),
                "exit_codes": sorted(exit_codes[name]),
                "median_ns": median,
                "mean_ns": statistics.fmean(elapsed),
                "min_ns": min(elapsed),
                "max_ns": max(elapsed),
                "p25_ns": percentile(elapsed, 0.25),
                "p75_ns": percentile(elapsed, 0.75),
                "median_ns_per_retired": median / retired if retired else None,
                "retired": retired,
                "max_stack_depth": profile.get("max_stack_depth", 0),
                "max_frame_depth": profile.get("max_frame_depth", 0),
                "heap": profile.get("heap", {}),
                "ffi": profile.get("ffi", {}),
            }
        )

    summary = {
        "schema": "nanoisa.benchmark-summary.v1",
        "environment": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "cc": command_output("cc", "--version").splitlines()[0],
            "git_commit": command_output("git", "rev-parse", "HEAD"),
        },
        "workloads": workloads,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# NanoISA Benchmark Summary",
        "",
        f"Commit: `{summary['environment']['git_commit']}`",
        "",
        "| Workload | Samples | Median ms | IQR ms | Retired | ns/retired |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in workloads:
        iqr = (item["p75_ns"] - item["p25_ns"]) / 1_000_000
        normalized = item["median_ns_per_retired"]
        lines.append(
            f"| `{item['name']}` | {item['samples']} | "
            f"{item['median_ns'] / 1_000_000:.3f} | {iqr:.3f} | "
            f"{item['retired']} | {normalized:.3f} |"
            if normalized is not None
            else f"| `{item['name']}` | {item['samples']} | "
            f"{item['median_ns'] / 1_000_000:.3f} | {iqr:.3f} | 0 | n/a |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
