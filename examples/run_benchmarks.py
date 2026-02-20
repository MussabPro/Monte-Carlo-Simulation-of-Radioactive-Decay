#!/usr/bin/env python3
"""
Performance benchmarks for the Monte Carlo decay simulation.

Measures decay-time generation speed, full statistical pipeline
throughput, and peak memory usage across a range of particle counts.
Results are printed to stdout and optionally saved as JSON.

Usage:
    python examples/run_benchmarks.py
    python examples/run_benchmarks.py --output results/benchmarks/bench.json
"""
import argparse
import json
import os
import platform
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np

from src.basic_simulation import generate_decay_times
from src.statistics import compare_to_theory


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_system_info() -> dict:
    """Collect system and library metadata."""
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or "N/A",
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _format_bytes(n_bytes: int) -> str:
    """Human-readable byte string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _bench_one(n_particles: int, decay_constant: float = 0.1,
               n_bootstrap: int = 500, seed: int = 42) -> dict:
    """Benchmark a single particle count and return timing + memory."""
    # --- Generation only ---
    tracemalloc.start()
    t0 = time.perf_counter()
    times = generate_decay_times(n_particles, decay_constant, random_seed=seed)
    gen_time = time.perf_counter() - t0
    gen_mem = tracemalloc.get_traced_memory()[1]  # peak
    tracemalloc.stop()

    # --- Full stats pipeline ---
    tracemalloc.start()
    t0 = time.perf_counter()
    _times = generate_decay_times(
        n_particles, decay_constant, random_seed=seed)
    compare_to_theory(_times, decay_constant,
                      n_bootstrap=n_bootstrap, random_seed=seed)
    full_time = time.perf_counter() - t0
    full_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return {
        "n_particles": n_particles,
        "gen_time_s": round(gen_time, 6),
        "full_time_s": round(full_time, 6),
        "gen_peak_mem": gen_mem,
        "full_peak_mem": full_mem,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks.")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save results as JSON to this path (e.g. results/benchmarks/bench.json)")
    parser.add_argument(
        "--particles", "-n", type=int, nargs="+",
        default=[1_000, 10_000, 100_000, 1_000_000],
        help="Particle counts to benchmark (default: 1000 10000 100000 1000000)")
    args = parser.parse_args()

    info = _get_system_info()

    print("=" * 66)
    print("  PERFORMANCE BENCHMARKS")
    print("=" * 66)
    print(f"  Machine : {info['machine']}")
    print(f"  OS      : {info['os']}")
    print(f"  Python  : {info['python']}")
    print(f"  NumPy   : {info['numpy']}")
    print("=" * 66)
    print()
    print(
        f"  {'N':>12s}  {'Generation':>12s}  {'Full Pipeline':>14s}  {'Peak Memory':>12s}")
    print(f"  {'—' * 12}  {'—' * 12}  {'—' * 14}  {'—' * 12}")

    results = []
    for n in args.particles:
        row = _bench_one(n)
        results.append(row)
        print(
            f"  {n:>12,}  {row['gen_time_s']:>11.4f}s"
            f"  {row['full_time_s']:>13.4f}s"
            f"  {_format_bytes(row['full_peak_mem']):>12s}"
        )

    print()
    print("=" * 66)

    # Optionally save to JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"system": info, "benchmarks": results}
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  Results saved to {out_path}")
        print("=" * 66)


if __name__ == "__main__":
    main()
