#!/usr/bin/env python3
"""
Full analysis pipeline: simulate → analyse → visualise.

Runs simulations at multiple particle counts, performs a parameter scan
over several decay constants, applies detector effects, and generates
a comprehensive statistical comparison against theory.  All plots are
saved to ``results/plots/`` and timing information is printed.

Usage:
    python examples/run_full_analysis.py
"""
import time

import numpy as np

from config import PathConfig, SimulationConfig
from src.basic_simulation import generate_decay_times, run_basic_simulation
from src.detector_effects import apply_all_detector_effects
from src.statistics import compare_to_theory
from src.visualization import (
    plot_decay_curve,
    plot_decay_histogram,
    plot_detector_comparison,
    plot_parameter_scan,
)


plots_dir = PathConfig.PLOTS_DIR

# ------------------------------------------------------------------
# 1. Multi-N comparison
# ------------------------------------------------------------------
print("=" * 60)
print("  FULL ANALYSIS PIPELINE")
print("=" * 60)

decay_constant = SimulationConfig.DEFAULT_DECAY_CONSTANT
print(f"\n[1/4] Simulating multiple particle counts (λ={decay_constant}) …")

for n in SimulationConfig.PARTICLE_COUNTS:
    t0 = time.perf_counter()
    results = run_basic_simulation(
        n_particles=n, decay_constant=decay_constant, random_seed=42
    )
    elapsed = time.perf_counter() - t0
    report = compare_to_theory(
        results["decay_times"], decay_constant, n_bootstrap=200, random_seed=42
    )
    print(
        f"  N={n:>7,}  τ={report['mean_lifetime']:.4f} s  "
        f"err={report['error_percent']:.2f}%  "
        f"χ² p={report['p_value']:.4f}  "
        f"time={elapsed:.3f} s"
    )

# Save histogram and curve for the largest N
plot_decay_histogram(
    results["decay_times"],
    decay_constant,
    save_path=str(plots_dir / "full_histogram.png"),
    show=False,
)
plot_decay_curve(
    results["time_points"],
    results["remaining"],
    results["theoretical_curve"],
    results["n_particles"],
    decay_constant,
    save_path=str(plots_dir / "full_decay_curve.png"),
    show=False,
)

# ------------------------------------------------------------------
# 2. Parameter scan
# ------------------------------------------------------------------
print(f"\n[2/4] Parameter scan over λ = {SimulationConfig.DECAY_CONSTANTS} …")

scan_results = {}
for lam in SimulationConfig.DECAY_CONSTANTS:
    times = generate_decay_times(10_000, lam, random_seed=42)
    scan_results[lam] = times
    print(f"  λ={lam:.2f}  mean τ={np.mean(times):.2f} s  (theory {1/lam:.2f} s)")

plot_parameter_scan(
    scan_results,
    save_path=str(plots_dir / "full_parameter_scan.png"),
    show=False,
)

# ------------------------------------------------------------------
# 3. Detector effects
# ------------------------------------------------------------------
print("\n[3/4] Applying detector effects …")

true_times = generate_decay_times(50_000, decay_constant, random_seed=42)
detected = apply_all_detector_effects(
    true_times,
    efficiency=0.85,
    min_time=0.5,
    max_time=80.0,
    resolution=0.5,
    random_seed=42,
)
print(f"  True events   : {len(true_times):,}")
print(f"  Detected      : {len(detected):,}")
print(f"  True mean τ   : {np.mean(true_times):.4f} s")
print(f"  Detected mean : {np.mean(detected):.4f} s")

plot_detector_comparison(
    true_times,
    detected,
    efficiency=0.85,
    save_path=str(plots_dir / "full_detector_comparison.png"),
    show=False,
)

# ------------------------------------------------------------------
# 4. Full statistical report
# ------------------------------------------------------------------
print("\n[4/4] Final statistical report (N=100,000) …")

big_times = generate_decay_times(100_000, decay_constant, random_seed=42)
report = compare_to_theory(big_times, decay_constant,
                           n_bootstrap=500, random_seed=42)

print(f"  Mean τ         : {report['mean_lifetime']:.4f} s")
print(f"  Theory τ       : {report['theoretical_mean']:.4f} s")
print(f"  Error          : {report['error_percent']:.3f}%")
print(f"  χ² (p-value)   : {report['chi2']:.2f}  (p={report['p_value']:.4f})")
boot = report["bootstrap"]
print(f"  95% CI         : [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
print(f"  Accuracy test  : {'PASS' if report['passes_accuracy'] else 'FAIL'}")
print(f"  χ² test        : {'PASS' if report['passes_chi2'] else 'FAIL'}")

print("\n" + "=" * 60)
print(f"  All plots saved to {plots_dir}")
print("=" * 60)
