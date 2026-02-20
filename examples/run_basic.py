#!/usr/bin/env python3
"""
Run a basic exponential decay simulation.

Generates 1 000 decay times, prints summary statistics, and saves
a histogram plot to ``results/plots/``.

Usage:
    python examples/run_basic.py
"""

from config import PathConfig
from src.basic_simulation import run_basic_simulation
from src.statistics import calculate_mean_lifetime
from src.visualization import plot_decay_histogram

# Run simulation with 1 000 particles
results = run_basic_simulation(
    n_particles=1_000, decay_constant=0.1, random_seed=42)

# Print results
mean, std, err = calculate_mean_lifetime(results["decay_times"])
print("=" * 50)
print("  BASIC DECAY SIMULATION")
print("=" * 50)
print(f"  Particles  : {results['n_particles']:,}")
print(f"  λ          : {results['decay_constant']} s⁻¹")
print(f"  Mean τ     : {mean:.4f} ± {err:.4f} s")
print(f"  Theory τ   : {results['theoretical_mean']:.4f} s")
print("=" * 50)

# Save plot
save_path = str(PathConfig.PLOTS_DIR / "basic_histogram.png")
plot_decay_histogram(
    results["decay_times"],
    results["decay_constant"],
    save_path=save_path,
    show=False,
)
print(f"Plot saved to {save_path}")
