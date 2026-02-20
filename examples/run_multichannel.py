#!/usr/bin/env python3
"""
Run a multi-channel decay simulation.

Simulates 10 000 particles decaying through three B-meson-inspired
channels, prints per-channel statistics, and saves a comparison plot.

Usage:
    python examples/run_multichannel.py
"""

from src.visualization import plot_multichannel_comparison
from src.multichannel_simulation import run_multichannel_simulation
from config import PathConfig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Define branching ratios (B-meson inspired)
branching_ratios = {
    "π⁺π⁻": 0.60,
    "K⁺π⁻": 0.30,
    "Other": 0.10,
}

# Run simulation
results = run_multichannel_simulation(
    n_particles=10_000,
    decay_constant=0.1,
    branching_ratios=branching_ratios,
    random_seed=42,
)

# Print per-channel results
print("=" * 60)
print("  MULTI-CHANNEL DECAY SIMULATION")
print("=" * 60)
print(f"  Particles  : {results['n_particles']:,}")
print(f"  λ          : {results['decay_constant']} s⁻¹")
print()

for channel, stats in results["channel_stats"].items():
    expected = branching_ratios[channel]
    print(f"  Channel '{channel}':")
    print(f"    Count          : {stats['count']:,}")
    print(
        f"    Observed ratio : {stats['fraction']:.4f}  (expected {expected:.2f})")
    print(f"    Mean lifetime  : {stats['mean_lifetime']:.4f} s")
    print()

print("=" * 60)

# Save comparison plot
save_path = str(PathConfig.PLOTS_DIR / "multichannel_comparison.png")
plot_multichannel_comparison(
    results["channel_stats"],
    results["decay_constant"],
    save_path=save_path,
    show=False,
)
print(f"Plot saved to {save_path}")
