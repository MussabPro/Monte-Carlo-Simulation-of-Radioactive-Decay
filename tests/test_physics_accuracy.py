"""
Physics accuracy tests.

Verifies that simulation results match theoretical predictions for
exponential decay within acceptable tolerances.
"""

from src.statistics import calculate_mean_lifetime, chi_squared_test
from src.multichannel_simulation import (
    analyze_channels,
    generate_multichannel_decay,
)
from src.detector_effects import (
    apply_acceptance_cuts,
    apply_detector_efficiency,
    simulate_resolution_smearing,
)
from src.basic_simulation import generate_decay_times
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestMeanLifetimeAccuracy:
    """Verify simulated mean lifetime matches 1/λ."""

    @pytest.mark.parametrize(
        "decay_constant",
        [0.05, 0.1, 0.2, 0.5, 1.0],
    )
    def test_mean_lifetime_within_5_percent(self, decay_constant: float):
        """Simulated mean lifetime matches theory within 5% for N=10 000."""
        n_particles = 10_000
        times = generate_decay_times(
            n_particles, decay_constant, random_seed=42)
        mean = float(np.mean(times))
        theory = 1.0 / decay_constant
        error_pct = abs(mean - theory) / theory * 100
        assert error_pct < 5.0, (
            f"λ={decay_constant}: error {error_pct:.2f}% exceeds 5% "
            f"(mean={mean:.4f}, theory={theory:.4f})"
        )

    def test_high_precision_large_n(self):
        """With N=100 000 the error should be under 1%."""
        times = generate_decay_times(100_000, 0.1, random_seed=42)
        mean = float(np.mean(times))
        error_pct = abs(mean - 10.0) / 10.0 * 100
        assert error_pct < 1.0, f"Error {error_pct:.2f}% exceeds 1%"


class TestChiSquaredAccuracy:
    """Verify χ² goodness-of-fit passes for correct parameters."""

    @pytest.mark.parametrize(
        "decay_constant",
        [0.05, 0.1, 0.15, 0.2],
    )
    def test_chi2_passes(self, decay_constant: float):
        """χ² test should pass (p > 0.05) for correctly-generated data."""
        times = generate_decay_times(10_000, decay_constant, random_seed=42)
        _, p_value, _ = chi_squared_test(times, decay_constant)
        assert p_value > 0.05, (
            f"λ={decay_constant}: p-value {p_value:.4f} < 0.05"
        )


class TestMultichannelPhysics:
    """Verify branching fractions are statistically consistent."""

    def test_branching_ratios_close(self):
        """Observed fractions should be within 3% of expected for N=50 000."""
        ratios = {"α": 0.6, "β": 0.3, "γ": 0.1}
        data = generate_multichannel_decay(50_000, 0.1, ratios, random_seed=42)
        stats = analyze_channels(data)

        for channel, expected in ratios.items():
            observed = stats[channel]["fraction"]
            diff = abs(observed - expected)
            assert diff < 0.03, (
                f"Channel '{channel}': observed {observed:.4f} vs "
                f"expected {expected:.4f} (diff {diff:.4f})"
            )

    def test_all_channels_same_lifetime(self):
        """All channels share the same exponential lifetime."""
        ratios = {"a": 0.5, "b": 0.3, "c": 0.2}
        data = generate_multichannel_decay(30_000, 0.1, ratios, random_seed=42)
        stats = analyze_channels(data)

        theory = 10.0
        for channel, s in stats.items():
            error_pct = abs(s["mean_lifetime"] - theory) / theory * 100
            assert error_pct < 5.0, (
                f"Channel '{channel}': mean τ {s['mean_lifetime']:.2f} "
                f"deviates {error_pct:.1f}% from theory"
            )


class TestDetectorPhysics:
    """Verify detector effects behave physically."""

    def test_efficiency_reduces_count(self):
        """Detected count should be ≈ efficiency × total."""
        times = generate_decay_times(100_000, 0.1, random_seed=42)
        detected = apply_detector_efficiency(times, 0.85, random_seed=42)
        ratio = len(detected) / len(times)
        assert 0.83 < ratio < 0.87, f"Ratio {ratio:.3f} not near 0.85"

    def test_efficiency_preserves_distribution(self):
        """Flat efficiency should not bias the mean lifetime."""
        times = generate_decay_times(100_000, 0.1, random_seed=42)
        detected = apply_detector_efficiency(times, 0.85, random_seed=42)
        mean_true = float(np.mean(times))
        mean_det = float(np.mean(detected))
        diff_pct = abs(mean_det - mean_true) / mean_true * 100
        assert diff_pct < 2.0, f"Efficiency biased mean by {diff_pct:.2f}%"

    def test_acceptance_cuts_remove_tails(self):
        """All surviving events should be within the cut window."""
        times = generate_decay_times(10_000, 0.1, random_seed=42)
        cut = apply_acceptance_cuts(times, min_time=2.0, max_time=50.0)
        assert np.all(cut >= 2.0)
        assert np.all(cut <= 50.0)

    def test_smearing_adds_noise(self):
        """Smeared times should differ from true times."""
        times = generate_decay_times(1_000, 0.1, random_seed=42)
        smeared = simulate_resolution_smearing(times, 0.5, random_seed=42)
        assert not np.array_equal(times, smeared)
        # Mean should still be approximately correct
        diff_pct = abs(np.mean(smeared) - np.mean(times)) / \
            np.mean(times) * 100
        assert diff_pct < 5.0

    def test_smeared_times_non_negative(self):
        """Smearing should clip negative times to zero."""
        times = np.array([0.01, 0.02, 0.03])
        smeared = simulate_resolution_smearing(times, 10.0, random_seed=42)
        assert np.all(smeared >= 0)


class TestReproducibility:
    """Verify that fixed seeds produce identical results everywhere."""

    def test_basic_simulation_deterministic(self):
        """Two runs with the same seed should be identical."""
        t1 = generate_decay_times(1_000, 0.1, random_seed=123)
        t2 = generate_decay_times(1_000, 0.1, random_seed=123)
        np.testing.assert_array_equal(t1, t2)

    def test_multichannel_deterministic(self):
        """Multi-channel simulation is deterministic with a seed."""
        ratios = {"a": 0.5, "b": 0.5}
        d1 = generate_multichannel_decay(1_000, 0.1, ratios, random_seed=7)
        d2 = generate_multichannel_decay(1_000, 0.1, ratios, random_seed=7)
        np.testing.assert_array_equal(d1["decay_times"], d2["decay_times"])
        np.testing.assert_array_equal(d1["channels"], d2["channels"])
