"""
Tests for the basic decay simulation module.

Covers decay-time generation, theoretical curves, input validation,
and reproducibility with fixed seeds.
"""
# isort: skip_file
from src.basic_simulation import (
    calculate_theoretical_curve,
    calculate_theoretical_pdf,
    compute_remaining_particles,
    generate_decay_times,
    run_basic_simulation,
)
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestGenerateDecayTimes:
    """Tests for generate_decay_times."""

    def test_correct_shape(self):
        """Output array has the requested number of particles."""
        times = generate_decay_times(500, 0.1, random_seed=42)
        assert times.shape == (500,)

    def test_all_positive(self):
        """Every decay time must be positive."""
        times = generate_decay_times(1000, 0.2, random_seed=42)
        assert np.all(times > 0)

    def test_mean_close_to_theory(self):
        """Mean lifetime should approximate 1/λ for large N."""
        decay_constant = 0.1
        times = generate_decay_times(50_000, decay_constant, random_seed=42)
        error_pct = abs(np.mean(times) - 1 / decay_constant) / \
            (1 / decay_constant) * 100
        assert error_pct < 2.0, f"Mean error {error_pct:.2f}% exceeds 2%"

    def test_reproducibility(self):
        """Same seed must produce identical results."""
        t1 = generate_decay_times(1000, 0.1, random_seed=99)
        t2 = generate_decay_times(1000, 0.1, random_seed=99)
        np.testing.assert_array_equal(t1, t2)

    def test_different_seeds_differ(self):
        """Different seeds must produce different results."""
        t1 = generate_decay_times(100, 0.1, random_seed=1)
        t2 = generate_decay_times(100, 0.1, random_seed=2)
        assert not np.array_equal(t1, t2)

    def test_invalid_n_particles_zero(self):
        """n_particles=0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_particles"):
            generate_decay_times(0, 0.1)

    def test_invalid_n_particles_negative(self):
        """Negative n_particles should raise ValueError."""
        with pytest.raises(ValueError, match="n_particles"):
            generate_decay_times(-5, 0.1)

    def test_invalid_decay_constant_zero(self):
        """decay_constant=0 should raise ValueError."""
        with pytest.raises(ValueError, match="decay_constant"):
            generate_decay_times(100, 0.0)

    def test_invalid_decay_constant_negative(self):
        """Negative decay_constant should raise ValueError."""
        with pytest.raises(ValueError, match="decay_constant"):
            generate_decay_times(100, -0.1)

    def test_single_particle(self):
        """Edge case: N=1 should return a single value."""
        times = generate_decay_times(1, 0.1, random_seed=42)
        assert times.shape == (1,)
        assert times[0] > 0

    def test_large_decay_constant(self):
        """Very large λ should give very short lifetimes."""
        times = generate_decay_times(10000, 10.0, random_seed=42)
        assert np.mean(times) < 0.2  # 1/10 = 0.1

    def test_small_decay_constant(self):
        """Very small λ should give long lifetimes."""
        times = generate_decay_times(10000, 0.001, random_seed=42)
        assert np.mean(times) > 500  # 1/0.001 = 1000


class TestTheoreticalCurve:
    """Tests for calculate_theoretical_curve."""

    def test_initial_value(self):
        """N(0) = N₀."""
        t = np.array([0.0])
        curve = calculate_theoretical_curve(t, 0.1, 1000)
        assert curve[0] == pytest.approx(1000.0)

    def test_decay_at_mean_lifetime(self):
        """N(1/λ) = N₀ / e."""
        decay_constant = 0.1
        n = 1000
        t = np.array([1.0 / decay_constant])
        curve = calculate_theoretical_curve(t, decay_constant, n)
        assert curve[0] == pytest.approx(n / np.e, rel=1e-10)

    def test_monotonically_decreasing(self):
        """Curve should always decrease."""
        t = np.linspace(0, 100, 500)
        curve = calculate_theoretical_curve(t, 0.1, 1000)
        assert np.all(np.diff(curve) <= 0)


class TestTheoreticalPDF:
    """Tests for calculate_theoretical_pdf."""

    def test_peak_at_zero(self):
        """PDF peaks at t=0 with value λ."""
        pdf = calculate_theoretical_pdf(np.array([0.0]), 0.1)
        assert pdf[0] == pytest.approx(0.1)

    def test_normalisation(self):
        """Integral of PDF from 0 to ∞ should be ~1."""
        t = np.linspace(0, 200, 10_000)
        pdf = calculate_theoretical_pdf(t, 0.1)
        integral = np.trapezoid(pdf, t)
        assert integral == pytest.approx(1.0, abs=0.01)


class TestComputeRemainingParticles:
    """Tests for compute_remaining_particles."""

    def test_all_alive_at_zero(self):
        """At t=0, all particles remain."""
        dt = np.array([1.0, 2.0, 3.0])
        result = compute_remaining_particles(dt, np.array([0.0]))
        assert result[0] == 3

    def test_all_decayed_at_end(self):
        """After the last decay, count should be 0."""
        dt = np.array([1.0, 2.0, 3.0])
        result = compute_remaining_particles(dt, np.array([10.0]))
        assert result[0] == 0


class TestRunBasicSimulation:
    """Tests for run_basic_simulation."""

    def test_returns_expected_keys(self):
        """Result dict contains all required keys."""
        results = run_basic_simulation(n_particles=100, decay_constant=0.1)
        expected_keys = {
            "decay_times",
            "time_points",
            "remaining",
            "theoretical_curve",
            "theoretical_pdf",
            "n_particles",
            "decay_constant",
            "mean_lifetime",
            "theoretical_mean",
        }
        assert expected_keys.issubset(results.keys())

    def test_result_shapes_consistent(self):
        """Array lengths should be consistent."""
        results = run_basic_simulation(n_particles=200, decay_constant=0.1)
        n = results["n_particles"]
        assert results["decay_times"].shape == (n,)
        assert len(results["time_points"]) == n
        assert len(results["remaining"]) == n
        assert len(results["theoretical_curve"]) == n
