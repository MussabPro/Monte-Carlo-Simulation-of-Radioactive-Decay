"""
Tests for the statistics module.

Covers mean-lifetime calculation, χ² goodness-of-fit,
bootstrap uncertainties, and the combined theory comparison.
"""

from src.statistics import (
    bootstrap_uncertainty,
    calculate_mean_lifetime,
    chi_squared_test,
    compare_to_theory,
)
from src.basic_simulation import generate_decay_times
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCalculateMeanLifetime:
    """Tests for calculate_mean_lifetime."""

    def test_basic_values(self):
        """Mean and std should be reasonable for exponential data."""
        times = generate_decay_times(10_000, 0.1, random_seed=42)
        mean, std, err = calculate_mean_lifetime(times)
        assert 8.0 < mean < 12.0
        assert std > 0
        assert err < std

    def test_standard_error_scales(self):
        """Standard error should decrease with √N."""
        t1 = generate_decay_times(1_000, 0.1, random_seed=42)
        t2 = generate_decay_times(10_000, 0.1, random_seed=42)
        _, _, err1 = calculate_mean_lifetime(t1)
        _, _, err2 = calculate_mean_lifetime(t2)
        # err2 should be ~√10 times smaller
        assert err2 < err1

    def test_empty_raises(self):
        """Empty array should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_mean_lifetime(np.array([]))


class TestChiSquaredTest:
    """Tests for chi_squared_test."""

    def test_good_fit(self):
        """A correct exponential sample should pass (p > 0.01)."""
        times = generate_decay_times(10_000, 0.1, random_seed=42)
        chi2, p, dof = chi_squared_test(times, 0.1)
        assert p > 0.01, f"p-value {p:.4f} is suspiciously low"
        assert dof > 0

    def test_wrong_lambda_fails(self):
        """Using a very wrong λ should give a low p-value."""
        times = generate_decay_times(10_000, 0.1, random_seed=42)
        chi2, p, dof = chi_squared_test(times, 1.0)  # wrong λ
        assert p < 0.01, f"Expected low p-value for wrong λ, got {p:.4f}"

    def test_invalid_decay_constant(self):
        """Zero or negative λ should raise."""
        times = generate_decay_times(100, 0.1, random_seed=42)
        with pytest.raises(ValueError):
            chi_squared_test(times, 0.0)

    def test_empty_data(self):
        """Empty data should raise."""
        with pytest.raises(ValueError):
            chi_squared_test(np.array([]), 0.1)


class TestBootstrapUncertainty:
    """Tests for bootstrap_uncertainty."""

    def test_ci_contains_mean(self):
        """The 95% CI should contain the sample mean."""
        times = generate_decay_times(5_000, 0.1, random_seed=42)
        result = bootstrap_uncertainty(times, n_bootstrap=500, random_seed=42)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_ci_contains_theory(self):
        """For large N, the 95% CI should include 1/λ."""
        times = generate_decay_times(20_000, 0.1, random_seed=42)
        result = bootstrap_uncertainty(times, n_bootstrap=500, random_seed=42)
        assert result["ci_lower"] < 10.0 < result["ci_upper"]

    def test_result_keys(self):
        """Result dictionary should contain expected keys."""
        times = generate_decay_times(100, 0.1, random_seed=42)
        result = bootstrap_uncertainty(times, n_bootstrap=50, random_seed=42)
        for key in ("mean", "std", "ci_lower", "ci_upper", "confidence_level"):
            assert key in result

    def test_empty_raises(self):
        """Empty array should raise."""
        with pytest.raises(ValueError):
            bootstrap_uncertainty(np.array([]))


class TestCompareToTheory:
    """Tests for compare_to_theory."""

    def test_passes_for_correct_lambda(self):
        """With correct λ and large N, both tests should pass."""
        times = generate_decay_times(10_000, 0.1, random_seed=42)
        report = compare_to_theory(times, 0.1, n_bootstrap=200, random_seed=42)
        assert report["passes_accuracy"], f"Error {report['error_percent']:.2f}%"
        assert report["passes_chi2"], f"p-value {report['p_value']:.4f}"

    def test_report_keys(self):
        """Report should contain all expected keys."""
        times = generate_decay_times(1_000, 0.1, random_seed=42)
        report = compare_to_theory(times, 0.1, n_bootstrap=50, random_seed=42)
        for key in (
            "mean_lifetime",
            "theoretical_mean",
            "error_percent",
            "chi2",
            "p_value",
            "bootstrap",
            "passes_accuracy",
            "passes_chi2",
        ):
            assert key in report
