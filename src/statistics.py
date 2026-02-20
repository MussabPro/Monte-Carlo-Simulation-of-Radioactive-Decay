"""
Statistical analysis tools for Monte Carlo decay simulations.

Provides functions for computing summary statistics, performing
goodness-of-fit tests, and estimating uncertainties via bootstrap
resampling.  Results are compared against the known theoretical
predictions of exponential decay.

Example:
    >>> from src.statistics import calculate_mean_lifetime, chi_squared_test
    >>> import numpy as np
    >>> times = np.random.exponential(10.0, size=5000)
    >>> mean, std, err = calculate_mean_lifetime(times)
"""

from src.basic_simulation import (
    _validate_positive_float,
    _validate_positive_integer,
)
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

# Ensure the project root is on sys.path so the module works both when
# imported from the root and when run directly (python src/…).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def calculate_mean_lifetime(
    decay_times: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute the mean lifetime, standard deviation, and standard error.

    The standard error of the mean is σ / √N, which quantifies how
    precisely the sample mean estimates the true mean.

    Args:
        decay_times: 1-D array of simulated decay times.

    Returns:
        A tuple ``(mean, std, standard_error)``.

    Raises:
        ValueError: If *decay_times* is empty.

    Example:
        >>> import numpy as np
        >>> times = np.random.exponential(10.0, size=5000)
        >>> mean, std, err = calculate_mean_lifetime(times)
        >>> print(f"τ = {mean:.2f} ± {err:.2f} s")
    """
    if len(decay_times) == 0:
        raise ValueError("decay_times must not be empty")

    mean = float(np.mean(decay_times))
    std = float(np.std(decay_times, ddof=1))
    standard_error = std / np.sqrt(len(decay_times))

    logger.info(
        "Mean lifetime: %.4f ± %.4f s  (N=%d)",
        mean,
        standard_error,
        len(decay_times),
    )

    return mean, std, standard_error


# ---------------------------------------------------------------------------
# Goodness-of-fit
# ---------------------------------------------------------------------------

def chi_squared_test(
    decay_times: np.ndarray,
    decay_constant: float,
    n_bins: int = 30,
) -> Tuple[float, float, int]:
    """Pearson χ² test comparing observed histogram to the exponential PDF.

    The decay times are binned, and the observed counts are compared
    to the expected counts derived from the theoretical PDF
    f(t) = λ·exp(-λt).  Bins with fewer than 5 expected counts are
    merged to satisfy the test's validity conditions.

    Args:
        decay_times: 1-D array of simulated decay times.
        decay_constant: Theoretical decay rate λ (s⁻¹).
        n_bins: Number of histogram bins before merging.

    Returns:
        A tuple ``(chi2_statistic, p_value, degrees_of_freedom)``.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> import numpy as np
        >>> times = np.random.exponential(10.0, size=10000)
        >>> chi2, p, dof = chi_squared_test(times, 0.1)
        >>> print(f"χ² = {chi2:.2f}, p = {p:.4f}, dof = {dof}")
    """
    _validate_positive_float(decay_constant, "decay_constant")

    n_total = len(decay_times)
    if n_total == 0:
        raise ValueError("decay_times must not be empty")

    # Build histogram
    observed, bin_edges = np.histogram(decay_times, bins=n_bins)
    bin_widths = np.diff(bin_edges)

    # Expected counts from theoretical CDF
    cdf_lower = 1.0 - np.exp(-decay_constant * bin_edges[:-1])
    cdf_upper = 1.0 - np.exp(-decay_constant * bin_edges[1:])
    expected = n_total * (cdf_upper - cdf_lower)

    # Merge bins with expected < 5 (from the right end)
    merged_observed = []
    merged_expected = []
    obs_acc, exp_acc = 0, 0.0

    for obs, exp in zip(observed, expected):
        obs_acc += obs
        exp_acc += exp
        if exp_acc >= 5.0:
            merged_observed.append(obs_acc)
            merged_expected.append(exp_acc)
            obs_acc, exp_acc = 0, 0.0

    # Flush remainder into last bin
    if obs_acc > 0 or exp_acc > 0:
        if merged_expected:
            merged_observed[-1] += obs_acc
            merged_expected[-1] += exp_acc
        else:
            merged_observed.append(obs_acc)
            merged_expected.append(exp_acc)

    merged_observed = np.array(merged_observed, dtype=float)
    merged_expected = np.array(merged_expected, dtype=float)

    # χ² statistic
    chi2 = float(
        np.sum((merged_observed - merged_expected) ** 2 / merged_expected))
    dof = len(merged_observed) - 1  # one parameter (λ) estimated
    p_value = float(1.0 - sp_stats.chi2.cdf(chi2, dof))

    logger.info(
        "Chi-squared test: χ²=%.4f, dof=%d, p=%.4f",
        chi2,
        dof,
        p_value,
    )

    return chi2, p_value, dof


# ---------------------------------------------------------------------------
# Bootstrap uncertainty
# ---------------------------------------------------------------------------

def bootstrap_uncertainty(
    decay_times: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate mean-lifetime uncertainty using bootstrap resampling.

    The bootstrap draws *n_bootstrap* samples (with replacement) of the
    same size as *decay_times*, computes the mean of each, and reports
    the resulting confidence interval.

    Args:
        decay_times: 1-D array of simulated decay times.
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: Confidence level for the interval (e.g. 0.95).
        random_seed: Optional RNG seed for reproducibility.

    Returns:
        A dictionary with keys:
            - ``"mean"``: Point estimate (sample mean).
            - ``"std"``: Bootstrap standard deviation of the mean.
            - ``"ci_lower"``: Lower bound of the confidence interval.
            - ``"ci_upper"``: Upper bound of the confidence interval.
            - ``"confidence_level"``: The level used.
            - ``"n_bootstrap"``: Number of resamples used.

    Raises:
        ValueError: If *decay_times* is empty or parameters are invalid.

    Example:
        >>> import numpy as np
        >>> times = np.random.exponential(10.0, size=5000)
        >>> result = bootstrap_uncertainty(times, n_bootstrap=500, random_seed=42)
        >>> print(f"τ = {result['mean']:.2f}  95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
    """
    if len(decay_times) == 0:
        raise ValueError("decay_times must not be empty")
    _validate_positive_integer(n_bootstrap, "n_bootstrap")
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    rng = np.random.default_rng(seed=random_seed)
    n_samples = len(decay_times)

    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(decay_times, size=n_samples, replace=True)
        bootstrap_means[i] = np.mean(resample)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    result = {
        "mean": float(np.mean(decay_times)),
        "std": float(np.std(bootstrap_means)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
    }

    logger.info(
        "Bootstrap (n=%d): τ = %.4f, %.0f%% CI [%.4f, %.4f]",
        n_bootstrap,
        result["mean"],
        confidence_level * 100,
        ci_lower,
        ci_upper,
    )

    return result


# ---------------------------------------------------------------------------
# Theory comparison
# ---------------------------------------------------------------------------

def compare_to_theory(
    decay_times: np.ndarray,
    decay_constant: float,
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a comprehensive comparison of simulated data against theory.

    Combines mean-lifetime calculation, relative error, χ² test, and
    bootstrap confidence intervals into a single report.

    Args:
        decay_times: 1-D array of simulated decay times.
        decay_constant: Theoretical decay rate λ (s⁻¹).
        n_bootstrap: Number of bootstrap resamples.
        random_seed: Optional RNG seed.

    Returns:
        A dictionary containing:
            - ``"mean_lifetime"``, ``"std"``, ``"standard_error"``:
              From :func:`calculate_mean_lifetime`.
            - ``"theoretical_mean"``: 1/λ.
            - ``"error_percent"``: Relative error as a percentage.
            - ``"chi2"``, ``"p_value"``, ``"dof"``:
              From :func:`chi_squared_test`.
            - ``"bootstrap"``: Full result dict from
              :func:`bootstrap_uncertainty`.
            - ``"passes_accuracy"``: Whether error < 5%.
            - ``"passes_chi2"``: Whether p > 0.05.

    Example:
        >>> import numpy as np
        >>> times = np.random.exponential(10.0, size=10000)
        >>> report = compare_to_theory(times, 0.1, random_seed=42)
        >>> print(f"Error: {report['error_percent']:.2f}%")
    """
    _validate_positive_float(decay_constant, "decay_constant")

    # Mean lifetime
    mean, std, standard_error = calculate_mean_lifetime(decay_times)
    theoretical_mean = 1.0 / decay_constant
    error_percent = abs(mean - theoretical_mean) / theoretical_mean * 100

    # Chi-squared
    chi2, p_value, dof = chi_squared_test(decay_times, decay_constant)

    # Bootstrap
    boot = bootstrap_uncertainty(
        decay_times, n_bootstrap=n_bootstrap, random_seed=random_seed
    )

    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(
        __file__).resolve().parent.parent))
    from config import SimulationConfig

    report: Dict[str, Any] = {
        "n_particles": len(decay_times),
        "decay_constant": decay_constant,
        "mean_lifetime": mean,
        "std": std,
        "standard_error": standard_error,
        "theoretical_mean": theoretical_mean,
        "error_percent": error_percent,
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "bootstrap": boot,
        "passes_accuracy": error_percent < SimulationConfig.ACCURACY_THRESHOLD_PERCENT,
        "passes_chi2": p_value > SimulationConfig.CHI2_P_VALUE_THRESHOLD,
    }

    logger.info(
        "Theory comparison: error=%.2f%% (%s), χ² p=%.4f (%s)",
        error_percent,
        "PASS" if report["passes_accuracy"] else "FAIL",
        p_value,
        "PASS" if report["passes_chi2"] else "FAIL",
    )

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(
        __file__).resolve().parent.parent))
    from config import SimulationConfig

    rng = np.random.default_rng(SimulationConfig.RANDOM_SEED)
    decay_constant = SimulationConfig.DEFAULT_DECAY_CONSTANT
    n_particles = SimulationConfig.DEFAULT_N_PARTICLES
    decay_times = rng.exponential(1.0 / decay_constant, size=n_particles)

    report = compare_to_theory(decay_times, decay_constant, random_seed=42)

    print("=" * 60)
    print("STATISTICAL ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Particles            : {report['n_particles']:,}")
    print(f"  Decay constant (λ)   : {report['decay_constant']:.4f} s⁻¹")
    print(f"  Simulated mean τ     : {report['mean_lifetime']:.4f} s")
    print(f"  Theoretical mean τ   : {report['theoretical_mean']:.4f} s")
    print(f"  Standard error       : {report['standard_error']:.4f} s")
    print(f"  Percentage error     : {report['error_percent']:.2f}%")
    print(
        f"  Accuracy test        : {'PASS' if report['passes_accuracy'] else 'FAIL'}")
    print()
    print(f"  χ² statistic         : {report['chi2']:.4f}")
    print(f"  p-value              : {report['p_value']:.4f}")
    print(f"  Degrees of freedom   : {report['dof']}")
    print(
        f"  χ² test              : {'PASS' if report['passes_chi2'] else 'FAIL'}")
    print()
    boot = report["bootstrap"]
    print(
        f"  Bootstrap 95% CI     : [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
    print(f"  Bootstrap std(mean)  : {boot['std']:.4f} s")
    print("=" * 60)
