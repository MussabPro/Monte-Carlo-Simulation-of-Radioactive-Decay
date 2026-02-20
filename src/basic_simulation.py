"""
Core exponential decay simulation using Monte Carlo methods.

This module implements the fundamental radioactive decay simulation where
particle lifetimes are drawn from an exponential distribution. It provides
functions to generate decay times, compute theoretical predictions, and
run complete simulation pipelines.

Physics background:
    Radioactive decay follows the exponential law N(t) = N₀ · exp(-λt),
    where λ is the decay constant and 1/λ is the mean lifetime. Individual
    decay times are random variables sampled from the exponential PDF
    f(t) = λ · exp(-λt).

    NumPy's ``np.random.exponential`` is used instead of per-particle loops
    because vectorised sampling is orders of magnitude faster for large N
    (see Performance notes in docs/PERFORMANCE.md).

Example:
    >>> from src.basic_simulation import generate_decay_times, run_basic_simulation
    >>> times = generate_decay_times(n_particles=1000, decay_constant=0.1, random_seed=42)
    >>> print(f"Mean lifetime: {times.mean():.2f} s")
"""

import logging
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def generate_decay_times(
    n_particles: int,
    decay_constant: float,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random decay times from an exponential distribution.

    Each element of the returned array represents the time at which one
    particle decays. The underlying distribution is f(t) = λ · exp(-λt)
    with mean 1/λ.

    Args:
        n_particles: Number of particles to simulate. Must be positive.
        decay_constant: Decay rate λ in s⁻¹. Must be positive.
        random_seed: Optional seed for the random number generator.
            When set, results are exactly reproducible.

    Returns:
        A 1-D ``np.ndarray`` of shape ``(n_particles,)`` containing
        the decay time of each particle in seconds.

    Raises:
        ValueError: If *n_particles* is not a positive integer or
            *decay_constant* is not a positive number.

    Example:
        >>> times = generate_decay_times(5000, 0.1, random_seed=42)
        >>> print(times.shape)
        (5000,)
        >>> print(f"Mean: {times.mean():.1f} s  (theory: 10.0 s)")
    """
    _validate_positive_integer(n_particles, "n_particles")
    _validate_positive_float(decay_constant, "decay_constant")

    rng = np.random.default_rng(seed=random_seed)
    # scale = mean = 1 / λ for the exponential distribution
    decay_times: np.ndarray = rng.exponential(
        scale=1.0 / decay_constant, size=n_particles
    )

    logger.info(
        "Generated %d decay times  (λ=%.4f, seed=%s)",
        n_particles,
        decay_constant,
        random_seed,
    )
    return decay_times


def calculate_theoretical_curve(
    times: np.ndarray,
    decay_constant: float,
    n_particles: int,
) -> np.ndarray:
    """Compute the theoretical surviving-particle curve N(t) = N₀·exp(-λt).

    This is the *expected* number of particles remaining at each point
    in *times*, useful for overlaying on simulation results.

    Args:
        times: 1-D array of time values (seconds) at which to evaluate
            the curve.
        decay_constant: Decay rate λ in s⁻¹. Must be positive.
        n_particles: Initial number of particles N₀.

    Returns:
        A 1-D ``np.ndarray`` the same length as *times* with the
        theoretical particle count at each time.

    Raises:
        ValueError: If *decay_constant* or *n_particles* is not positive.

    Example:
        >>> t = np.linspace(0, 50, 100)
        >>> curve = calculate_theoretical_curve(t, 0.1, 1000)
        >>> print(f"N(0)={curve[0]:.0f}, N(10)={curve[50]:.0f}")
    """
    _validate_positive_float(decay_constant, "decay_constant")
    _validate_positive_integer(n_particles, "n_particles")

    theoretical: np.ndarray = n_particles * np.exp(-decay_constant * times)
    return theoretical


def calculate_theoretical_pdf(
    times: np.ndarray,
    decay_constant: float,
) -> np.ndarray:
    """Compute the theoretical probability density f(t) = λ·exp(-λt).

    Used for overlaying on normalised histograms of simulated decay times.

    Args:
        times: 1-D array of time values (seconds).
        decay_constant: Decay rate λ in s⁻¹. Must be positive.

    Returns:
        A 1-D ``np.ndarray`` of probability densities at each time.

    Example:
        >>> t = np.linspace(0, 50, 200)
        >>> pdf = calculate_theoretical_pdf(t, 0.1)
        >>> print(f"f(0)={pdf[0]:.3f}")
    """
    _validate_positive_float(decay_constant, "decay_constant")
    return decay_constant * np.exp(-decay_constant * times)


def compute_remaining_particles(
    decay_times: np.ndarray,
    time_points: np.ndarray,
) -> np.ndarray:
    """Count how many particles are still alive at each time point.

    A particle is "alive" at time *t* if its decay time is strictly
    greater than *t*.

    Args:
        decay_times: 1-D array of individual particle decay times.
        time_points: 1-D array of observation times.

    Returns:
        A 1-D ``np.ndarray`` the same length as *time_points* with the
        count of surviving particles at each time.

    Example:
        >>> dt = np.array([1.0, 3.0, 5.0, 7.0])
        >>> tp = np.array([0, 2, 4, 6, 8])
        >>> print(compute_remaining_particles(dt, tp))
        [4 3 2 1 0]
    """
    # Using searchsorted on a sorted copy is O(N log N + M log N),
    # much faster than broadcasting for large arrays.
    sorted_times = np.sort(decay_times)
    n_total = len(decay_times)
    n_decayed = np.searchsorted(sorted_times, time_points, side="right")
    return n_total - n_decayed


def run_basic_simulation(
    n_particles: Optional[int] = None,
    decay_constant: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> dict:
    """Run a complete basic decay simulation and return all results.

    This is the high-level entry point that ties together decay-time
    generation, theoretical curve computation, and summary statistics.
    Missing parameters are filled from ``SimulationConfig`` defaults.

    Args:
        n_particles: Number of particles. Defaults to
            ``SimulationConfig.DEFAULT_N_PARTICLES``.
        decay_constant: Decay rate λ (s⁻¹). Defaults to
            ``SimulationConfig.DEFAULT_DECAY_CONSTANT``.
        random_seed: RNG seed. Defaults to
            ``SimulationConfig.RANDOM_SEED``.

    Returns:
        A dictionary with keys:
            - ``"decay_times"``: 1-D array of simulated decay times.
            - ``"time_points"``: 1-D array of evenly-spaced evaluation times.
            - ``"remaining"``: Count of surviving particles at each time point.
            - ``"theoretical_curve"``: Theoretical N(t) at each time point.
            - ``"theoretical_pdf"``: Theoretical PDF at each time point.
            - ``"n_particles"``: Number of particles simulated.
            - ``"decay_constant"``: Decay constant used.
            - ``"mean_lifetime"``: Simulated mean lifetime (seconds).
            - ``"theoretical_mean"``: Expected mean lifetime 1/λ.

    Example:
        >>> results = run_basic_simulation(n_particles=1000, decay_constant=0.1)
        >>> print(f"Simulated mean: {results['mean_lifetime']:.2f} s")
    """
    # Import here to avoid circular import at module level
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(
        __file__).resolve().parent.parent))
    from config import SimulationConfig

    if n_particles is None:
        n_particles = SimulationConfig.DEFAULT_N_PARTICLES
    if decay_constant is None:
        decay_constant = SimulationConfig.DEFAULT_DECAY_CONSTANT
    if random_seed is None:
        random_seed = SimulationConfig.RANDOM_SEED

    logger.info(
        "Running basic simulation: N=%d, λ=%.4f, seed=%d",
        n_particles,
        decay_constant,
        random_seed,
    )

    # Generate decay times
    decay_times = generate_decay_times(
        n_particles, decay_constant, random_seed)

    # Build evaluation grid
    max_time = float(np.max(decay_times)) * 1.1
    time_points = np.linspace(0, max_time, n_particles)

    # Observed surviving-particle counts
    remaining = compute_remaining_particles(decay_times, time_points)

    # Theoretical predictions
    theoretical_curve = calculate_theoretical_curve(
        time_points, decay_constant, n_particles
    )
    theoretical_pdf = calculate_theoretical_pdf(time_points, decay_constant)

    # Summary statistics
    mean_lifetime = float(np.mean(decay_times))
    theoretical_mean = 1.0 / decay_constant

    logger.info(
        "Simulation complete — mean lifetime: %.4f s (theory: %.4f s)",
        mean_lifetime,
        theoretical_mean,
    )

    return {
        "decay_times": decay_times,
        "time_points": time_points,
        "remaining": remaining,
        "theoretical_curve": theoretical_curve,
        "theoretical_pdf": theoretical_pdf,
        "n_particles": n_particles,
        "decay_constant": decay_constant,
        "mean_lifetime": mean_lifetime,
        "theoretical_mean": theoretical_mean,
    }


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_positive_integer(value: int, name: str) -> None:
    """Raise ``ValueError`` if *value* is not a positive integer."""
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        )


def _validate_positive_float(value: float, name: str) -> None:
    """Raise ``ValueError`` if *value* is not a positive number."""
    if not isinstance(value, (int, float, np.floating)) or value <= 0:
        raise ValueError(
            f"{name} must be a positive number, got {value!r}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    results = run_basic_simulation()

    print("=" * 60)
    print("BASIC DECAY SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Particles simulated : {results['n_particles']:,}")
    print(f"  Decay constant (λ)  : {results['decay_constant']:.4f} s⁻¹")
    print(f"  Simulated mean τ    : {results['mean_lifetime']:.4f} s")
    print(f"  Theoretical mean τ  : {results['theoretical_mean']:.4f} s")
    error_pct = (
        abs(results["mean_lifetime"] - results["theoretical_mean"])
        / results["theoretical_mean"]
        * 100
    )
    print(f"  Percentage error    : {error_pct:.2f}%")
    print("=" * 60)
