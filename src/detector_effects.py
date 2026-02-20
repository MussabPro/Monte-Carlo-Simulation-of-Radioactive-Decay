"""
Detector effect modelling for Monte Carlo decay simulations.

Real particle physics experiments do not observe every decay event.
This module simulates three common detector imperfections:

1. **Efficiency** — a flat probability of detecting each event.
2. **Acceptance cuts** — discarding events outside a time window.
3. **Resolution smearing** — Gaussian blurring of measured times.

These effects are applied *after* the physics simulation so that one
can study how detector limitations distort the true distribution.

Example:
    >>> from src.detector_effects import apply_detector_efficiency
    >>> import numpy as np
    >>> times = np.random.exponential(10, size=10000)
    >>> detected = apply_detector_efficiency(times, efficiency=0.85)
    >>> print(f"Detected {len(detected)} / {len(times)} events")
"""

from src.basic_simulation import _validate_positive_float
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure the project root is on sys.path so the module works both when
# imported from the root and when run directly (python src/…).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detector effect functions
# ---------------------------------------------------------------------------

def apply_detector_efficiency(
    decay_times: np.ndarray,
    efficiency: float = 0.85,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Randomly keep each event with probability *efficiency*.

    This models a detector whose geometric acceptance and trigger
    efficiency combine to give a flat detection probability for every
    decay, independent of the decay time.

    Args:
        decay_times: 1-D array of simulated decay times.
        efficiency: Detection probability per event (0 < ε ≤ 1).
        random_seed: Optional RNG seed for reproducibility.

    Returns:
        A 1-D ``np.ndarray`` containing only the *detected* decay times
        (a subset of *decay_times*).

    Raises:
        ValueError: If *efficiency* is not in the range (0, 1].

    Example:
        >>> import numpy as np
        >>> times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> det = apply_detector_efficiency(times, 0.6, random_seed=42)
        >>> print(len(det) <= len(times))
        True
    """
    if not 0 < efficiency <= 1.0:
        raise ValueError(
            f"efficiency must be in (0, 1], got {efficiency}"
        )

    rng = np.random.default_rng(seed=random_seed)
    detected_mask = rng.random(len(decay_times)) < efficiency
    detected_times = decay_times[detected_mask]

    n_total = len(decay_times)
    n_detected = len(detected_times)
    n_missed = n_total - n_detected

    logger.info(
        "Detector efficiency %.1f%%: %d detected, %d missed out of %d total",
        efficiency * 100,
        n_detected,
        n_missed,
        n_total,
    )

    return detected_times


def apply_acceptance_cuts(
    decay_times: np.ndarray,
    min_time: float = 0.0,
    max_time: Optional[float] = None,
) -> np.ndarray:
    """Discard events outside an acceptance time window.

    In real experiments, very short-lived decays may be masked by
    prompt backgrounds, and very long-lived ones may escape the
    detector volume.  This function models both effects with simple
    rectangular cuts.

    Args:
        decay_times: 1-D array of simulated decay times.
        min_time: Lower bound of the acceptance window (inclusive).
            Must be non-negative.
        max_time: Upper bound of the acceptance window (inclusive).
            If ``None``, no upper cut is applied.

    Returns:
        A 1-D ``np.ndarray`` with only the events inside *[min_time,
        max_time]*.

    Raises:
        ValueError: If *min_time* is negative or *max_time* < *min_time*.

    Example:
        >>> import numpy as np
        >>> times = np.array([0.5, 5.0, 15.0, 50.0, 200.0])
        >>> cut = apply_acceptance_cuts(times, min_time=1.0, max_time=100.0)
        >>> print(cut)
        [  5.  15.  50.]
    """
    if min_time < 0:
        raise ValueError(
            f"min_time must be non-negative, got {min_time}"
        )

    mask = decay_times >= min_time

    if max_time is not None:
        if max_time < min_time:
            raise ValueError(
                f"max_time ({max_time}) must be >= min_time ({min_time})"
            )
        mask &= decay_times <= max_time

    accepted_times = decay_times[mask]

    logger.info(
        "Acceptance cuts [%.2f, %s]: kept %d / %d events",
        min_time,
        f"{max_time:.2f}" if max_time is not None else "∞",
        len(accepted_times),
        len(decay_times),
    )

    return accepted_times


def simulate_resolution_smearing(
    decay_times: np.ndarray,
    resolution: float = 0.1,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Apply Gaussian resolution smearing to measured decay times.

    A real detector measures each decay time with finite precision.
    This function adds Gaussian noise with standard deviation *resolution*
    to each time measurement.  Negative smeared times are clipped to
    zero (a decay cannot occur before the experiment starts).

    Args:
        decay_times: 1-D array of true decay times.
        resolution: Standard deviation σ of the Gaussian smearing
            (seconds). Must be positive.
        random_seed: Optional RNG seed for reproducibility.

    Returns:
        A 1-D ``np.ndarray`` of smeared decay times, same length as
        *decay_times*.

    Raises:
        ValueError: If *resolution* is not positive.

    Example:
        >>> import numpy as np
        >>> true_times = np.array([10.0, 20.0, 30.0])
        >>> smeared = simulate_resolution_smearing(true_times, 0.5, random_seed=42)
        >>> print(smeared.shape)
        (3,)
    """
    _validate_positive_float(resolution, "resolution")

    rng = np.random.default_rng(seed=random_seed)
    noise = rng.normal(loc=0.0, scale=resolution, size=len(decay_times))
    smeared_times = decay_times + noise

    # Physical constraint: decay time >= 0
    smeared_times = np.clip(smeared_times, 0.0, None)

    logger.info(
        "Applied Gaussian smearing (σ=%.4f s) to %d events",
        resolution,
        len(decay_times),
    )

    return smeared_times


def apply_all_detector_effects(
    decay_times: np.ndarray,
    efficiency: float = 0.85,
    min_time: float = 0.0,
    max_time: Optional[float] = None,
    resolution: float = 0.1,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Apply efficiency, acceptance cuts, and smearing in sequence.

    This is a convenience wrapper that chains all three detector effects.
    The order is: smearing → efficiency → acceptance cuts (matching
    how a real detector pipeline operates).

    Args:
        decay_times: 1-D array of true decay times.
        efficiency: Detection probability per event (0 < ε ≤ 1).
        min_time: Lower acceptance cut (seconds).
        max_time: Upper acceptance cut (seconds), or ``None``.
        resolution: Gaussian smearing σ (seconds).
        random_seed: Optional RNG seed for reproducibility.

    Returns:
        A 1-D ``np.ndarray`` of detector-processed decay times.

    Example:
        >>> import numpy as np
        >>> times = np.random.exponential(10, size=10000)
        >>> processed = apply_all_detector_effects(times, efficiency=0.9)
        >>> print(f"Kept {len(processed)} / {len(times)} events")
    """
    logger.info("Applying full detector pipeline")

    processed = simulate_resolution_smearing(
        decay_times, resolution, random_seed
    )
    processed = apply_detector_efficiency(
        processed, efficiency, random_seed
    )
    processed = apply_acceptance_cuts(processed, min_time, max_time)

    logger.info(
        "Detector pipeline complete: %d → %d events",
        len(decay_times),
        len(processed),
    )

    return processed


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

    n_particles = SimulationConfig.DEFAULT_N_PARTICLES
    decay_constant = SimulationConfig.DEFAULT_DECAY_CONSTANT
    rng = np.random.default_rng(SimulationConfig.RANDOM_SEED)
    decay_times = rng.exponential(1.0 / decay_constant, size=n_particles)

    print("=" * 60)
    print("DETECTOR EFFECTS DEMONSTRATION")
    print("=" * 60)
    print(f"  Total events          : {len(decay_times):,}")
    print(f"  True mean lifetime    : {np.mean(decay_times):.4f} s")
    print()

    detected = apply_detector_efficiency(decay_times, efficiency=0.85)
    print(f"  After 85% efficiency  : {len(detected):,} events")
    print(f"  Detected mean τ       : {np.mean(detected):.4f} s")
    print()

    cut = apply_acceptance_cuts(decay_times, min_time=0.5, max_time=80.0)
    print(f"  After acceptance cuts : {len(cut):,} events")
    print(f"  Cut mean τ            : {np.mean(cut):.4f} s")
    print()

    smeared = simulate_resolution_smearing(decay_times, resolution=0.5)
    print(f"  After smearing (σ=0.5): {len(smeared):,} events")
    print(f"  Smeared mean τ        : {np.mean(smeared):.4f} s")
    print()

    full = apply_all_detector_effects(
        decay_times, efficiency=0.85, min_time=0.5, max_time=80.0, resolution=0.5
    )
    print(f"  Full pipeline result  : {len(full):,} events")
    print(f"  Final mean τ          : {np.mean(full):.4f} s")
    print("=" * 60)
