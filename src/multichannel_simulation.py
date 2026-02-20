"""
Multi-channel particle decay simulation.

Extends the basic simulation to support multiple decay channels with
configurable branching ratios — modelling how a single particle species
can decay into different final states with known probabilities.

Physics background:
    Many particles (e.g. B mesons studied at KEK's Belle II experiment)
    decay through several competing channels.  The *branching ratio* of
    a channel is the probability that a given decay produces that
    particular final state.  All branching ratios must sum to 1.

Example:
    >>> from src.multichannel_simulation import generate_multichannel_decay
    >>> ratios = {"π⁺π⁻": 0.6, "K⁺π⁻": 0.3, "Other": 0.1}
    >>> data = generate_multichannel_decay(10000, 0.1, ratios, random_seed=42)
    >>> print(data["channels"][:5])
"""

from src.basic_simulation import (
    _validate_positive_float,
    _validate_positive_integer,
    generate_decay_times,
)
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure the project root is on sys.path so the module works both when
# imported from the root and when run directly (python src/…).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def generate_multichannel_decay(
    n_particles: int,
    decay_constant: float,
    branching_ratios: Dict[str, float],
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Simulate particle decay across multiple channels.

    Decay times are drawn from a single exponential distribution (all
    channels share the same lifetime), and each particle is randomly
    assigned to a channel according to *branching_ratios*.

    Args:
        n_particles: Number of particles to simulate. Must be positive.
        decay_constant: Decay rate λ in s⁻¹. Must be positive.
        branching_ratios: Mapping of channel name → probability.
            Values must be non-negative and sum to 1.0 (within
            floating-point tolerance).
        random_seed: Optional RNG seed for reproducibility.

    Returns:
        A dictionary with keys:
            - ``"decay_times"``: 1-D array of decay times (seconds).
            - ``"channels"``: 1-D array of channel name strings,
              one per particle.
            - ``"branching_ratios"``: Copy of the input ratios.
            - ``"n_particles"``: Number of particles simulated.
            - ``"decay_constant"``: Decay constant used.

    Raises:
        ValueError: If inputs are invalid or branching ratios do not
            sum to 1.0.

    Example:
        >>> ratios = {"α": 0.7, "β": 0.2, "γ": 0.1}
        >>> data = generate_multichannel_decay(5000, 0.1, ratios, random_seed=42)
        >>> print(data["decay_times"].shape)
        (5000,)
    """
    _validate_positive_integer(n_particles, "n_particles")
    _validate_positive_float(decay_constant, "decay_constant")
    _validate_branching_ratios(branching_ratios)

    rng = np.random.default_rng(seed=random_seed)

    # Generate decay times (shared lifetime across channels)
    decay_times = generate_decay_times(
        n_particles, decay_constant, random_seed)

    # Assign each particle to a channel
    channels = list(branching_ratios.keys())
    probabilities = list(branching_ratios.values())
    channel_assignments = rng.choice(
        channels, size=n_particles, p=probabilities)

    logger.info(
        "Multi-channel decay: N=%d, λ=%.4f, channels=%s",
        n_particles,
        decay_constant,
        channels,
    )

    return {
        "decay_times": decay_times,
        "channels": channel_assignments,
        "branching_ratios": dict(branching_ratios),
        "n_particles": n_particles,
        "decay_constant": decay_constant,
    }


def analyze_channels(
    decay_data: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Compute per-channel statistics from multi-channel simulation results.

    For each channel, the function reports total count, observed fraction,
    mean lifetime, and standard deviation.

    Args:
        decay_data: Dictionary returned by
            :func:`generate_multichannel_decay`.

    Returns:
        A nested dictionary keyed by channel name.  Each value is a dict
        with keys ``"count"``, ``"fraction"``, ``"mean_lifetime"``,
        and ``"std_lifetime"``.

    Example:
        >>> ratios = {"α": 0.7, "β": 0.3}
        >>> data = generate_multichannel_decay(10000, 0.1, ratios, random_seed=42)
        >>> stats = analyze_channels(data)
        >>> print(f"α count: {stats['α']['count']}")
    """
    decay_times = decay_data["decay_times"]
    channels = decay_data["channels"]
    n_total = len(decay_times)

    channel_stats: Dict[str, Dict[str, float]] = {}

    for channel_name in decay_data["branching_ratios"]:
        mask = channels == channel_name
        channel_times = decay_times[mask]
        count = int(np.sum(mask))

        channel_stats[channel_name] = {
            "count": count,
            "fraction": count / n_total if n_total > 0 else 0.0,
            "mean_lifetime": float(np.mean(channel_times)) if count > 0 else 0.0,
            "std_lifetime": float(np.std(channel_times)) if count > 0 else 0.0,
        }

        logger.debug(
            "Channel '%s': %d decays (%.1f%%), mean τ = %.4f s",
            channel_name,
            count,
            channel_stats[channel_name]["fraction"] * 100,
            channel_stats[channel_name]["mean_lifetime"],
        )

    return channel_stats


def run_multichannel_simulation(
    n_particles: Optional[int] = None,
    decay_constant: Optional[float] = None,
    branching_ratios: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a complete multi-channel decay simulation with analysis.

    Missing parameters are filled from ``SimulationConfig`` defaults.

    Args:
        n_particles: Number of particles. Defaults to config value.
        decay_constant: Decay rate λ (s⁻¹). Defaults to config value.
        branching_ratios: Channel → probability mapping. Defaults to
            ``SimulationConfig.DEFAULT_BRANCHING_RATIOS``.
        random_seed: RNG seed. Defaults to config value.

    Returns:
        A dictionary containing:
            - All keys from :func:`generate_multichannel_decay`.
            - ``"channel_stats"``: Per-channel analysis from
              :func:`analyze_channels`.

    Example:
        >>> results = run_multichannel_simulation(n_particles=5000)
        >>> for ch, st in results["channel_stats"].items():
        ...     print(f"{ch}: {st['count']} decays")
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(
        __file__).resolve().parent.parent))
    from config import SimulationConfig

    if n_particles is None:
        n_particles = SimulationConfig.DEFAULT_N_PARTICLES
    if decay_constant is None:
        decay_constant = SimulationConfig.DEFAULT_DECAY_CONSTANT
    if branching_ratios is None:
        branching_ratios = SimulationConfig.DEFAULT_BRANCHING_RATIOS
    if random_seed is None:
        random_seed = SimulationConfig.RANDOM_SEED

    logger.info("Starting multi-channel simulation")

    decay_data = generate_multichannel_decay(
        n_particles, decay_constant, branching_ratios, random_seed
    )
    channel_stats = analyze_channels(decay_data)

    decay_data["channel_stats"] = channel_stats
    return decay_data


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_branching_ratios(branching_ratios: Dict[str, float]) -> None:
    """Ensure branching ratios are valid probabilities summing to 1."""
    if not isinstance(branching_ratios, dict) or len(branching_ratios) == 0:
        raise ValueError(
            "branching_ratios must be a non-empty dictionary, "
            f"got {type(branching_ratios).__name__}"
        )

    for name, prob in branching_ratios.items():
        if not isinstance(prob, (int, float)) or prob < 0:
            raise ValueError(
                f"Branching ratio for '{name}' must be non-negative, got {prob}"
            )

    total = sum(branching_ratios.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Branching ratios must sum to 1.0, got {total:.6f}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    results = run_multichannel_simulation()

    print("=" * 60)
    print("MULTI-CHANNEL DECAY SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Particles simulated : {results['n_particles']:,}")
    print(f"  Decay constant (λ)  : {results['decay_constant']:.4f} s⁻¹")
    print(
        f"  Channels            : {list(results['branching_ratios'].keys())}")
    print()

    for channel, stats in results["channel_stats"].items():
        expected = results["branching_ratios"][channel]
        print(f"  Channel '{channel}':")
        print(f"    Count           : {stats['count']:,}")
        print(
            f"    Observed ratio  : {stats['fraction']:.4f}  (expected {expected:.4f})")
        print(f"    Mean lifetime   : {stats['mean_lifetime']:.4f} s")
        print(f"    Std deviation   : {stats['std_lifetime']:.4f} s")
        print()

    print("=" * 60)
