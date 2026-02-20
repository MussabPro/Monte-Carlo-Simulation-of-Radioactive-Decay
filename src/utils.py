"""
Utility functions for the Monte Carlo decay simulation framework.

Provides helpers for timing, logging setup, result serialisation,
and reproducible RNG creation that are used across multiple modules.

Example:
    >>> from src.utils import create_rng, format_duration
    >>> rng = create_rng(seed=42)
    >>> print(format_duration(3.456))
    '3.46 s'
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def create_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a reproducible NumPy random number generator.

    Uses the modern ``np.random.default_rng`` API for high-quality
    randomness and explicit seed control.

    Args:
        seed: Integer seed, or ``None`` for non-deterministic behaviour.

    Returns:
        A ``numpy.random.Generator`` instance.

    Example:
        >>> rng = create_rng(42)
        >>> sample = rng.exponential(10.0, size=5)
        >>> print(sample.shape)
        (5,)
    """
    rng = np.random.default_rng(seed=seed)
    logger.debug("Created RNG with seed=%s", seed)
    return rng


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = "Operation") -> Generator[None, None, None]:
    """Context manager that logs wall-clock time for a code block.

    Args:
        label: Human-readable label for the timed operation.

    Yields:
        Nothing — used as a context manager.

    Example:
        >>> with timer("Simulation"):
        ...     _ = [i ** 2 for i in range(10_000)]
    """
    start = time.perf_counter()
    logger.info("%s — started", label)
    yield
    elapsed = time.perf_counter() - start
    logger.info("%s — completed in %s", label, format_duration(elapsed))


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A string like ``"1.23 s"``, ``"45.6 ms"``, or ``"2 min 15.3 s"``.

    Example:
        >>> format_duration(0.0034)
        '3.40 ms'
        >>> format_duration(125.7)
        '2 min 5.70 s'
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.2f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes} min {remaining:.2f} s"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> None:
    """Configure the root logger with a consistent format.

    Args:
        level: Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
        log_file: Optional path to a log file.  If provided, log messages
            are written both to the console and to the file.

    Example:
        >>> setup_logging(level=logging.DEBUG)
    """
    fmt = "%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s"
    handlers: list = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    logger.info("Logging initialised (level=%s)", logging.getLevelName(level))


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def save_results_json(
    results: Dict[str, Any],
    filepath: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialise simulation results to a JSON file.

    NumPy arrays are converted to lists, and a ``_metadata`` key is
    injected with a timestamp and any extra information.

    Args:
        results: Dictionary of simulation results.
        filepath: Destination file path.
        metadata: Optional extra metadata to include.

    Returns:
        The ``Path`` where the file was written.

    Example:
        >>> from pathlib import Path
        >>> save_results_json({"mean": 10.1}, Path("/tmp/test.json"))
        PosixPath('/tmp/test.json')
    """
    serialisable = _make_serialisable(results)

    meta = {
        "saved_at": datetime.now().isoformat(),
    }
    if metadata:
        meta.update(metadata)
    serialisable["_metadata"] = meta

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", filepath)
    return filepath


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_decay_times(decay_times: np.ndarray) -> None:
    """Raise if *decay_times* is not a valid 1-D array of positive values.

    Args:
        decay_times: Array to validate.

    Raises:
        ValueError: If the array is empty, not 1-D, or contains
            non-positive values.
    """
    if not isinstance(decay_times, np.ndarray):
        raise TypeError(
            f"Expected np.ndarray, got {type(decay_times).__name__}"
        )
    if decay_times.ndim != 1:
        raise ValueError(
            f"decay_times must be 1-D, got {decay_times.ndim}-D"
        )
    if len(decay_times) == 0:
        raise ValueError("decay_times must not be empty")
    if np.any(decay_times < 0):
        raise ValueError("decay_times contains negative values")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_simulation_header(
    title: str,
    n_particles: int,
    decay_constant: float,
) -> None:
    """Print a formatted header for simulation output.

    Args:
        title: Title string for the header.
        n_particles: Number of particles.
        decay_constant: Decay constant λ.

    Example:
        >>> print_simulation_header("Basic Simulation", 10000, 0.1)
    """
    width = 60
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  Particles           : {n_particles:,}")
    print(f"  Decay constant (λ)  : {decay_constant:.4f} s⁻¹")
    print(f"  Mean lifetime (1/λ) : {1 / decay_constant:.4f} s")
    print(f"  Half-life (ln2/λ)   : {0.693 / decay_constant:.4f} s")
    print("-" * width)
