"""
Visualization module for Monte Carlo decay simulations.

Provides publication-quality plotting functions with consistent styling
drawn from ``PlotConfig``.  Every function can optionally save the
figure to the ``results/plots/`` directory.

All plots use Matplotlib and are styled for a professional, research-
portfolio presentation.

Example:
    >>> from src.visualization import plot_decay_histogram
    >>> import numpy as np
    >>> times = np.random.exponential(10.0, size=5000)
    >>> plot_decay_histogram(times, decay_constant=0.1)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _get_configs():
    """Lazy-load configuration to avoid circular imports."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import PathConfig, PlotConfig
    return PlotConfig, PathConfig


def _apply_style() -> None:
    """Apply the project-wide Matplotlib style."""
    plot_cfg, _ = _get_configs()
    try:
        plt.style.use(plot_cfg.STYLE)
    except OSError:
        logger.warning("Style '%s' not found; using default", plot_cfg.STYLE)


def _save_figure(
    fig: plt.Figure,
    save_path: Optional[str] = None,
    default_name: str = "plot",
) -> Optional[Path]:
    """Save *fig* to *save_path* or to the default plots directory.

    If *save_path* is ``None``, the figure is not saved.

    Args:
        fig: Matplotlib figure to save.
        save_path: Explicit file path, or ``None`` to skip saving.
        default_name: Fallback filename stem when *save_path* is a
            directory.

    Returns:
        The ``Path`` where the figure was saved, or ``None``.
    """
    if save_path is None:
        return None

    plot_cfg, path_cfg = _get_configs()
    out = Path(save_path)

    if out.is_dir():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out / f"{default_name}_{timestamp}.{plot_cfg.SAVE_FORMAT}"

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=plot_cfg.DPI, bbox_inches="tight")
    logger.info("Figure saved to %s", out)
    return out


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_decay_histogram(
    decay_times: np.ndarray,
    decay_constant: float,
    title: str = "Monte Carlo Simulation of Radioactive Decay",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a normalised histogram of decay times with the theoretical PDF.

    Args:
        decay_times: 1-D array of simulated decay times.
        decay_constant: Decay rate λ (s⁻¹) for the theory overlay.
        title: Figure title.
        save_path: Path to save the figure, or ``None``.
        show: Whether to call ``plt.show()``.

    Returns:
        The Matplotlib ``Figure`` object.

    Example:
        >>> import numpy as np
        >>> times = np.random.exponential(10.0, size=5000)
        >>> fig = plot_decay_histogram(times, 0.1, show=False)
    """
    _apply_style()
    plot_cfg, _ = _get_configs()

    fig, ax = plt.subplots(figsize=plot_cfg.FIGURE_SIZE)

    # Histogram
    ax.hist(
        decay_times,
        bins=plot_cfg.HISTOGRAM_BINS,
        density=True,
        alpha=plot_cfg.HISTOGRAM_ALPHA,
        color=plot_cfg.COLOR_SIMULATED,
        edgecolor=plot_cfg.HISTOGRAM_EDGE_COLOR,
        label="Simulated Data",
    )

    # Theoretical PDF overlay
    t = np.linspace(0, float(np.max(decay_times)), 500)
    pdf = decay_constant * np.exp(-decay_constant * t)
    ax.plot(
        t,
        pdf,
        color=plot_cfg.COLOR_THEORETICAL,
        linewidth=plot_cfg.LINE_WIDTH,
        label="Theoretical Curve",
    )

    ax.set_xlabel("Decay Time (seconds)", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax.set_ylabel("Probability Density", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax.set_title(title, fontsize=plot_cfg.FONT_SIZE_TITLE)
    ax.legend(fontsize=plot_cfg.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=plot_cfg.GRID_ALPHA)
    ax.tick_params(labelsize=plot_cfg.FONT_SIZE_TICK)

    fig.tight_layout()
    _save_figure(fig, save_path, default_name="decay_histogram")

    if show:
        plt.show()

    return fig


def plot_decay_curve(
    time_points: np.ndarray,
    remaining_counts: np.ndarray,
    theoretical_curve: np.ndarray,
    n_particles: int,
    decay_constant: float,
    title: str = "Particle Decay Curve",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot observed vs. theoretical surviving-particle curves.

    Args:
        time_points: 1-D array of evaluation times.
        remaining_counts: Observed particle counts at each time.
        theoretical_curve: Theoretical N(t) at each time.
        n_particles: Initial number of particles.
        decay_constant: Decay rate λ (s⁻¹).
        title: Figure title.
        save_path: Path to save the figure, or ``None``.
        show: Whether to call ``plt.show()``.

    Returns:
        The Matplotlib ``Figure`` object.

    Example:
        >>> from src.basic_simulation import run_basic_simulation
        >>> res = run_basic_simulation(n_particles=1000, decay_constant=0.1)
        >>> fig = plot_decay_curve(
        ...     res["time_points"], res["remaining"],
        ...     res["theoretical_curve"], res["n_particles"],
        ...     res["decay_constant"], show=False,
        ... )
    """
    _apply_style()
    plot_cfg, _ = _get_configs()

    fig, ax = plt.subplots(figsize=plot_cfg.FIGURE_SIZE)

    ax.plot(
        time_points,
        remaining_counts,
        color=plot_cfg.COLOR_SIMULATED,
        linewidth=plot_cfg.LINE_WIDTH,
        label="Simulation",
    )
    ax.plot(
        time_points,
        theoretical_curve,
        color=plot_cfg.COLOR_THEORETICAL,
        linewidth=plot_cfg.LINE_WIDTH,
        linestyle="--",
        label="Theory: N₀·exp(−λt)",
    )

    ax.set_xlabel("Time (seconds)", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax.set_ylabel("Particles Remaining", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax.set_title(title, fontsize=plot_cfg.FONT_SIZE_TITLE)
    ax.legend(fontsize=plot_cfg.FONT_SIZE_LEGEND)
    ax.grid(True, alpha=plot_cfg.GRID_ALPHA)
    ax.tick_params(labelsize=plot_cfg.FONT_SIZE_TICK)

    # Annotation box
    stats_text = (
        f"N₀ = {n_particles:,}\n"
        f"λ = {decay_constant} s⁻¹\n"
        f"τ = {1 / decay_constant:.1f} s\n"
        f"t½ = {0.693 / decay_constant:.1f} s"
    )
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=plot_cfg.FONT_SIZE_TICK,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    fig.tight_layout()
    _save_figure(fig, save_path, default_name="decay_curve")

    if show:
        plt.show()

    return fig


def plot_multichannel_comparison(
    channel_data: Dict[str, Dict[str, Any]],
    decay_constant: float,
    title: str = "Multi-Channel Decay Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Bar chart comparing observed vs. expected branching fractions.

    Also shows per-channel mean lifetimes as a secondary axis.

    Args:
        channel_data: Dictionary from
            :func:`src.multichannel_simulation.analyze_channels`.
        decay_constant: Decay rate λ (s⁻¹) for reference line.
        title: Figure title.
        save_path: Path to save the figure, or ``None``.
        show: Whether to call ``plt.show()``.

    Returns:
        The Matplotlib ``Figure`` object.

    Example:
        >>> from src.multichannel_simulation import run_multichannel_simulation
        >>> res = run_multichannel_simulation(n_particles=5000)
        >>> fig = plot_multichannel_comparison(
        ...     res["channel_stats"], res["decay_constant"], show=False,
        ... )
    """
    _apply_style()
    plot_cfg, _ = _get_configs()

    channels = list(channel_data.keys())
    fractions = [channel_data[ch]["fraction"] for ch in channels]
    mean_lifetimes = [channel_data[ch]["mean_lifetime"] for ch in channels]
    colors = plot_cfg.CHANNEL_COLORS[: len(channels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_cfg.FIGURE_SIZE_WIDE)

    # Panel 1: Branching fractions
    bars = ax1.bar(
        channels,
        fractions,
        color=colors,
        alpha=plot_cfg.HISTOGRAM_ALPHA,
        edgecolor="black",
    )
    ax1.set_xlabel("Decay Channel", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax1.set_ylabel("Observed Fraction", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax1.set_title("Branching Fractions", fontsize=plot_cfg.FONT_SIZE_TITLE)
    ax1.grid(True, alpha=plot_cfg.GRID_ALPHA, axis="y")
    ax1.tick_params(labelsize=plot_cfg.FONT_SIZE_TICK)

    # Annotate bars with values
    for bar, frac in zip(bars, fractions):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{frac:.3f}",
            ha="center",
            va="bottom",
            fontsize=plot_cfg.FONT_SIZE_TICK,
        )

    # Panel 2: Per-channel mean lifetime
    ax2.bar(
        channels,
        mean_lifetimes,
        color=colors,
        alpha=plot_cfg.HISTOGRAM_ALPHA,
        edgecolor="black",
    )
    theoretical_mean = 1.0 / decay_constant
    ax2.axhline(
        theoretical_mean,
        color=plot_cfg.COLOR_THEORETICAL,
        linewidth=plot_cfg.LINE_WIDTH,
        linestyle="--",
        label=f"Theory: τ = {theoretical_mean:.1f} s",
    )
    ax2.set_xlabel("Decay Channel", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax2.set_ylabel("Mean Lifetime (s)", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax2.set_title("Per-Channel Mean Lifetime",
                  fontsize=plot_cfg.FONT_SIZE_TITLE)
    ax2.legend(fontsize=plot_cfg.FONT_SIZE_LEGEND)
    ax2.grid(True, alpha=plot_cfg.GRID_ALPHA, axis="y")
    ax2.tick_params(labelsize=plot_cfg.FONT_SIZE_TICK)

    fig.tight_layout()
    _save_figure(fig, save_path, default_name="multichannel_comparison")

    if show:
        plt.show()

    return fig


def plot_parameter_scan(
    results_dict: Dict[float, np.ndarray],
    title: str = "Parameter Scan: Varying Decay Constant",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Grid of histograms for different decay constants.

    Args:
        results_dict: Mapping of ``{decay_constant: decay_times_array}``.
        title: Overall figure title.
        save_path: Path to save the figure, or ``None``.
        show: Whether to call ``plt.show()``.

    Returns:
        The Matplotlib ``Figure`` object.

    Example:
        >>> import numpy as np
        >>> scan = {lam: np.random.exponential(1/lam, 5000)
        ...         for lam in [0.05, 0.1, 0.2]}
        >>> fig = plot_parameter_scan(scan, show=False)
    """
    _apply_style()
    plot_cfg, _ = _get_configs()

    n_plots = len(results_dict)
    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
    )
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (lam, times) in enumerate(sorted(results_dict.items())):
        ax = axes[idx]
        ax.hist(
            times,
            bins=plot_cfg.HISTOGRAM_BINS,
            density=True,
            alpha=plot_cfg.HISTOGRAM_ALPHA,
            color=plot_cfg.COLOR_SIMULATED,
            edgecolor=plot_cfg.HISTOGRAM_EDGE_COLOR,
        )

        # Theoretical overlay
        t = np.linspace(0, float(np.max(times)), 500)
        pdf = lam * np.exp(-lam * t)
        ax.plot(
            t,
            pdf,
            color=plot_cfg.COLOR_THEORETICAL,
            linewidth=plot_cfg.LINE_WIDTH,
        )

        half_life = 0.693 / lam
        ax.set_title(
            f"λ = {lam}, t½ = {half_life:.2f} s",
            fontsize=plot_cfg.FONT_SIZE_LABEL,
        )
        ax.set_xlabel("Time (s)", fontsize=plot_cfg.FONT_SIZE_TICK)
        ax.set_ylabel("Probability Density", fontsize=plot_cfg.FONT_SIZE_TICK)
        ax.grid(True, alpha=plot_cfg.GRID_ALPHA)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=plot_cfg.FONT_SIZE_TITLE, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path, default_name="parameter_scan")

    if show:
        plt.show()

    return fig


def plot_detector_comparison(
    true_times: np.ndarray,
    detected_times: np.ndarray,
    efficiency: float,
    title: str = "True vs. Detected Distribution",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Side-by-side histograms of true vs. detector-filtered distributions.

    Args:
        true_times: 1-D array of all simulated decay times.
        detected_times: 1-D array of detected decay times.
        efficiency: Detector efficiency used (for the label).
        title: Figure title.
        save_path: Path to save the figure, or ``None``.
        show: Whether to call ``plt.show()``.

    Returns:
        The Matplotlib ``Figure`` object.

    Example:
        >>> import numpy as np
        >>> true = np.random.exponential(10.0, 10000)
        >>> det = true[np.random.random(len(true)) < 0.85]
        >>> fig = plot_detector_comparison(true, det, 0.85, show=False)
    """
    _apply_style()
    plot_cfg, _ = _get_configs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_cfg.FIGURE_SIZE_WIDE)

    mean_true = float(np.mean(true_times))
    mean_det = float(np.mean(detected_times))

    # True distribution
    ax1.hist(
        true_times,
        bins=plot_cfg.HISTOGRAM_BINS,
        alpha=plot_cfg.HISTOGRAM_ALPHA,
        color=plot_cfg.COLOR_SIMULATED,
        edgecolor=plot_cfg.HISTOGRAM_EDGE_COLOR,
        label="All Decays",
    )
    ax1.axvline(
        mean_true,
        color=plot_cfg.COLOR_THEORETICAL,
        linestyle="--",
        linewidth=plot_cfg.LINE_WIDTH,
        label=f"Mean = {mean_true:.2f} s",
    )
    ax1.set_xlabel("Time (s)", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax1.set_ylabel("Count", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax1.set_title("True Distribution", fontsize=plot_cfg.FONT_SIZE_TITLE)
    ax1.legend(fontsize=plot_cfg.FONT_SIZE_LEGEND)
    ax1.grid(True, alpha=plot_cfg.GRID_ALPHA)

    # Detected distribution
    ax2.hist(
        detected_times,
        bins=plot_cfg.HISTOGRAM_BINS,
        alpha=plot_cfg.HISTOGRAM_ALPHA,
        color=plot_cfg.COLOR_DETECTED,
        edgecolor=plot_cfg.HISTOGRAM_EDGE_COLOR,
        label="Detected Decays",
    )
    ax2.axvline(
        mean_det,
        color=plot_cfg.COLOR_THEORETICAL,
        linestyle="--",
        linewidth=plot_cfg.LINE_WIDTH,
        label=f"Mean = {mean_det:.2f} s",
    )
    ax2.set_xlabel("Time (s)", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax2.set_ylabel("Count", fontsize=plot_cfg.FONT_SIZE_LABEL)
    ax2.set_title(
        f"Detected Distribution ({efficiency * 100:.0f}% efficiency)",
        fontsize=plot_cfg.FONT_SIZE_TITLE,
    )
    ax2.legend(fontsize=plot_cfg.FONT_SIZE_LEGEND)
    ax2.grid(True, alpha=plot_cfg.GRID_ALPHA)

    fig.suptitle(title, fontsize=plot_cfg.FONT_SIZE_TITLE +
                 2, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path, default_name="detector_comparison")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import PathConfig, SimulationConfig

    rng = np.random.default_rng(SimulationConfig.RANDOM_SEED)
    decay_constant = SimulationConfig.DEFAULT_DECAY_CONSTANT
    n_particles = SimulationConfig.DEFAULT_N_PARTICLES

    decay_times = rng.exponential(1.0 / decay_constant, size=n_particles)

    print("Generating sample plots …")

    plot_decay_histogram(
        decay_times,
        decay_constant,
        save_path=str(PathConfig.PLOTS_DIR / "demo_histogram.png"),
        show=False,
    )

    from src.basic_simulation import (
        calculate_theoretical_curve,
        compute_remaining_particles,
    )

    t = np.linspace(0, float(np.max(decay_times)) * 1.1, n_particles)
    remaining = compute_remaining_particles(decay_times, t)
    theory = calculate_theoretical_curve(t, decay_constant, n_particles)

    plot_decay_curve(
        t,
        remaining,
        theory,
        n_particles,
        decay_constant,
        save_path=str(PathConfig.PLOTS_DIR / "demo_curve.png"),
        show=False,
    )

    scan = {
        lam: rng.exponential(1.0 / lam, size=5000)
        for lam in SimulationConfig.DECAY_CONSTANTS
    }
    plot_parameter_scan(
        scan,
        save_path=str(PathConfig.PLOTS_DIR / "demo_scan.png"),
        show=False,
    )

    print("All demo plots saved to", PathConfig.PLOTS_DIR)
