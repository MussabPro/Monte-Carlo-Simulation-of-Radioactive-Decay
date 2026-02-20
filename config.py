"""
Centralized configuration for Monte Carlo particle decay simulations.

This module defines all configurable parameters used across the simulation
framework, including physics constants, plotting styles, animation settings,
and file paths. All hard-coded values are centralized here for easy
modification and reproducibility.

Usage:
    >>> from config import SimulationConfig, PlotConfig
    >>> sim = SimulationConfig()
    >>> print(sim.DEFAULT_N_PARTICLES)
    10000
"""

from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

class PathConfig:
    """Cross-platform file paths for project directories and outputs.

    Uses ``pathlib.Path`` so that all paths work on Windows, macOS, and Linux
    without modification.

    Attributes:
        PROJECT_ROOT: Absolute path to the project root directory.
        SRC_DIR: Path to the source code directory.
        ANIMATIONS_DIR: Path to animation module directory.
        EXAMPLES_DIR: Path to example scripts directory.
        TESTS_DIR: Path to test suite directory.
        DOCS_DIR: Path to documentation directory.
        RESULTS_DIR: Path to top-level results directory.
        PLOTS_DIR: Path to generated plot images.
        ANIMATION_OUTPUT_DIR: Path to generated animation files.
        BENCHMARKS_DIR: Path to benchmark result files.
    """

    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    SRC_DIR: Path = PROJECT_ROOT / "src"
    ANIMATIONS_DIR: Path = PROJECT_ROOT / "animations"
    EXAMPLES_DIR: Path = PROJECT_ROOT / "examples"
    TESTS_DIR: Path = PROJECT_ROOT / "tests"
    DOCS_DIR: Path = PROJECT_ROOT / "docs"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    PLOTS_DIR: Path = RESULTS_DIR / "plots"
    ANIMATION_OUTPUT_DIR: Path = RESULTS_DIR / "animations"
    BENCHMARKS_DIR: Path = RESULTS_DIR / "benchmarks"

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all output directories if they do not already exist.

        This is called automatically before saving any results so that
        downstream code never needs to worry about missing folders.

        Example:
            >>> PathConfig.ensure_directories()
        """
        for directory in (
            cls.RESULTS_DIR,
            cls.PLOTS_DIR,
            cls.ANIMATION_OUTPUT_DIR,
            cls.BENCHMARKS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

class SimulationConfig:
    """Physics and simulation parameters for Monte Carlo decay.

    All particle counts, decay constants, and random-seed settings live here
    so that every module draws from a single source of truth.

    Attributes:
        DEFAULT_N_PARTICLES: Default number of particles to simulate.
        PARTICLE_COUNTS: Preset particle counts for parameter scans.
        DEFAULT_DECAY_CONSTANT: Default decay rate lambda (s^-1).
        DECAY_CONSTANTS: Preset decay constants for parameter scans.
        DEFAULT_BRANCHING_RATIOS: Default multi-channel branching ratios
            modelled after B-meson decay channels.
        RANDOM_SEED: Global seed for ``numpy.random`` reproducibility.
        DEFAULT_DETECTOR_EFFICIENCY: Probability of detecting each decay.
        DEFAULT_ENERGY_RESOLUTION: Gaussian smearing sigma (seconds).
        DEFAULT_MIN_DECAY_TIME: Lower acceptance cut on decay time (s).
        DEFAULT_MAX_DECAY_TIME: Upper acceptance cut on decay time (s).
        N_BOOTSTRAP: Number of bootstrap resamples for uncertainty estimation.
    """

    # Particle counts
    DEFAULT_N_PARTICLES: int = 10_000
    PARTICLE_COUNTS: List[int] = [100, 1_000, 10_000, 100_000]

    # Decay constants (lambda, s^-1)
    DEFAULT_DECAY_CONSTANT: float = 0.1
    DECAY_CONSTANTS: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25]

    # Multi-channel branching ratios (B-meson inspired)
    DEFAULT_BRANCHING_RATIOS: Dict[str, float] = {
        "π⁺π⁻": 0.60,
        "K⁺π⁻": 0.30,
        "Other": 0.10,
    }

    # Reproducibility
    RANDOM_SEED: int = 42

    # Detector modelling defaults
    DEFAULT_DETECTOR_EFFICIENCY: float = 0.85
    DEFAULT_ENERGY_RESOLUTION: float = 0.1
    DEFAULT_MIN_DECAY_TIME: float = 0.0
    DEFAULT_MAX_DECAY_TIME: float = 100.0

    # Statistical analysis
    N_BOOTSTRAP: int = 1_000
    ACCURACY_THRESHOLD_PERCENT: float = 5.0
    CHI2_P_VALUE_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# Plotting configuration
# ---------------------------------------------------------------------------

class PlotConfig:
    """Matplotlib styling and output settings for all visualizations.

    Consistent styling ensures that every plot produced by the framework
    looks professional and publication-ready.

    Attributes:
        FIGURE_SIZE: Default ``(width, height)`` in inches.
        FIGURE_SIZE_WIDE: Wider figure size for side-by-side panels.
        DPI: Dots per inch for saved figures.
        STYLE: Matplotlib style sheet name.
        FONT_SIZE_TITLE: Font size for plot titles.
        FONT_SIZE_LABEL: Font size for axis labels.
        FONT_SIZE_TICK: Font size for tick labels.
        FONT_SIZE_LEGEND: Font size for legend entries.
        COLOR_SIMULATED: Colour for simulated data.
        COLOR_THEORETICAL: Colour for theoretical curves.
        COLOR_DETECTED: Colour for detector-filtered data.
        CHANNEL_COLORS: Per-channel colour palette.
        HISTOGRAM_ALPHA: Transparency for histogram bars.
        HISTOGRAM_BINS: Default number of histogram bins.
        GRID_ALPHA: Transparency for background grid lines.
        LINE_WIDTH: Default line width for curves.
        SAVE_FORMAT: Default file format for saved figures.
    """

    # Figure dimensions
    FIGURE_SIZE: Tuple[int, int] = (12, 8)
    FIGURE_SIZE_WIDE: Tuple[int, int] = (16, 6)
    DPI: int = 300

    # Style
    STYLE: str = "seaborn-v0_8-whitegrid"

    # Font sizes
    FONT_SIZE_TITLE: int = 16
    FONT_SIZE_LABEL: int = 14
    FONT_SIZE_TICK: int = 12
    FONT_SIZE_LEGEND: int = 12

    # Colour palette
    COLOR_SIMULATED: str = "#3498db"
    COLOR_THEORETICAL: str = "#e74c3c"
    COLOR_DETECTED: str = "#f39c12"
    CHANNEL_COLORS: List[str] = [
        "#2ecc71",  # green
        "#9b59b6",  # purple
        "#e67e22",  # orange
        "#1abc9c",  # teal
        "#e74c3c",  # red
    ]

    # Histogram
    HISTOGRAM_ALPHA: float = 0.7
    HISTOGRAM_BINS: int = 50
    HISTOGRAM_EDGE_COLOR: str = "black"

    # Grid and lines
    GRID_ALPHA: float = 0.3
    LINE_WIDTH: float = 2.0

    # Output
    SAVE_FORMAT: str = "png"


# ---------------------------------------------------------------------------
# Animation configuration
# ---------------------------------------------------------------------------

class AnimationConfig:
    """Settings for 2-D grid, 3-D cloud, and interactive animations.

    Attributes:
        FPS: Frames per second for saved videos.
        MAX_TIME: Default simulation duration (seconds of physics time).
        BITRATE: Video bitrate in kbps.
        VIDEO_FORMAT: Container format for saved videos.
        GIF_ENABLED: Whether to also export a GIF alongside the video.
        GRID_2D_PARTICLE_SIZE_ALIVE: Marker size for living particles (2-D).
        GRID_2D_PARTICLE_SIZE_DECAYED: Marker size for decayed particles (2-D).
        GRID_2D_COLOR_ALIVE: Colour for living particles (2-D grid).
        GRID_2D_COLOR_DECAYED: Colour for decayed particles (2-D grid).
        SPACE_3D_FPS: Frames per second for the 3-D animation.
        SPACE_3D_ROTATION_SPEED: Degrees of azimuth rotation per frame.
        SPACE_3D_SPREAD: Standard deviation of initial 3-D positions.
        SPACE_3D_FADE_DURATION: Seconds over which decayed particles fade.
        PYGAME_WIDTH: Window width for interactive simulation.
        PYGAME_HEIGHT: Window height for interactive simulation.
        PYGAME_FPS: Target frame rate for the Pygame loop.
        PYGAME_PARTICLE_RADIUS: Radius of drawn particles (pixels).
    """

    # General video settings
    FPS: int = 10
    MAX_TIME: float = 50.0
    BITRATE: int = 1800
    VIDEO_FORMAT: str = "mp4"
    GIF_ENABLED: bool = True

    # 2-D grid animation
    GRID_2D_PARTICLE_SIZE_ALIVE: int = 30
    GRID_2D_PARTICLE_SIZE_DECAYED: int = 15
    GRID_2D_COLOR_ALIVE: str = "lime"
    GRID_2D_COLOR_DECAYED: str = "red"
    GRID_2D_FIGURE_SIZE: Tuple[int, int] = (14, 6)

    # 3-D space animation
    SPACE_3D_FPS: int = 15
    SPACE_3D_ROTATION_SPEED: float = 2.0
    SPACE_3D_SPREAD: float = 5.0
    SPACE_3D_FADE_DURATION: float = 5.0
    SPACE_3D_FIGURE_SIZE: Tuple[int, int] = (12, 10)
    SPACE_3D_BITRATE: int = 2000

    # Pygame interactive simulation
    PYGAME_WIDTH: int = 1200
    PYGAME_HEIGHT: int = 800
    PYGAME_FPS: int = 60
    PYGAME_PARTICLE_RADIUS: int = 4

    # Pygame colours (RGB tuples)
    PYGAME_BG_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PYGAME_COLOR_ALIVE: Tuple[int, int, int] = (0, 255, 0)
    PYGAME_COLOR_DECAYED: Tuple[int, int, int] = (255, 0, 0)
    PYGAME_COLOR_WARNING: Tuple[int, int, int] = (255, 255, 0)
    PYGAME_COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)


# ---------------------------------------------------------------------------
# CITATION.cff (academic citation metadata)
# ---------------------------------------------------------------------------

CITATION_CFF = """\
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "Monte Carlo Simulation of Radioactive Particle Decay"
version: 1.0.0
authors:
  - family-names: Omair
    given-names: Mussab
    affiliation: University of Gujrat
date-released: 2026-02-21
license: MIT
repository-code: https://github.com/MussabPro/Monte-Carlo-Simulation-of-Radioactive-Decay
keywords:
  - monte-carlo
  - particle-physics
  - radioactive-decay
  - simulation
"""


# ---------------------------------------------------------------------------
# Convenience: ensure output directories exist on import
# ---------------------------------------------------------------------------
PathConfig.ensure_directories()
