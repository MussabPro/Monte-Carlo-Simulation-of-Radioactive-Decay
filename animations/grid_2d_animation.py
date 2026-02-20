"""
2D grid animation of particle decay.

Displays particles on a grid, with living particles shown in green and
decayed particles fading to red.  A real-time decay curve is plotted
alongside the grid.

Usage:
    python -m animations.grid_2d_animation
    python animations/grid_2d_animation.py --n-particles 500 --fps 15
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional
from datetime import datetime
import logging
import argparse
from config import AnimationConfig, PathConfig, PlotConfig, SimulationConfig
import sys
from pathlib import Path

# Ensure the project root is on sys.path for both direct execution
# (python animations/grid_2d_animation.py) and package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger(__name__)


class Grid2DAnimation:
    """Animated 2-D grid showing particle decay over time.

    Each particle occupies one cell of a square grid.  Living particles
    are drawn as green dots; decayed particles shrink and turn red before
    fading out.  A second panel plots the surviving-particle count
    alongside the theoretical curve.

    Args:
        n_particles: Number of particles to simulate.
        decay_constant: Decay rate λ (s⁻¹).
        max_time: Duration of the animation in physics time (seconds).
        fps: Frames per second for the saved video.
        random_seed: Optional RNG seed for reproducibility.

    Example:
        >>> anim = Grid2DAnimation(n_particles=500, decay_constant=0.1)
        >>> anim.setup()
        >>> anim.save_video("decay_grid.mp4")
    """

    def __init__(
        self,
        n_particles: int = SimulationConfig.DEFAULT_N_PARTICLES,
        decay_constant: float = SimulationConfig.DEFAULT_DECAY_CONSTANT,
        max_time: float = AnimationConfig.MAX_TIME,
        fps: int = AnimationConfig.FPS,
        random_seed: Optional[int] = SimulationConfig.RANDOM_SEED,
    ) -> None:
        self.n_particles = n_particles
        self.decay_constant = decay_constant
        self.max_time = max_time
        self.fps = fps
        self.random_seed = random_seed

        # State populated by setup()
        self.decay_times: Optional[np.ndarray] = None
        self.positions: Optional[np.ndarray] = None
        self.fig: Optional[plt.Figure] = None
        self.anim: Optional[animation.FuncAnimation] = None

    def setup(self) -> None:
        """Generate decay times, build the grid, and prepare the figure."""
        rng = np.random.default_rng(self.random_seed)
        self.decay_times = rng.exponential(
            1.0 / self.decay_constant, size=self.n_particles
        )

        grid_size = int(np.ceil(np.sqrt(self.n_particles)))
        coords: List[tuple] = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(coords) < self.n_particles:
                    coords.append((i, j))
        self.positions = np.array(coords)
        self.grid_size = grid_size

        dt = 1.0 / self.fps
        self.time_steps = np.arange(0, self.max_time, dt)
        self.remaining_counts: List[int] = []
        self.times_recorded: List[float] = []

        self.fig, (self.ax_grid, self.ax_curve) = plt.subplots(
            1, 2, figsize=AnimationConfig.GRID_2D_FIGURE_SIZE
        )

        logger.info(
            "Grid2DAnimation setup: N=%d, λ=%.3f, grid=%dx%d, frames=%d",
            self.n_particles,
            self.decay_constant,
            self.grid_size,
            self.grid_size,
            len(self.time_steps),
        )

    def _update_frame(self, frame: int) -> None:
        """Render a single animation frame."""
        current_time = self.time_steps[frame]
        alive = self.decay_times > current_time
        num_alive = int(np.sum(alive))
        num_decayed = self.n_particles - num_alive

        self.remaining_counts.append(num_alive)
        self.times_recorded.append(current_time)

        self.ax_grid.clear()
        self.ax_curve.clear()

        # --- Grid panel ---
        self.ax_grid.set_xlim(-1, self.grid_size)
        self.ax_grid.set_ylim(-1, self.grid_size)
        self.ax_grid.set_aspect("equal")
        self.ax_grid.set_title(
            f"Particle Decay (t = {current_time:.1f} s)",
            fontsize=PlotConfig.FONT_SIZE_TITLE,
            fontweight="bold",
        )
        self.ax_grid.axis("off")

        alive_pos = self.positions[alive]
        dead_pos = self.positions[~alive]

        if len(alive_pos) > 0:
            self.ax_grid.scatter(
                alive_pos[:, 0],
                alive_pos[:, 1],
                c=AnimationConfig.GRID_2D_COLOR_ALIVE,
                s=AnimationConfig.GRID_2D_PARTICLE_SIZE_ALIVE,
                alpha=0.8,
                edgecolors="darkgreen",
                linewidth=0.5,
            )
        if len(dead_pos) > 0:
            self.ax_grid.scatter(
                dead_pos[:, 0],
                dead_pos[:, 1],
                c=AnimationConfig.GRID_2D_COLOR_DECAYED,
                s=AnimationConfig.GRID_2D_PARTICLE_SIZE_DECAYED,
                alpha=0.2,
                marker="x",
            )

        stats_text = (
            f"Living: {num_alive}\n"
            f"Decayed: {num_decayed}\n"
            f"Decay Rate: {num_decayed / self.n_particles * 100:.1f}%"
        )
        self.ax_grid.text(
            0.02,
            0.98,
            stats_text,
            transform=self.ax_grid.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # --- Curve panel ---
        self.ax_curve.set_xlim(0, self.max_time)
        self.ax_curve.set_ylim(0, self.n_particles)
        self.ax_curve.set_xlabel(
            "Time (seconds)", fontsize=PlotConfig.FONT_SIZE_LABEL)
        self.ax_curve.set_ylabel(
            "Particles Remaining", fontsize=PlotConfig.FONT_SIZE_LABEL
        )
        self.ax_curve.set_title(
            "Decay Curve", fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight="bold"
        )
        self.ax_curve.grid(True, alpha=PlotConfig.GRID_ALPHA)

        if len(self.times_recorded) > 1:
            self.ax_curve.plot(
                self.times_recorded,
                self.remaining_counts,
                color=PlotConfig.COLOR_SIMULATED,
                linewidth=PlotConfig.LINE_WIDTH,
                label="Simulation",
            )

        t_theory = np.linspace(0, self.max_time, 200)
        n_theory = self.n_particles * np.exp(-self.decay_constant * t_theory)
        self.ax_curve.plot(
            t_theory,
            n_theory,
            color=PlotConfig.COLOR_THEORETICAL,
            linewidth=PlotConfig.LINE_WIDTH,
            linestyle="--",
            alpha=0.7,
            label="Theory",
        )
        self.ax_curve.axvline(
            current_time, color="green", linestyle=":", linewidth=1.5, alpha=0.5
        )
        self.ax_curve.legend(loc="upper right")

        if num_decayed > 0:
            decayed_times = self.decay_times[~alive]
            mean_lt = float(np.mean(decayed_times))
            info = (
                f"Observed Mean τ: {mean_lt:.2f} s\n"
                f"Theoretical τ: {1 / self.decay_constant:.2f} s\n"
                f"Half-life: {0.693 / self.decay_constant:.2f} s"
            )
            self.ax_curve.text(
                0.98,
                0.02,
                info,
                transform=self.ax_curve.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )

        self.fig.tight_layout()

        if frame % 50 == 0:
            progress = (frame + 1) / len(self.time_steps) * 100
            logger.info("Frame %d / %d (%.0f%%)", frame,
                        len(self.time_steps), progress)

    def run(self) -> animation.FuncAnimation:
        """Build and return the FuncAnimation object."""
        if self.fig is None:
            self.setup()

        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(self.time_steps),
            interval=1000 / self.fps,
            repeat=False,
        )
        return self.anim

    def save_video(self, filepath: Optional[str] = None) -> Path:
        """Save the animation as an MP4 video.

        Args:
            filepath: Destination path.  Defaults to ``results/animations/``.

        Returns:
            The path where the video was saved.
        """
        if self.anim is None:
            self.run()

        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(
                PathConfig.ANIMATION_OUTPUT_DIR / f"grid_2d_{ts}.mp4"
            )

        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving MP4 to %s …", out)
        writer = animation.FFMpegWriter(
            fps=self.fps,
            metadata={"artist": "Mussab Omair"},
            bitrate=AnimationConfig.BITRATE,
        )
        self.anim.save(str(out), writer=writer)
        logger.info("MP4 saved: %s", out)
        return out

    def save_gif(self, filepath: Optional[str] = None) -> Path:
        """Save the animation as a GIF.

        Args:
            filepath: Destination path.  Defaults to ``results/animations/``.

        Returns:
            The path where the GIF was saved.
        """
        if self.anim is None:
            self.run()

        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(
                PathConfig.ANIMATION_OUTPUT_DIR / f"grid_2d_{ts}.gif"
            )

        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving GIF to %s …", out)
        self.anim.save(str(out), writer="pillow", fps=self.fps)
        logger.info("GIF saved: %s", out)
        return out


def main() -> None:
    """CLI entry point for generating the 2-D grid animation."""
    parser = argparse.ArgumentParser(
        description="Generate a 2-D grid particle decay animation."
    )
    parser.add_argument(
        "--n-particles", type=int, default=1000, help="Number of particles"
    )
    parser.add_argument(
        "--decay-constant", type=float, default=0.1, help="Decay constant λ (s⁻¹)"
    )
    parser.add_argument("--max-time", type=float,
                        default=50.0, help="Duration (s)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--gif", action="store_true", help="Also save as GIF")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    anim_obj = Grid2DAnimation(
        n_particles=args.n_particles,
        decay_constant=args.decay_constant,
        max_time=args.max_time,
        fps=args.fps,
        random_seed=args.seed,
    )
    anim_obj.setup()
    anim_obj.run()
    anim_obj.save_video(args.output)

    if args.gif:
        anim_obj.save_gif()

    print("Done.")


if __name__ == "__main__":
    main()
