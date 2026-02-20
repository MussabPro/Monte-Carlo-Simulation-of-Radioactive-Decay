"""
3D rotating particle cloud animation of decay.

Particles are placed randomly in 3-D space and colour-coded by their
remaining lifetime.  The camera rotates slowly around the cloud while
decayed particles fade out.

Usage:
    python -m animations.space_3d_animation
    python animations/space_3d_animation.py --n-particles 800 --fps 15
"""
# isort: skip_file
from config import AnimationConfig, PathConfig, PlotConfig, SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Tuple
from datetime import datetime
import logging
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger(__name__)


class Space3DAnimation:
    """3-D rotating particle cloud with decay colour gradients.

    Living particles transition green → yellow → orange as their decay
    time approaches.  After decaying they turn red and fade over a
    configurable duration.  The camera rotates around the cloud.

    Args:
        n_particles: Number of particles.
        decay_constant: Decay rate λ (s⁻¹).
        max_time: Physics duration (seconds).
        fps: Frames per second.
        random_seed: RNG seed.

    Example:
        >>> anim = Space3DAnimation(n_particles=500)
        >>> anim.setup()
        >>> anim.save_video()
    """

    def __init__(
        self,
        n_particles: int = SimulationConfig.DEFAULT_N_PARTICLES,
        decay_constant: float = SimulationConfig.DEFAULT_DECAY_CONSTANT,
        max_time: float = AnimationConfig.MAX_TIME,
        fps: int = AnimationConfig.SPACE_3D_FPS,
        random_seed: Optional[int] = SimulationConfig.RANDOM_SEED,
    ) -> None:
        self.n_particles = n_particles
        self.decay_constant = decay_constant
        self.max_time = max_time
        self.fps = fps
        self.random_seed = random_seed

        self.decay_times: Optional[np.ndarray] = None
        self.positions: Optional[np.ndarray] = None
        self.fig: Optional[plt.Figure] = None
        self.ax = None
        self.anim: Optional[animation.FuncAnimation] = None

    def setup(self) -> None:
        """Generate particles and prepare the 3-D figure."""
        rng = np.random.default_rng(self.random_seed)
        self.decay_times = rng.exponential(
            1.0 / self.decay_constant, size=self.n_particles
        )
        spread = AnimationConfig.SPACE_3D_SPREAD
        self.positions = rng.standard_normal((self.n_particles, 3)) * spread

        dt = 1.0 / self.fps
        self.time_steps = np.arange(0, self.max_time, dt)

        self.fig = plt.figure(figsize=AnimationConfig.SPACE_3D_FIGURE_SIZE)
        self.ax = self.fig.add_subplot(111, projection="3d")

        logger.info(
            "Space3DAnimation setup: N=%d, λ=%.3f, frames=%d",
            self.n_particles,
            self.decay_constant,
            len(self.time_steps),
        )

    def _classify_particles(
        self, current_time: float
    ) -> Tuple[List[str], List[float], List[float]]:
        """Return per-particle colour, size, and alpha lists."""
        colors: List[str] = []
        sizes: List[float] = []
        alphas: List[float] = []
        fade_dur = AnimationConfig.SPACE_3D_FADE_DURATION

        for t_decay in self.decay_times:
            if current_time < t_decay:
                frac = min((t_decay - current_time) / 10.0, 1.0)
                if frac > 0.7:
                    colors.append("lime")
                elif frac > 0.3:
                    colors.append("yellow")
                else:
                    colors.append("orange")
                sizes.append(50)
                alphas.append(0.8)
            else:
                fade = max(0, 1 - (current_time - t_decay) / fade_dur)
                if fade > 0:
                    colors.append("red")
                    sizes.append(20)
                    alphas.append(fade * 0.3)
                else:
                    colors.append("red")
                    sizes.append(0)
                    alphas.append(0)

        return colors, sizes, alphas

    def _update_frame(self, frame: int) -> None:
        """Render one 3-D frame."""
        current_time = self.time_steps[frame]
        self.ax.clear()

        limit = AnimationConfig.SPACE_3D_SPREAD * 2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        rotation = AnimationConfig.SPACE_3D_ROTATION_SPEED
        self.ax.view_init(elev=20, azim=frame * rotation)

        colors, sizes, alphas = self._classify_particles(current_time)

        # Draw only visible particles (alpha > 0, size > 0)
        for pos, c, s, a in zip(self.positions, colors, sizes, alphas):
            if s > 0 and a > 0:
                self.ax.scatter(pos[0], pos[1], pos[2], c=c, s=s, alpha=a)

        num_alive = int(np.sum(self.decay_times > current_time))
        num_decayed = self.n_particles - num_alive

        title = (
            f"3D Particle Decay Simulation\n"
            f"Time: {current_time:.1f} s | "
            f"Living: {num_alive} | Decayed: {num_decayed}"
        )
        self.ax.set_title(title, fontsize=14, fontweight="bold")

        stats = (
            f"λ = {self.decay_constant}\n"
            f"Half-life = {0.693 / self.decay_constant:.2f} s\n"
            f"Mean τ = {1 / self.decay_constant:.2f} s"
        )
        self.fig.text(
            0.02,
            0.95,
            stats,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            transform=self.fig.transFigure,
        )

        if frame % 50 == 0:
            progress = (frame + 1) / len(self.time_steps) * 100
            logger.info("3D frame %d / %d (%.0f%%)", frame,
                        len(self.time_steps), progress)

    def run(self) -> animation.FuncAnimation:
        """Build and return the FuncAnimation."""
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
        """Save as MP4.

        Args:
            filepath: Output path.  Defaults to ``results/animations/``.

        Returns:
            Path where the file was saved.
        """
        if self.anim is None:
            self.run()

        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(
                PathConfig.ANIMATION_OUTPUT_DIR / f"space_3d_{ts}.mp4"
            )

        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving 3D MP4 to %s …", out)
        writer = animation.FFMpegWriter(
            fps=self.fps,
            metadata={"artist": "Mussab Omair"},
            bitrate=AnimationConfig.SPACE_3D_BITRATE,
        )
        self.anim.save(str(out), writer=writer)
        logger.info("3D MP4 saved: %s", out)
        return out

    def save_gif(self, filepath: Optional[str] = None) -> Path:
        """Save as GIF.

        Args:
            filepath: Output path.

        Returns:
            Path where the file was saved.
        """
        if self.anim is None:
            self.run()

        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(
                PathConfig.ANIMATION_OUTPUT_DIR / f"space_3d_{ts}.gif"
            )

        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving 3D GIF to %s …", out)
        self.anim.save(str(out), writer="pillow", fps=self.fps)
        logger.info("3D GIF saved: %s", out)
        return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a 3-D rotating particle decay animation."
    )
    parser.add_argument("--n-particles", type=int,
                        default=500, help="Particles")
    parser.add_argument("--decay-constant", type=float,
                        default=0.1, help="λ (s⁻¹)")
    parser.add_argument("--max-time", type=float,
                        default=40.0, help="Duration (s)")
    parser.add_argument("--fps", type=int, default=15, help="FPS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--gif", action="store_true", help="Also save GIF")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    anim_obj = Space3DAnimation(
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
