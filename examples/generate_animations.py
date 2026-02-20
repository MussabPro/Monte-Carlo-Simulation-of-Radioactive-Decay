#!/usr/bin/env python3
"""
Generate all animations (2-D grid and 3-D cloud).

Saves videos and optional GIFs to ``results/animations/``.
The interactive Pygame simulation is not included here — run it
separately with ``python animations/realtime_interactive.py``.

Usage:
    python examples/generate_animations.py
    python examples/generate_animations.py --gif --n-particles 500
"""
import argparse
import logging

from animations.grid_2d_animation import Grid2DAnimation
from animations.space_3d_animation import Space3DAnimation


def main() -> None:
    """Generate 2-D and 3-D decay animations."""
    parser = argparse.ArgumentParser(description="Generate decay animations.")
    parser.add_argument("--n-particles", type=int,
                        default=500, help="Particles")
    parser.add_argument("--decay-constant", type=float,
                        default=0.1, help="λ (s⁻¹)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gif", action="store_true", help="Also export GIFs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # --- 2-D Grid Animation ---
    print("Generating 2-D grid animation …")
    grid = Grid2DAnimation(
        n_particles=args.n_particles,
        decay_constant=args.decay_constant,
        max_time=50.0,
        fps=10,
        random_seed=args.seed,
    )
    grid.setup()
    grid.run()
    grid.save_video()
    if args.gif:
        grid.save_gif()
    print("  2-D grid animation saved.")

    # --- 3-D Space Animation ---
    print("Generating 3-D space animation …")
    space = Space3DAnimation(
        n_particles=args.n_particles,
        decay_constant=args.decay_constant,
        max_time=40.0,
        fps=15,
        random_seed=args.seed,
    )
    space.setup()
    space.run()
    space.save_video()
    if args.gif:
        space.save_gif()
    print("  3-D space animation saved.")

    print("\nAll animations generated.")


if __name__ == "__main__":
    main()
