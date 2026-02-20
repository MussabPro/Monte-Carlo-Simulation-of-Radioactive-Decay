"""
Real-time interactive particle decay simulation using Pygame.

Controls:
    SPACE — Pause / Resume
    R     — Reset simulation
    S     — Save screenshot
    ESC   — Quit

Usage:
    python -m animations.realtime_interactive
    python animations/realtime_interactive.py
"""
# isort: skip_file
from config import AnimationConfig, PathConfig, SimulationConfig
import numpy as np
from typing import Optional
from datetime import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger(__name__)


class InteractiveSimulation:
    """Pygame-based real-time particle decay visualisation.

    Particles are placed randomly on screen and transition through
    colour stages as they approach and pass their decay time.  The
    display shows live statistics including FPS, particle counts,
    mean lifetime, and theoretical comparison.

    Args:
        n_particles: Number of particles.
        decay_constant: Decay rate λ (s⁻¹).
        random_seed: Optional RNG seed.

    Example:
        >>> sim = InteractiveSimulation(n_particles=1000)
        >>> sim.run()   # opens a Pygame window
    """

    def __init__(
        self,
        n_particles: int = 1000,
        decay_constant: float = SimulationConfig.DEFAULT_DECAY_CONSTANT,
        random_seed: Optional[int] = SimulationConfig.RANDOM_SEED,
    ) -> None:
        self.n_particles = n_particles
        self.decay_constant = decay_constant
        self.random_seed = random_seed

        self.width = AnimationConfig.PYGAME_WIDTH
        self.height = AnimationConfig.PYGAME_HEIGHT
        self.target_fps = AnimationConfig.PYGAME_FPS
        self.radius = AnimationConfig.PYGAME_PARTICLE_RADIUS

        # Colours
        self.bg = AnimationConfig.PYGAME_BG_COLOR
        self.col_alive = AnimationConfig.PYGAME_COLOR_ALIVE
        self.col_dead = AnimationConfig.PYGAME_COLOR_DECAYED
        self.col_warn = AnimationConfig.PYGAME_COLOR_WARNING
        self.col_text = AnimationConfig.PYGAME_COLOR_TEXT

    def _reset(self) -> None:
        """Generate fresh decay times and positions."""
        rng = np.random.default_rng(self.random_seed)
        self.decay_times = rng.exponential(
            1.0 / self.decay_constant, size=self.n_particles
        )
        self.positions = (
            rng.random((self.n_particles, 2))
            * [self.width - 100, self.height - 200]
            + [50, 50]
        )
        self.current_time = 0.0
        self.paused = False

    def run(self) -> None:
        """Open the Pygame window and start the simulation loop."""
        try:
            import pygame
        except ImportError:
            logger.error(
                "Pygame is not installed.  Run: pip install pygame"
            )
            print("Error: pygame not installed. Install with: pip install pygame")
            return

        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Real-Time Particle Decay Simulation")
        clock = pygame.time.Clock()

        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)

        self._reset()
        running = True

        logger.info(
            "Interactive simulation started: N=%d, λ=%.3f",
            self.n_particles,
            self.decay_constant,
        )

        while running:
            dt = clock.tick(self.target_fps) / 1000.0
            actual_fps = clock.get_fps()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self._reset()
                    elif event.key == pygame.K_s:
                        self._save_screenshot(screen)
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            if not self.paused:
                self.current_time += dt

            screen.fill(self.bg)

            num_alive = 0
            num_decayed = 0

            for pos, t_decay in zip(self.positions, self.decay_times):
                ix, iy = int(pos[0]), int(pos[1])
                if self.current_time < t_decay:
                    num_alive += 1
                    time_left = t_decay - self.current_time
                    if time_left > 10:
                        color = self.col_alive
                    elif time_left > 5:
                        color = self.col_warn
                    else:
                        color = (255, int(255 * time_left / 5), 0)
                    pygame.draw.circle(screen, color, (ix, iy), self.radius)
                else:
                    num_decayed += 1
                    elapsed = self.current_time - t_decay
                    if elapsed < 3:
                        brightness = int(100 * (1 - elapsed / 3))
                        c = (min(255, 200 + int(55 * elapsed / 3)), 0, 0)
                        r = max(1, self.radius - int(elapsed))
                        pygame.draw.circle(screen, c, (ix, iy), r)

            # --- HUD ---
            hud_x = self.width - 260
            hud_y = 20

            lines = [
                (font, f"Time: {self.current_time:.1f} s", self.col_text),
                (font, f"Living: {num_alive}", self.col_alive),
                (font, f"Decayed: {num_decayed}", self.col_dead),
                (font, f"{num_decayed / self.n_particles * 100:.1f}% decayed", self.col_warn),
            ]

            for i, (f, txt, col) in enumerate(lines):
                surf = f.render(txt, True, col)
                screen.blit(surf, (hud_x, hud_y + i * 40))

            if num_decayed > 0:
                mask = self.decay_times <= self.current_time
                mean_life = float(np.mean(self.decay_times[mask]))
                theory = 1.0 / self.decay_constant
                extra = [
                    f"Mean τ: {mean_life:.2f} s",
                    f"Theory: {theory:.2f} s",
                    f"FPS: {actual_fps:.0f}",
                ]
                for j, txt in enumerate(extra):
                    surf = small_font.render(txt, True, self.col_text)
                    screen.blit(
                        surf, (hud_x, hud_y + len(lines) * 40 + j * 25))

            if self.paused:
                pause_surf = font.render("PAUSED", True, self.col_dead)
                screen.blit(pause_surf, (self.width // 2 - 70, 20))

            controls = small_font.render(
                "SPACE: Pause | R: Reset | S: Screenshot | ESC: Quit",
                True,
                self.col_text,
            )
            screen.blit(controls, (20, self.height - 30))

            pygame.display.flip()

        pygame.quit()
        logger.info("Interactive simulation ended")

    def _save_screenshot(self, screen) -> None:
        """Save current frame as PNG."""
        import pygame

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = PathConfig.ANIMATION_OUTPUT_DIR / f"screenshot_{ts}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(screen, str(out))
        logger.info("Screenshot saved: %s", out)
        print(f"Screenshot saved: {out}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    sim = InteractiveSimulation()
    sim.run()


if __name__ == "__main__":
    main()
