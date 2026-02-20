# Contributing Guide

Thank you for your interest in this project! Whether you want to run the
simulations on your own machine, report a bug, or add a new feature, this
guide will get you started.

---

## Prerequisites

| Requirement | Minimum Version |
|-------------|----------------|
| Python | 3.10+ |
| pip | 21+ |
| Git | 2.x |

Optional (for animations):
- **Pygame** ≥ 2.0 — real-time interactive simulation
- **ffmpeg** — saving MP4 videos from Matplotlib animations

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/MussabPro/Monte-Carlo-Simulation-of-Radioactive-Decay.git
cd Monte-Carlo-Simulation-of-Radioactive-Decay

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Running Simulations

Each example script is self-contained and can be run directly:

```bash
# Basic exponential decay (quick, <10 s)
python examples/run_basic.py

# Multi-channel B-meson-style decay
python examples/run_multichannel.py

# Full analysis pipeline (histograms, parameter scan, detector effects)
python examples/run_full_analysis.py
```

Outputs (plots, data files) are saved into the `results/` directory.

## Running Tests

The project uses **pytest**. From the project root:

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_basic_simulation.py -v

# Run with coverage (requires pytest-cov)
python -m pytest tests/ --cov=src --cov-report=term-missing
```

All 53 tests should pass. If any test fails on your system, please open an
issue with the full `pytest` output and your `python --version`.

## Generating Benchmarks

```bash
# Quick benchmarks printed to stdout
python -c "
import time, numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path('.')))
from src.basic_simulation import generate_decay_times
from src.statistics import compare_to_theory
for n in [1_000, 10_000, 100_000]:
    t0 = time.perf_counter()
    times = generate_decay_times(n, 0.1, random_seed=42)
    dt_gen = time.perf_counter() - t0
    t0 = time.perf_counter()
    compare_to_theory(times, 0.1, n_bootstrap=500, random_seed=42)
    dt_full = time.perf_counter() - t0
    print(f'N={n:>10,}  gen={dt_gen:.4f}s  full={dt_full:.4f}s')
"
```

## Generating Animations

```bash
# 2D + 3D animations
python examples/generate_animations.py

# Interactive Pygame simulation
python -m animations.realtime_interactive
```

## Project Layout

```
monte-carlo-decay/
├── config.py          # All tuneable parameters
├── src/               # Core simulation & analysis modules
├── animations/        # 2D, 3D, and Pygame visualisations
├── examples/          # Ready-to-run scripts
├── tests/             # pytest test suite
├── docs/              # Physics background, performance, this file
└── results/           # Generated outputs (git-ignored)
```

## Adding New Features

1. **Branch**: Create a feature branch (`git checkout -b feature/my-feature`).
2. **Code**: Follow the existing style—type hints, Google-style docstrings,
   `logging` instead of `print`, constants in `config.py`.
3. **Test**: Add tests in `tests/` covering the new functionality.
4. **Document**: Update relevant docs and examples.
5. **PR**: Open a pull request with a clear description.

### Style Checklist

- [ ] All functions have type hints and docstrings
- [ ] No hard-coded magic numbers (use `config.py`)
- [ ] Functions are ≤ 50 lines
- [ ] Lines are ≤ 100 characters
- [ ] `pathlib.Path` for file paths (no `os.path.join`)
- [ ] `logging` module (no bare `print` for status messages)

## Reporting Issues

Please include:
- Python version (`python --version`)
- OS and architecture
- Full traceback / pytest output
- Steps to reproduce

## License

This project is released under the MIT License. Contributions are welcome
under the same terms.
