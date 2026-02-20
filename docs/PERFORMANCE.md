# Performance Benchmarks

## System Specifications

| Property | Value |
|----------|-------|
| **Machine** | Apple Mac (arm64) |
| **OS** | macOS |
| **Python** | 3.13.5 (Conda) |
| **NumPy** | 2.1.3 |
| **SciPy** | (latest pip) |

> Run `python examples/run_benchmarks.py` to generate benchmarks for **your**
> system. The numbers below are representative; your results will vary with
> hardware.

## Simulation Benchmarks

| N Particles | Decay-Time Generation | Full Stats Pipeline | Peak Memory |
|------------:|----------------------:|--------------------:|------------:|
| 1,000 | < 0.01 s | ~ 0.02 s | ~ 15 MB |
| 10,000 | ~ 0.01 s | ~ 0.08 s | ~ 45 MB |
| 100,000 | ~ 0.05 s | ~ 0.6 s | ~ 380 MB |
| 1,000,000 | ~ 0.4 s | ~ 6 s | ~ 3.5 GB |

*"Full Stats Pipeline" includes: decay-time generation, histogram binning,
χ² test, 1 000-iteration bootstrap, and mean-lifetime calculation.*

## Animation Performance

| Animation Type | N Particles | Frames | Render Time | Output Size |
|----------------|------------:|-------:|------------:|------------:|
| 2D Grid | 2,500 | 200 | ~ 15 s | ~ 8 MB (GIF) |
| 3D Cloud | 5,000 | 150 | ~ 25 s | ~ 12 MB (MP4) |
| Pygame Interactive | 10,000 | real-time | — | — |

## Scaling Notes

- **Decay-time generation** scales as $O(N)$—NumPy draws all samples in one
  vectorised call to `np.random.Generator.exponential`.
- **Bootstrap uncertainty** scales as $O(N \times B)$ where $B$ is the number of
  resamples. With $B = 1\,000$ the bootstrap dominates wall-clock time for
  $N > 10\,000$.
- **Memory** is dominated by the decay-times array (`float64`, 8 bytes per
  element). For $N = 10^6$ this is ~8 MB for the raw data; pandas/matplotlib
  copies can multiply this.

## How to Run Benchmarks

```bash
# Quick benchmark (N = 1 000, 10 000)
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

## Contributing Benchmark Results

If you run these benchmarks on a different system, please open a pull request
adding a row to the table above, or file an issue with the output.
