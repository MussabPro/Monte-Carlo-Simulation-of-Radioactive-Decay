# Monte Carlo Simulation of Radioactive Decay

## Overview
This project implements a Monte Carlo simulation of radioactive decay processes, 
demonstrating statistical methods used in particle physics research.

## Physics Background
Radioactive decay follows an exponential distribution:
$$N(t) = N_0 e^{-\lambda t}$$

Where λ is the decay constant and the half-life is t½ = ln(2)/λ.

## Features
- Exponential decay simulation with configurable parameters
- Multiple decay channels with branching ratios
- Statistical analysis (mean lifetime, chi-squared test)
- Detector efficiency modeling
- Parameter scanning
- Professional visualizations

## Requirements
```bash
pip install numpy matplotlib scipy
```

## Usage
```bash
python decay_simulation.py
```

## Results
The simulation generates:
- Decay time histograms
- Statistical analysis reports
- Detector efficiency comparisons
- Parameter scan plots

## Technical Details
- **Method**: Monte Carlo sampling using NumPy's exponential distribution
- **Particles simulated**: 10,000 per run
- **Typical accuracy**: <2% error vs theoretical prediction

## Applications to Particle Physics
This simulation technique is used at facilities like KEK's Belle II experiment 
to model:
- B meson decay processes
- Detector response functions
- Background event estimation

## Author
Mussab Omair
University of Gujrat