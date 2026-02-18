# What is Monte Carlo Method?

## The Core Idea

Instead of calculating "exactly when each particle decays," we use random numbers to simulate probability.

For each particle, each second:

1. Generate random number between 0 and 1
2. If random number < (decay probability), particle decays
3. Record the decay time
4. Repeat for all particles

Named after the famous casino — it uses randomness like gambling!

## Why Physicists Use This

- Real particle physics is probabilistic (quantum mechanics)
- Complex systems (millions of particles, many interactions)
- Analytical solutions impossible → Simulate with random sampling
