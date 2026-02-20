# Physics of Radioactive Decay

## The Basic Concept

Imagine a bag of popcorn kernels in a microwave. You cannot predict *which* kernel
will pop next, but you *can* predict, from experience, approximately how many will
have popped after 30 seconds, or 60. Each kernel "decides" independently and
randomly—yet the aggregate follows a simple, smooth trend.

Radioactive decay works the same way. An unstable atomic nucleus has a constant
probability of decaying in any given instant. The individual moment of decay is
genuinely random (quantum mechanics forbids us from predicting it exactly), but
when millions of nuclei are involved, their collective behaviour is highly
predictable.

## The Mathematics (Simplified)

If we start with $N_0$ particles, the number still alive at time $t$ is

$$N(t) = N_0 \, e^{-\lambda\,t}$$

where $\lambda$ is the **decay constant**—a number that encodes how quickly a
particular isotope (or particle) falls apart. A large $\lambda$ means the
substance is short-lived; a small one means it lingers.

Two related quantities appear everywhere in nuclear and particle physics:

| Quantity | Formula | Meaning |
|----------|---------|---------|
| Mean lifetime | $\tau = 1/\lambda$ | Average time a particle survives |
| Half-life | $t_{1/2} = \ln 2 / \lambda$ | Time for half the sample to decay |

The time at which any *single* particle decays follows an **exponential
distribution** with rate $\lambda$. This is the distribution our simulation
samples.

## Why Monte Carlo?

The name "Monte Carlo" is a nod to the famous casino in Monaco. Just as casino
games rely on chance, Monte Carlo simulations use *controlled randomness* to solve
problems that are too complex—or too fundamentally random—for a neat algebraic
formula.

The algorithm is deceptively simple:

1. For each particle, draw a random number from an exponential distribution.
2. That number is the particle's decay time.
3. Repeat for every particle in your sample.
4. Histograms and statistics of the resulting times reproduce the theoretical
   curve—complete with realistic statistical fluctuations.

Because each "throw of the dice" is independent, Monte Carlo methods are easy to
parallelise and extend. Need to model a detector that only catches 85 % of events?
Throw away 15 % of your simulated decays at random. Need to see what happens with
three competing decay channels? Assign each event to a channel with the
appropriate probability.

## Real-World Applications

Monte Carlo simulation is indispensable in science and engineering:

- **Medical imaging**: PET and SPECT scanners use Monte Carlo codes to model how
  photons travel through human tissue.
- **Carbon dating**: The exponential-decay curve for Carbon-14 lets
  archaeologists date organic material up to ~50 000 years old.
- **Radiation safety**: Shielding designs for nuclear reactors and spacecraft are
  validated against Monte Carlo radiation transport calculations.
- **Financial modelling**: Option pricing often relies on Monte Carlo sampling of
  possible future stock paths.

## Connection to Particle Physics at KEK

At the KEK laboratory in Tsukuba, Japan, the Belle II experiment collects
billions of $e^+e^-$ collision events to study **B mesons**—particles containing
a *bottom* quark. B mesons are unstable; they travel roughly 200 µm before
decaying into lighter particles.

Understanding the *shape* of their decay-time distribution—and how well the
detector can measure it—is critical for testing the Standard Model's predictions
about CP violation (the slight asymmetry between matter and antimatter). The same
Monte Carlo techniques used in this project underpin the full detector simulation
at Belle II, scaled up to hundreds of millions of events and dozens of physics
channels.

## Further Reading

- **Particle Data Group**: [pdg.lbl.gov](https://pdg.lbl.gov) — the definitive
  reference for particle properties and decay data.
- **Leo, W. R.** *Techniques for Nuclear and Particle Physics Experiments*
  (Springer) — an excellent undergraduate-level lab textbook.
- **James, F.** *Monte Carlo Theory and Practice*, Rep. Prog. Phys. 43 (1980)
  1145 — a classic review of Monte Carlo in high-energy physics.
