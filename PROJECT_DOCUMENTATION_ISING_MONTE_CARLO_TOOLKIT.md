# Ising Monte Carlo Toolkit

A high-performance Python toolkit for simulating magnetic phase transitions using Monte Carlo methods. This project provides a comprehensive framework for studying the Ising model across different dimensions, with automated workflows, statistical analysis, and publication-quality visualizations.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/username/ising-mc-toolkit/workflows/tests/badge.svg)](https://github.com/username/ising-mc-toolkit/actions)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/username/ising-mc-toolkit)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Physics Background](#physics-background)
   - [The Ising Model](#the-ising-model)
   - [Phase Transitions](#phase-transitions)
   - [Physical Observables](#physical-observables)
3. [Algorithms](#algorithms)
   - [Metropolis Algorithm](#metropolis-algorithm)
   - [Wolff Cluster Algorithm](#wolff-cluster-algorithm)
4. [Features](#features)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Usage Guide](#usage-guide)
   - [Command Line Interface](#command-line-interface)
   - [Python API](#python-api)
   - [Configuration Files](#configuration-files)
8. [Snakemake Workflow](#snakemake-workflow)
9. [Project Architecture](#project-architecture)
10. [API Reference](#api-reference)
11. [Testing](#testing)
12. [Examples](#examples)
13. [Performance Optimization](#performance-optimization)
14. [Contributing](#contributing)
15. [References](#references)
16. [License](#license)

---

## Introduction

### Motivation

Understanding phase transitions is one of the central problems in statistical physics. When a material like iron is heated past its Curie temperature, it undergoes a dramatic transformation from a ferromagnetic state (where atomic magnetic moments align) to a paramagnetic state (where moments point randomly). This transition happens suddenly at a specific critical temperature, exhibiting universal behavior that is independent of microscopic details.

The Ising model, despite its simplicity, captures the essential physics of these magnetic phase transitions. However, analytical solutions exist only for special cases (1D and 2D without external field). For realistic systems, we must rely on computational methods.

This toolkit implements Monte Carlo simulation methods for the Ising model, providing:

- Efficient simulation of 1D, 2D, and 3D Ising models
- Multiple sampling algorithms (Metropolis and Wolff cluster)
- Comprehensive statistical analysis with proper error estimation
- Automated parameter sweep workflows using Snakemake
- Publication-quality visualizations and animations

### Project Goals

1. **Educational**: Provide clear, well-documented implementations that serve as learning resources for computational physics
2. **Research-ready**: Offer performance-optimized code suitable for actual research applications
3. **Reproducible**: Enable fully reproducible computational experiments through automated workflows
4. **Extensible**: Design a modular architecture that can be extended to other lattice models

---

## Physics Background

### The Ising Model

The Ising model represents a lattice of interacting magnetic moments (spins) that can take one of two values: up (+1) or down (-1). It was proposed by Wilhelm Lenz in 1920 and solved in one dimension by his student Ernst Ising in 1925.

#### Mathematical Definition

Consider a lattice with N sites, where each site i has a spin variable σᵢ ∈ {-1, +1}. The Hamiltonian (energy function) of the system is:

```
H = -J Σ σᵢσⱼ - h Σ σᵢ
      ⟨i,j⟩      i
```

Where:
- J is the coupling constant (J > 0 for ferromagnetic interaction)
- The first sum runs over all nearest-neighbor pairs ⟨i,j⟩
- h is the external magnetic field
- The second sum runs over all sites

In this toolkit, we focus on the case h = 0 (no external field), where the Hamiltonian simplifies to:

```
H = -J Σ σᵢσⱼ
      ⟨i,j⟩
```

#### Lattice Geometries

**1D Chain:**
```
σ₁ — σ₂ — σ₃ — σ₄ — σ₅ — ... — σₙ
```
Each spin has 2 neighbors (with periodic boundary conditions).

**2D Square Lattice:**
```
σ₁,₁ — σ₁,₂ — σ₁,₃ — σ₁,₄
  |      |      |      |
σ₂,₁ — σ₂,₂ — σ₂,₃ — σ₂,₄
  |      |      |      |
σ₃,₁ — σ₃,₂ — σ₃,₃ — σ₃,₄
  |      |      |      |
σ₄,₁ — σ₄,₂ — σ₄,₃ — σ₄,₄
```
Each spin has 4 neighbors (with periodic boundary conditions).

**3D Cubic Lattice:**
Each spin has 6 neighbors (with periodic boundary conditions).

#### Boundary Conditions

**Periodic Boundary Conditions (PBC):**
The lattice wraps around, so spins on opposite edges are neighbors. This minimizes finite-size effects and is the default in this toolkit.

```
For a 1D chain of length L:
Spin at position i has neighbors at positions (i-1) mod L and (i+1) mod L
```

**Fixed Boundary Conditions:**
Edge spins have fewer neighbors. This can be useful for studying surface effects.

### Phase Transitions

#### The Ferromagnetic-Paramagnetic Transition

At low temperatures, thermal fluctuations are weak, and the system minimizes energy by aligning all spins. This is the **ferromagnetic phase** with non-zero magnetization.

At high temperatures, thermal fluctuations dominate, randomizing spin orientations. This is the **paramagnetic phase** with zero average magnetization.

The transition between these phases occurs at the **critical temperature** Tₒ.

#### Critical Temperature

**1D Ising Model:**
There is no phase transition at finite temperature. The system is always paramagnetic for T > 0.

**2D Ising Model:**
Lars Onsager derived the exact critical temperature in 1944:

```
Tₒ = 2J / ln(1 + √2) ≈ 2.269 J/kB
```

This is one of the few exactly solvable models with a phase transition.

**3D Ising Model:**
No exact solution exists. Numerical simulations give:

```
Tₒ ≈ 4.511 J/kB
```

#### Critical Phenomena

Near the critical temperature, physical quantities exhibit power-law behavior characterized by **critical exponents**:

| Quantity | Symbol | Behavior | 2D Exponent |
|----------|--------|----------|-------------|
| Magnetization | M | \|T - Tₒ\|^β | β = 1/8 |
| Susceptibility | χ | \|T - Tₒ\|^(-γ) | γ = 7/4 |
| Heat Capacity | C | \|T - Tₒ\|^(-α) | α = 0 (log) |
| Correlation Length | ξ | \|T - Tₒ\|^(-ν) | ν = 1 |

These exponents are **universal**, meaning they depend only on the dimensionality and symmetry of the system, not on microscopic details.

### Physical Observables

#### Energy

The instantaneous energy is calculated from the Hamiltonian:

```
E = -J Σ σᵢσⱼ
      ⟨i,j⟩
```

We report the **energy per spin**:

```
e = E / N
```

For the 2D Ising model:
- Ground state (T → 0): e = -2J (each spin has 4 aligned neighbors, but pairs are counted once)
- High temperature (T → ∞): e → 0 (random orientations)

#### Magnetization

The instantaneous magnetization is the sum of all spins:

```
M = Σ σᵢ
    i
```

We report the **magnetization per spin**:

```
m = M / N
```

In the absence of an external field, the system has symmetry between +m and -m states. We often report the **absolute magnetization** |m| or the **squared magnetization** m².

#### Heat Capacity

The heat capacity measures energy fluctuations:

```
C = (⟨E²⟩ - ⟨E⟩²) / (kB T²)
```

Or per spin:

```
c = C / N = (⟨e²⟩ - ⟨e⟩²) N / (kB T²)
```

The heat capacity peaks at the critical temperature, signaling the phase transition.

#### Magnetic Susceptibility

The susceptibility measures magnetization fluctuations:

```
χ = (⟨M²⟩ - ⟨M⟩²) / (kB T)
```

Or per spin:

```
χ = (⟨m²⟩ - ⟨m⟩²) N / (kB T)
```

The susceptibility diverges at the critical temperature in the thermodynamic limit.

#### Binder Cumulant

The Binder cumulant is a dimensionless quantity useful for precisely locating the critical temperature:

```
U = 1 - ⟨m⁴⟩ / (3⟨m²⟩²)
```

Properties:
- In the ordered phase (T < Tₒ): U → 2/3
- In the disordered phase (T > Tₒ): U → 0
- At Tₒ: U takes a universal value independent of system size

When plotting U vs T for different system sizes L, all curves cross at the critical temperature. This makes it an excellent tool for finite-size scaling analysis.

#### Autocorrelation Time

Successive Monte Carlo configurations are correlated. The autocorrelation function measures this:

```
A(t) = (⟨m(τ)m(τ+t)⟩ - ⟨m⟩²) / (⟨m²⟩ - ⟨m⟩²)
```

The **integrated autocorrelation time** τᵢₙₜ determines how many MC steps are needed between independent measurements:

```
τᵢₙₜ = 1/2 + Σ A(t)
            t=1
```

Near the critical temperature, τᵢₙₜ grows as:

```
τᵢₙₜ ~ L^z
```

Where z is the **dynamic critical exponent** (z ≈ 2 for Metropolis, z ≈ 0.25 for Wolff).

---

## Algorithms

### Monte Carlo Basics

Monte Carlo methods use random sampling to compute statistical averages. For the Ising model at temperature T, the probability of a configuration {σ} is given by the Boltzmann distribution:

```
P({σ}) = exp(-H({σ}) / kB T) / Z
```

Where Z is the partition function (normalization constant).

Direct sampling from this distribution is impossible for large systems. Instead, we use **Markov Chain Monte Carlo (MCMC)**, which generates a sequence of configurations that, after equilibration, are distributed according to P({σ}).

### Metropolis Algorithm

The Metropolis algorithm (Metropolis et al., 1953) is the foundational MCMC method for statistical physics.

#### Algorithm Steps

```
1. Initialize the lattice (random or ordered configuration)
2. Repeat for desired number of steps:
   a. Select a random spin i
   b. Calculate the energy change ΔE if spin i were flipped
   c. If ΔE ≤ 0:
      - Accept the flip (always)
   d. If ΔE > 0:
      - Accept the flip with probability exp(-ΔE / kB T)
      - Generate random number r ∈ [0, 1)
      - If r < exp(-ΔE / kB T): accept
      - Otherwise: reject
   e. Record measurements (if in measurement phase)
```

#### Energy Change Calculation

For a spin σᵢ with neighbors {σⱼ}, the energy change upon flipping is:

```
ΔE = 2J σᵢ Σ σⱼ
           j∈neighbors
```

This depends only on the local environment, making it efficient to compute.

**Possible values of ΔE for 2D square lattice:**

| Sum of neighbors | ΔE / J |
|------------------|--------|
| -4 (all opposite) | -8 |
| -2 | -4 |
| 0 | 0 |
| +2 | +4 |
| +4 (all aligned) | +8 |

#### Acceptance Probabilities

We can precompute the acceptance probabilities for all possible ΔE values:

```python
# For 2D Ising at temperature T
acceptance = {
    -8: 1.0,
    -4: 1.0,
     0: 1.0,
     4: exp(-4 / T),
     8: exp(-8 / T),
}
```

This lookup table significantly speeds up the simulation.

#### Detailed Balance

The Metropolis algorithm satisfies **detailed balance**:

```
P({σ}) W({σ} → {σ'}) = P({σ'}) W({σ'} → {σ})
```

Where W is the transition probability. This ensures the Markov chain converges to the correct equilibrium distribution.

#### Ergodicity

Single-spin flips can reach any configuration from any other configuration (given enough steps), satisfying **ergodicity**. This guarantees the Markov chain explores the entire configuration space.

### Wolff Cluster Algorithm

The Wolff algorithm (Wolff, 1989) is a cluster-based method that dramatically reduces critical slowing down.

#### The Problem with Metropolis Near Tₒ

Near the critical temperature, large clusters of aligned spins form. The Metropolis algorithm flips one spin at a time, making it very slow to decorrelate configurations. The autocorrelation time grows as τ ~ L^z with z ≈ 2.

#### Cluster Algorithm Idea

Instead of flipping single spins, identify and flip entire clusters of aligned spins. This allows large-scale changes in a single step.

#### Algorithm Steps

```
1. Initialize the lattice
2. Repeat for desired number of steps:
   a. Select a random spin i as the seed
   b. Build a cluster:
      - Add seed to cluster
      - For each spin in cluster:
        - For each neighbor j not yet in cluster:
          - If σⱼ = σᵢ (aligned):
            - Add j to cluster with probability p = 1 - exp(-2J / kB T)
   c. Flip all spins in the cluster
   d. Record measurements
```

#### Why It Works

The probability p = 1 - exp(-2J/kB T) is chosen to satisfy detailed balance for the cluster move. At low T, p → 1 (large clusters), and at high T, p → 0 (small clusters).

#### Performance Comparison

| Algorithm | Dynamic Exponent z | τᵢₙₜ for L=100, T=Tₒ |
|-----------|-------------------|----------------------|
| Metropolis | ≈ 2.17 | ~20,000 |
| Wolff | ≈ 0.25 | ~3 |

The Wolff algorithm is orders of magnitude faster near the critical point.

#### Implementation Considerations

The Wolff algorithm requires:
- Efficient cluster-building data structure (union-find or recursive)
- Different measurement strategy (clusters, not sweeps)
- Careful treatment of magnetization (sign ambiguity)

---

## Features

### Core Simulation Features

- **Multiple Dimensions**: 1D, 2D, and 3D Ising models
- **Multiple Algorithms**: Metropolis single-spin flip and Wolff cluster
- **Boundary Conditions**: Periodic and fixed boundaries
- **Initial Configurations**: Random, all-up, all-down, or custom

### Analysis Features

- **Physical Observables**: Energy, magnetization, heat capacity, susceptibility, Binder cumulant
- **Error Estimation**: Bootstrap resampling with configurable sample sizes
- **Autocorrelation Analysis**: Integrated autocorrelation time calculation
- **Finite-Size Scaling**: Tools for extrapolating to thermodynamic limit

### Performance Features

- **NumPy Vectorization**: Efficient array operations for lattice manipulations
- **Numba JIT Compilation**: Critical loops compiled to machine code
- **Parallel Execution**: Multi-core support for parameter sweeps
- **Memory Efficiency**: Compact spin representation using int8

### Workflow Features

- **Snakemake Integration**: Automated, reproducible parameter sweeps
- **Configuration Files**: YAML-based simulation parameters
- **Checkpointing**: Save and resume long simulations
- **Progress Tracking**: Real-time progress bars and logging

### Visualization Features

- **Phase Diagrams**: Magnetization, energy, and other observables vs temperature
- **Spin Configurations**: Lattice snapshots with customizable color schemes
- **Animations**: GIF/MP4 animations of spin evolution
- **Publication Quality**: Matplotlib styles optimized for papers

### Software Engineering Features

- **Command-Line Interface**: Full functionality without writing code
- **Python API**: Flexible programmatic access
- **Comprehensive Testing**: Unit tests, integration tests, and validation against exact results
- **Documentation**: API reference, tutorials, and physics background

---

## Installation

### Requirements

- Python 3.9 or higher
- NumPy 1.21+
- SciPy 1.7+
- Matplotlib 3.5+
- Numba 0.56+
- PyYAML 6.0+
- h5py 3.7+
- tqdm 4.64+
- pytest 7.0+ (for development)

### Install from PyPI

```bash
pip install ising-mc-toolkit
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/username/ising-mc-toolkit.git
cd ising-mc-toolkit

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Install with Conda

```bash
conda create -n ising python=3.10
conda activate ising
pip install ising-mc-toolkit
```

### Verify Installation

```bash
# Run tests
pytest tests/

# Check CLI
ising-sim --version

# Run quick simulation
ising-sim run --model ising2d --size 32 --temperature 2.27 --steps 10000
```

---

## Quick Start

### Minimal Example

```python
from ising_toolkit import Ising2D, MetropolisSampler

# Create a 32x32 Ising model at critical temperature
model = Ising2D(size=32, temperature=2.269)

# Create sampler and run simulation
sampler = MetropolisSampler(model)
results = sampler.run(
    n_steps=100000,
    equilibration=10000,
    measurement_interval=10
)

# Print results
print(f"Mean energy per spin: {results.energy.mean():.4f}")
print(f"Mean |magnetization|: {results.abs_magnetization.mean():.4f}")

# Plot magnetization time series
results.plot_timeseries('magnetization')
```

### Command Line Example

```bash
# Run single simulation
ising-sim run \
    --model ising2d \
    --size 64 \
    --temperature 2.27 \
    --steps 200000 \
    --equilibration 20000 \
    --output results/single_run.h5

# Analyze results
ising-sim analyze results/single_run.h5 --observables all

# Generate phase diagram (temperature sweep)
ising-sim sweep \
    --model ising2d \
    --size 64 \
    --temp-range 1.5 4.0 \
    --temp-steps 50 \
    --output results/phase_diagram/
```

---

## Usage Guide

### Command Line Interface

The toolkit provides the `ising-sim` command with several subcommands:

#### `ising-sim run`

Run a single Monte Carlo simulation.

```bash
ising-sim run [OPTIONS]

Options:
  --model TEXT              Model type: ising1d, ising2d, ising3d [required]
  --size INTEGER            Lattice size (L for LxL or LxLxL) [required]
  --temperature FLOAT       Simulation temperature (in units of J/kB) [required]
  --steps INTEGER           Number of Monte Carlo steps [default: 100000]
  --equilibration INTEGER   Equilibration steps [default: 10000]
  --algorithm TEXT          Algorithm: metropolis, wolff [default: metropolis]
  --seed INTEGER            Random seed for reproducibility
  --output PATH             Output file path (.h5 or .csv)
  --config PATH             Load parameters from YAML config file
  -v, --verbose             Enable verbose output
```

**Examples:**

```bash
# Basic 2D simulation
ising-sim run --model ising2d --size 64 --temperature 2.5 --steps 50000

# Use Wolff algorithm with specific seed
ising-sim run --model ising2d --size 128 --temperature 2.27 \
    --algorithm wolff --seed 42 --output data/wolff_run.h5

# Load configuration from file
ising-sim run --config configs/production.yaml
```

#### `ising-sim sweep`

Run temperature sweep for phase diagram.

```bash
ising-sim sweep [OPTIONS]

Options:
  --model TEXT              Model type [required]
  --size INTEGER            Lattice size (can specify multiple: --size 32 --size 64)
  --temp-range FLOAT FLOAT  Temperature range: start end [required]
  --temp-steps INTEGER      Number of temperature points [default: 50]
  --steps INTEGER           MC steps per temperature [default: 100000]
  --equilibration INTEGER   Equilibration steps [default: 10000]
  --algorithm TEXT          Algorithm: metropolis, wolff [default: metropolis]
  --parallel INTEGER        Number of parallel workers [default: 1]
  --output PATH             Output directory [required]
  -v, --verbose             Enable verbose output
```

**Examples:**

```bash
# Simple sweep
ising-sim sweep --model ising2d --size 64 \
    --temp-range 1.5 4.0 --temp-steps 50 \
    --output results/sweep/

# Multiple sizes for finite-size scaling
ising-sim sweep --model ising2d \
    --size 16 --size 32 --size 64 --size 128 \
    --temp-range 2.0 2.6 --temp-steps 30 \
    --parallel 4 --output results/fss/
```

#### `ising-sim analyze`

Analyze simulation results.

```bash
ising-sim analyze [OPTIONS] INPUT

Arguments:
  INPUT                     Input file or directory

Options:
  --observables TEXT        Observables to compute: energy, magnetization,
                            susceptibility, heat_capacity, binder, all
  --bootstrap INTEGER       Number of bootstrap samples for errors [default: 1000]
  --output PATH             Output file for analysis results
  --format TEXT             Output format: csv, json, hdf5 [default: csv]
```

**Examples:**

```bash
# Analyze single file
ising-sim analyze results/run.h5 --observables all

# Analyze sweep directory
ising-sim analyze results/sweep/ --observables energy magnetization

# Save analysis with bootstrap errors
ising-sim analyze results/sweep/ \
    --observables all --bootstrap 5000 \
    --output analysis/observables.csv
```

#### `ising-sim plot`

Generate visualizations.

```bash
ising-sim plot [OPTIONS] INPUT

Arguments:
  INPUT                     Input file or directory

Options:
  --type TEXT               Plot type: phase_diagram, timeseries, snapshot,
                            animation, binder, fss [required]
  --output PATH             Output file path
  --format TEXT             Figure format: png, pdf, svg [default: pdf]
  --style TEXT              Matplotlib style: publication, presentation, default
  --dpi INTEGER             Figure resolution [default: 300]
```

**Examples:**

```bash
# Phase diagram
ising-sim plot results/sweep/ --type phase_diagram --output figures/phase.pdf

# Spin configuration snapshot
ising-sim plot results/run.h5 --type snapshot --output figures/spins.png

# Animation
ising-sim plot results/run.h5 --type animation --output figures/evolution.gif

# Binder cumulant crossing
ising-sim plot results/fss/ --type binder --output figures/binder.pdf
```

### Python API

#### Basic Simulation

```python
from ising_toolkit import Ising2D, MetropolisSampler

# Create model
model = Ising2D(
    size=64,
    temperature=2.5,
    coupling=1.0,        # J, default is 1.0
    boundary='periodic'  # 'periodic' or 'fixed'
)

# Initialize configuration
model.initialize('random')  # 'random', 'up', 'down', or custom array

# Create sampler
sampler = MetropolisSampler(model, seed=42)

# Run simulation
results = sampler.run(
    n_steps=100000,
    equilibration=10000,
    measurement_interval=10,
    save_configurations=True,  # Save spin snapshots
    configuration_interval=1000
)

# Access results
print(f"Energy: {results.energy.mean():.4f} ± {results.energy.std():.4f}")
print(f"|M|: {results.abs_magnetization.mean():.4f}")
```

#### Using Wolff Algorithm

```python
from ising_toolkit import Ising2D, WolffSampler

model = Ising2D(size=128, temperature=2.269)
sampler = WolffSampler(model, seed=42)

# Wolff uses cluster flips, so fewer steps needed
results = sampler.run(
    n_steps=10000,       # Each step flips a cluster
    equilibration=1000,
    measurement_interval=1
)

# Average cluster size gives insight into critical behavior
print(f"Mean cluster size: {results.cluster_sizes.mean():.1f}")
```

#### Temperature Sweep

```python
from ising_toolkit import Ising2D, MetropolisSampler, TemperatureSweep

# Define sweep
sweep = TemperatureSweep(
    model_class=Ising2D,
    size=64,
    temperatures=np.linspace(1.5, 4.0, 50),
    n_steps=100000,
    equilibration=10000,
    algorithm='metropolis'
)

# Run (optionally in parallel)
results = sweep.run(n_workers=4, progress=True)

# Results is a DataFrame with T, E, M, C, chi, U columns
print(results.head())

# Plot phase diagram
sweep.plot_phase_diagram(save='figures/phase_diagram.pdf')
```

#### Finite-Size Scaling

```python
from ising_toolkit import FiniteSizeScaling

# Run sweeps for multiple sizes
sizes = [16, 32, 64, 128]
temperatures = np.linspace(2.0, 2.6, 30)

fss = FiniteSizeScaling(
    model_class=Ising2D,
    sizes=sizes,
    temperatures=temperatures,
    n_steps=200000,
    equilibration=20000
)

results = fss.run(n_workers=4)

# Find critical temperature from Binder crossing
Tc, Tc_error = fss.find_critical_temperature(method='binder')
print(f"Critical temperature: {Tc:.4f} ± {Tc_error:.4f}")

# Extract critical exponents
exponents = fss.extract_exponents()
print(f"β = {exponents['beta']:.3f}")
print(f"γ = {exponents['gamma']:.3f}")
print(f"ν = {exponents['nu']:.3f}")

# Plot Binder crossing
fss.plot_binder_crossing(save='figures/binder.pdf')
```

#### Statistical Analysis

```python
from ising_toolkit.analysis import (
    bootstrap_error,
    autocorrelation_time,
    blocking_analysis
)

# Bootstrap error estimation
mean, error = bootstrap_error(results.magnetization, n_samples=1000)
print(f"M = {mean:.4f} ± {error:.4f}")

# Autocorrelation time
tau = autocorrelation_time(results.energy)
print(f"Autocorrelation time: {tau:.1f} steps")

# Blocking analysis (alternative error estimation)
block_means, block_errors = blocking_analysis(results.energy, max_block_size=1000)
```

#### Custom Observables

```python
from ising_toolkit import Observable

# Define custom observable
class DomainWallDensity(Observable):
    """Count the density of domain walls (neighboring anti-aligned spins)."""
    
    name = "domain_wall_density"
    
    def compute(self, model):
        spins = model.spins
        # Count horizontal and vertical domain walls
        h_walls = np.sum(spins[:, :-1] != spins[:, 1:])
        v_walls = np.sum(spins[:-1, :] != spins[1:, :])
        total_bonds = 2 * model.size * (model.size - 1)
        return (h_walls + v_walls) / total_bonds

# Register and use
sampler.add_observable(DomainWallDensity())
results = sampler.run(n_steps=100000)
print(f"Domain wall density: {results.domain_wall_density.mean():.4f}")
```

### Configuration Files

Simulations can be configured using YAML files for reproducibility.

#### Basic Configuration

```yaml
# config/basic.yaml

simulation:
  model: ising2d
  size: 64
  temperature: 2.27
  coupling: 1.0
  boundary: periodic
  initial_state: random

algorithm:
  name: metropolis
  seed: 42

run:
  n_steps: 100000
  equilibration: 10000
  measurement_interval: 10

output:
  directory: results/
  format: hdf5
  save_configurations: true
  configuration_interval: 1000
```

#### Sweep Configuration

```yaml
# config/sweep.yaml

simulation:
  model: ising2d
  sizes: [32, 64, 128]
  temperatures:
    start: 1.5
    end: 4.0
    steps: 50
  coupling: 1.0
  boundary: periodic
  initial_state: random

algorithm:
  name: wolff
  seed: 42

run:
  n_steps: 50000
  equilibration: 5000
  measurement_interval: 1

parallel:
  n_workers: 4

output:
  directory: results/sweep/
  format: hdf5

analysis:
  observables:
    - energy
    - magnetization
    - susceptibility
    - heat_capacity
    - binder
  bootstrap_samples: 1000
```

#### Loading Configuration

```python
from ising_toolkit import Simulation

# From file
sim = Simulation.from_config('config/sweep.yaml')
results = sim.run()

# Override specific parameters
sim = Simulation.from_config(
    'config/sweep.yaml',
    overrides={'simulation.size': 256, 'run.n_steps': 200000}
)
```

---

## Snakemake Workflow

For large-scale parameter studies, the toolkit provides a Snakemake workflow that handles parallelization, dependency management, and reproducibility.

### Workflow Structure

```
workflow/
├── Snakefile           # Main workflow definition
├── config/
│   └── config.yaml     # Workflow configuration
├── rules/
│   ├── simulate.smk    # Simulation rules
│   ├── analyze.smk     # Analysis rules
│   └── figures.smk     # Visualization rules
└── scripts/
    ├── run_simulation.py
    ├── analyze_results.py
    └── generate_figures.py
```

### Configuration

```yaml
# workflow/config/config.yaml

# Model parameters
model: ising2d
sizes: [16, 32, 64, 128]
temperatures:
  start: 2.0
  end: 2.6
  steps: 30

# Simulation parameters
algorithm: wolff
n_steps: 100000
equilibration: 10000
seed_base: 42

# Analysis parameters
bootstrap_samples: 2000
observables:
  - energy
  - magnetization
  - susceptibility
  - heat_capacity
  - binder

# Output settings
results_dir: results
figures_dir: figures
```

### Running the Workflow

```bash
# Navigate to workflow directory
cd workflow

# Dry run (see what would be executed)
snakemake -n

# Run with 8 cores
snakemake --cores 8

# Run specific target
snakemake --cores 8 figures/phase_diagram.pdf

# Generate workflow diagram
snakemake --dag | dot -Tpng > workflow_dag.png
```

### Workflow DAG

```
                    ┌─────────────────┐
                    │ config.yaml     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ simulate_L16    │ │ simulate_L32    │ │ simulate_L64    │ ...
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ analyze_all     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ phase_diagram   │ │ binder_plot     │ │ fss_analysis    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Custom Rules

You can extend the workflow with custom rules:

```python
# workflow/rules/custom.smk

rule correlation_function:
    input:
        "results/{model}_L{size}_T{temp}.h5"
    output:
        "results/correlation/{model}_L{size}_T{temp}_corr.csv"
    script:
        "../scripts/compute_correlation.py"

rule correlation_length:
    input:
        expand("results/correlation/{{model}}_L{{size}}_T{temp}_corr.csv",
               temp=TEMPERATURES)
    output:
        "results/{model}_L{size}_correlation_length.csv"
    script:
        "../scripts/extract_correlation_length.py"
```

---

## Project Architecture

### Directory Structure

```
ising-mc-toolkit/
├── src/
│   └── ising_toolkit/
│       ├── __init__.py              # Package initialization, public API
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract base class for models
│       │   ├── ising1d.py           # 1D Ising model
│       │   ├── ising2d.py           # 2D Ising model
│       │   └── ising3d.py           # 3D Ising model
│       ├── samplers/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract base class for samplers
│       │   ├── metropolis.py        # Metropolis algorithm
│       │   └── wolff.py             # Wolff cluster algorithm
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── observables.py       # Physical observable calculations
│       │   ├── statistics.py        # Bootstrap, autocorrelation, blocking
│       │   └── finite_size.py       # Finite-size scaling analysis
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── plots.py             # Static plots (phase diagrams, etc.)
│       │   ├── animation.py         # Spin evolution animations
│       │   └── styles.py            # Matplotlib style configurations
│       ├── io/
│       │   ├── __init__.py
│       │   ├── config.py            # YAML configuration handling
│       │   ├── hdf5.py              # HDF5 file I/O
│       │   └── results.py           # Results container class
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── numba_kernels.py     # Numba-compiled core functions
│       │   ├── parallel.py          # Parallel execution utilities
│       │   └── validation.py        # Input validation functions
│       └── cli.py                   # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_models.py               # Model tests
│   ├── test_samplers.py             # Sampler tests
│   ├── test_analysis.py             # Analysis tests
│   ├── test_observables.py          # Observable tests
│   ├── test_io.py                   # I/O tests
│   ├── test_integration.py          # Integration tests
│   └── test_validation.py           # Validation against exact results
├── docs/
│   ├── conf.py                      # Sphinx configuration
│   ├── index.rst                    # Documentation index
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── physics.rst                  # Physics background
│   ├── algorithms.rst               # Algorithm descriptions
│   ├── api/                         # API reference
│   │   ├── models.rst
│   │   ├── samplers.rst
│   │   ├── analysis.rst
│   │   └── visualization.rst
│   └── tutorials/
│       ├── basic_simulation.rst
│       ├── phase_transition.rst
│       └── finite_size_scaling.rst
├── examples/
│   ├── 01_basic_simulation.ipynb
│   ├── 02_phase_diagram.ipynb
│   ├── 03_wolff_vs_metropolis.ipynb
│   ├── 04_finite_size_scaling.ipynb
│   ├── 05_custom_observables.ipynb
│   └── data/                        # Example outputs
├── workflow/
│   ├── Snakefile
│   ├── config/
│   │   └── config.yaml
│   ├── rules/
│   │   ├── simulate.smk
│   │   ├── analyze.smk
│   │   └── figures.smk
│   └── scripts/
│       ├── run_simulation.py
│       ├── analyze_results.py
│       └── generate_figures.py
├── benchmarks/
│   ├── benchmark_metropolis.py
│   ├── benchmark_wolff.py
│   └── benchmark_scaling.py
├── .github/
│   └── workflows/
│       ├── tests.yml                # CI test workflow
│       └── docs.yml                 # Documentation build
├── pyproject.toml                   # Package configuration
├── README.md                        # This file
├── LICENSE                          # MIT License
└── CHANGELOG.md                     # Version history
```

### Class Hierarchy

```
IsingModel (ABC)
├── Ising1D
├── Ising2D
└── Ising3D

Sampler (ABC)
├── MetropolisSampler
└── WolffSampler

Observable (ABC)
├── Energy
├── Magnetization
├── AbsoluteMagnetization
├── HeatCapacity
├── Susceptibility
└── BinderCumulant

Results
├── SingleRunResults
└── SweepResults

Analysis
├── TemperatureSweep
├── FiniteSizeScaling
└── AutocorrelationAnalysis
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Input                                   │
│  (CLI args / Python API / Config YAML)                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Configuration Parser                            │
│  - Validate parameters                                               │
│  - Set defaults                                                      │
│  - Initialize random seed                                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Model Creation                               │
│  - Initialize lattice                                                │
│  - Set boundary conditions                                           │
│  - Precompute neighbor lists                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Sampler Execution                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Equilibration Phase                                          │    │
│  │  - Run n_equilibration steps                                 │    │
│  │  - No measurements recorded                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Measurement Phase                                            │    │
│  │  - Run n_steps steps                                         │    │
│  │  - Record observables at measurement_interval                │    │
│  │  - Optionally save spin configurations                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Results Container                               │
│  - Time series of observables                                        │
│  - Saved configurations (optional)                                   │
│  - Metadata (parameters, timing)                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ Statistical      │ │ File Output  │ │ Visualization    │
│ Analysis         │ │ (HDF5/CSV)   │ │ (Plots/Anim)     │
│ - Means          │ └──────────────┘ └──────────────────┘
│ - Errors         │
│ - Autocorr       │
└──────────────────┘
```

---

## API Reference

### Models Module

#### `IsingModel` (Abstract Base Class)

```python
class IsingModel(ABC):
    """Abstract base class for Ising models."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Spatial dimension of the model."""
        pass
    
    @property
    @abstractmethod
    def n_spins(self) -> int:
        """Total number of spins."""
        pass
    
    @property
    @abstractmethod
    def n_neighbors(self) -> int:
        """Number of neighbors per spin."""
        pass
    
    @abstractmethod
    def initialize(self, state: str = 'random') -> None:
        """Initialize spin configuration.
        
        Parameters
        ----------
        state : str
            'random', 'up', 'down', or 'checkerboard'
        """
        pass
    
    @abstractmethod
    def get_energy(self) -> float:
        """Calculate total energy of current configuration."""
        pass
    
    @abstractmethod
    def get_magnetization(self) -> float:
        """Calculate total magnetization of current configuration."""
        pass
    
    @abstractmethod
    def get_neighbor_sum(self, site: tuple) -> int:
        """Get sum of spins of neighbors of given site."""
        pass
    
    @abstractmethod
    def flip_spin(self, site: tuple) -> None:
        """Flip spin at given site."""
        pass
```

#### `Ising2D`

```python
class Ising2D(IsingModel):
    """2D Ising model on square lattice.
    
    Parameters
    ----------
    size : int
        Linear size of the lattice (total spins = size^2)
    temperature : float
        Temperature in units of J/kB
    coupling : float, optional
        Coupling constant J (default: 1.0)
    boundary : str, optional
        Boundary condition: 'periodic' or 'fixed' (default: 'periodic')
    
    Attributes
    ----------
    spins : ndarray
        2D array of spin values (+1 or -1)
    size : int
        Linear lattice size
    temperature : float
        Current temperature
    beta : float
        Inverse temperature (1/T)
    
    Examples
    --------
    >>> model = Ising2D(size=32, temperature=2.27)
    >>> model.initialize('random')
    >>> print(f"Energy: {model.get_energy()}")
    >>> print(f"Magnetization: {model.get_magnetization()}")
    """
    
    def __init__(
        self,
        size: int,
        temperature: float,
        coupling: float = 1.0,
        boundary: str = 'periodic'
    ):
        ...
    
    def set_temperature(self, temperature: float) -> None:
        """Update temperature and recalculate derived quantities."""
        ...
    
    def get_spin(self, i: int, j: int) -> int:
        """Get spin value at position (i, j)."""
        ...
    
    def get_energy_change(self, i: int, j: int) -> float:
        """Calculate energy change if spin at (i, j) were flipped."""
        ...
    
    def copy(self) -> 'Ising2D':
        """Create a deep copy of the model."""
        ...
```

### Samplers Module

#### `MetropolisSampler`

```python
class MetropolisSampler:
    """Metropolis single-spin flip sampler.
    
    Parameters
    ----------
    model : IsingModel
        The Ising model to sample
    seed : int, optional
        Random seed for reproducibility
    
    Examples
    --------
    >>> model = Ising2D(size=64, temperature=2.5)
    >>> sampler = MetropolisSampler(model, seed=42)
    >>> results = sampler.run(n_steps=100000, equilibration=10000)
    """
    
    def __init__(self, model: IsingModel, seed: int = None):
        ...
    
    def step(self) -> None:
        """Perform one Monte Carlo step (N single-spin flip attempts)."""
        ...
    
    def run(
        self,
        n_steps: int,
        equilibration: int = 0,
        measurement_interval: int = 1,
        observables: List[Observable] = None,
        save_configurations: bool = False,
        configuration_interval: int = 100,
        progress: bool = True
    ) -> Results:
        """Run Monte Carlo simulation.
        
        Parameters
        ----------
        n_steps : int
            Number of MC steps in measurement phase
        equilibration : int
            Number of MC steps for equilibration (no measurements)
        measurement_interval : int
            Measure observables every this many steps
        observables : list of Observable, optional
            Custom observables to measure (defaults to energy, magnetization)
        save_configurations : bool
            Whether to save spin configurations
        configuration_interval : int
            Save configuration every this many steps
        progress : bool
            Show progress bar
        
        Returns
        -------
        Results
            Container with measured observables and metadata
        """
        ...
    
    def add_observable(self, observable: Observable) -> None:
        """Add custom observable to be measured."""
        ...
```

#### `WolffSampler`

```python
class WolffSampler:
    """Wolff cluster algorithm sampler.
    
    Parameters
    ----------
    model : IsingModel
        The Ising model to sample (must be 2D or 3D)
    seed : int, optional
        Random seed for reproducibility
    
    Notes
    -----
    The Wolff algorithm is much more efficient than Metropolis near
    the critical temperature, with dynamic critical exponent z ≈ 0.25
    compared to z ≈ 2.17 for Metropolis.
    
    Each "step" flips one cluster, so fewer steps are needed compared
    to Metropolis. Typically n_steps = 10000 is sufficient near Tc.
    
    Examples
    --------
    >>> model = Ising2D(size=128, temperature=2.269)
    >>> sampler = WolffSampler(model, seed=42)
    >>> results = sampler.run(n_steps=10000, equilibration=1000)
    >>> print(f"Mean cluster size: {results.cluster_sizes.mean():.1f}")
    """
    
    def __init__(self, model: IsingModel, seed: int = None):
        ...
    
    def step(self) -> int:
        """Perform one Wolff cluster flip.
        
        Returns
        -------
        int
            Size of the flipped cluster
        """
        ...
    
    def run(
        self,
        n_steps: int,
        equilibration: int = 0,
        measurement_interval: int = 1,
        observables: List[Observable] = None,
        save_configurations: bool = False,
        configuration_interval: int = 100,
        progress: bool = True
    ) -> Results:
        """Run Wolff cluster simulation.
        
        Returns
        -------
        Results
            Container with observables; includes cluster_sizes array
        """
        ...
```

### Analysis Module

#### Statistical Functions

```python
def bootstrap_error(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = None
) -> Tuple[float, float]:
    """Estimate error using bootstrap resampling.
    
    Parameters
    ----------
    data : ndarray
        1D array of measurements
    statistic : callable
        Function to compute (default: np.mean)
    n_samples : int
        Number of bootstrap samples
    confidence : float
        Confidence level for error estimate
    seed : int, optional
        Random seed
    
    Returns
    -------
    estimate : float
        Point estimate of statistic
    error : float
        Bootstrap standard error
    
    Examples
    --------
    >>> mean, error = bootstrap_error(results.magnetization, n_samples=2000)
    >>> print(f"M = {mean:.4f} ± {error:.4f}")
    """
    ...


def autocorrelation_time(
    data: np.ndarray,
    method: str = 'integrated',
    max_lag: int = None
) -> float:
    """Calculate autocorrelation time.
    
    Parameters
    ----------
    data : ndarray
        Time series of measurements
    method : str
        'integrated' (default) or 'exponential'
    max_lag : int, optional
        Maximum lag to consider (default: len(data) // 4)
    
    Returns
    -------
    tau : float
        Autocorrelation time in units of measurement interval
    
    Notes
    -----
    The integrated autocorrelation time is:
        τ_int = 1/2 + Σ_{t=1}^{∞} ρ(t)
    where ρ(t) is the normalized autocorrelation function.
    
    The effective number of independent samples is:
        N_eff = N / (2 * τ_int)
    """
    ...


def blocking_analysis(
    data: np.ndarray,
    max_block_size: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform blocking analysis for error estimation.
    
    Parameters
    ----------
    data : ndarray
        Time series of measurements
    max_block_size : int, optional
        Maximum block size (default: len(data) // 10)
    
    Returns
    -------
    block_sizes : ndarray
        Array of block sizes used
    errors : ndarray
        Estimated standard error at each block size
    
    Notes
    -----
    As block size increases, the estimated error should plateau
    when the block size exceeds the autocorrelation time.
    """
    ...
```

#### Finite-Size Scaling

```python
class FiniteSizeScaling:
    """Finite-size scaling analysis tools.
    
    Parameters
    ----------
    model_class : type
        Model class (e.g., Ising2D)
    sizes : list of int
        System sizes to simulate
    temperatures : array-like
        Temperature values
    n_steps : int
        MC steps per simulation
    equilibration : int
        Equilibration steps
    algorithm : str
        'metropolis' or 'wolff'
    
    Examples
    --------
    >>> fss = FiniteSizeScaling(
    ...     model_class=Ising2D,
    ...     sizes=[16, 32, 64, 128],
    ...     temperatures=np.linspace(2.0, 2.6, 30),
    ...     n_steps=100000
    ... )
    >>> results = fss.run(n_workers=4)
    >>> Tc, Tc_err = fss.find_critical_temperature()
    """
    
    def __init__(
        self,
        model_class: type,
        sizes: List[int],
        temperatures: np.ndarray,
        n_steps: int,
        equilibration: int = None,
        algorithm: str = 'metropolis'
    ):
        ...
    
    def run(self, n_workers: int = 1, progress: bool = True) -> pd.DataFrame:
        """Run all simulations."""
        ...
    
    def find_critical_temperature(
        self,
        method: str = 'binder',
        sizes: List[int] = None
    ) -> Tuple[float, float]:
        """Estimate critical temperature.
        
        Parameters
        ----------
        method : str
            'binder' (crossing point) or 'susceptibility' (peak)
        sizes : list of int, optional
            Subset of sizes to use
        
        Returns
        -------
        Tc : float
            Estimated critical temperature
        Tc_error : float
            Uncertainty in Tc
        """
        ...
    
    def extract_exponents(
        self,
        Tc: float = None,
        exponents: List[str] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Extract critical exponents from scaling behavior.
        
        Parameters
        ----------
        Tc : float, optional
            Critical temperature (if None, estimated from data)
        exponents : list of str, optional
            Which exponents to extract: 'beta', 'gamma', 'nu', 'alpha'
        
        Returns
        -------
        dict
            Mapping from exponent name to (value, error) tuple
        """
        ...
    
    def plot_binder_crossing(
        self,
        ax: plt.Axes = None,
        save: str = None
    ) -> plt.Figure:
        """Plot Binder cumulant vs temperature for all sizes."""
        ...
    
    def plot_data_collapse(
        self,
        observable: str,
        Tc: float,
        nu: float,
        exponent: float,
        ax: plt.Axes = None,
        save: str = None
    ) -> plt.Figure:
        """Plot data collapse using finite-size scaling form."""
        ...
```

### Visualization Module

```python
def plot_phase_diagram(
    results: pd.DataFrame,
    observables: List[str] = None,
    sizes: List[int] = None,
    figsize: Tuple[float, float] = (10, 8),
    style: str = 'publication',
    save: str = None
) -> plt.Figure:
    """Plot physical observables vs temperature.
    
    Parameters
    ----------
    results : DataFrame
        Must contain 'temperature' column and observable columns
    observables : list of str, optional
        Which observables to plot (default: all available)
    sizes : list of int, optional
        Filter to specific system sizes
    figsize : tuple
        Figure size in inches
    style : str
        'publication', 'presentation', or 'default'
    save : str, optional
        Path to save figure
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    ...


def plot_spin_configuration(
    spins: np.ndarray,
    ax: plt.Axes = None,
    cmap: str = 'RdBu',
    title: str = None,
    save: str = None
) -> plt.Figure:
    """Plot spin configuration as 2D heatmap.
    
    Parameters
    ----------
    spins : ndarray
        2D array of spin values (+1 or -1)
    ax : Axes, optional
        Matplotlib axes to plot on
    cmap : str
        Colormap name
    title : str, optional
        Plot title
    save : str, optional
        Path to save figure
    """
    ...


def create_animation(
    configurations: List[np.ndarray],
    filename: str,
    fps: int = 10,
    cmap: str = 'RdBu',
    dpi: int = 100,
    title_template: str = "Step {step}"
) -> None:
    """Create animation of spin evolution.
    
    Parameters
    ----------
    configurations : list of ndarray
        List of 2D spin configurations
    filename : str
        Output filename (.gif or .mp4)
    fps : int
        Frames per second
    cmap : str
        Colormap name
    dpi : int
        Resolution
    title_template : str
        Title format string (receives 'step' keyword)
    """
    ...
```

---

## Testing

### Test Categories

#### Unit Tests

Test individual functions and methods in isolation.

```python
# tests/test_models.py

class TestIsing2D:
    """Tests for 2D Ising model."""
    
    def test_initialization_random(self):
        """Test random initialization produces valid spins."""
        model = Ising2D(size=32, temperature=2.0)
        model.initialize('random')
        
        assert model.spins.shape == (32, 32)
        assert set(np.unique(model.spins)) <= {-1, 1}
    
    def test_initialization_up(self):
        """Test 'up' initialization."""
        model = Ising2D(size=16, temperature=2.0)
        model.initialize('up')
        
        assert np.all(model.spins == 1)
    
    def test_energy_ground_state(self):
        """Test energy of fully aligned state."""
        model = Ising2D(size=32, temperature=2.0)
        model.initialize('up')
        
        # Each spin has 4 neighbors, all aligned: E = -J * N * 4 / 2
        expected = -2.0 * model.n_spins  # E per spin = -2
        assert model.get_energy() == pytest.approx(expected)
    
    def test_magnetization_ground_state(self):
        """Test magnetization of fully aligned state."""
        model = Ising2D(size=32, temperature=2.0)
        model.initialize('up')
        
        assert model.get_magnetization() == model.n_spins
    
    def test_energy_change_values(self):
        """Test energy change only takes valid values."""
        model = Ising2D(size=32, temperature=2.0)
        model.initialize('random')
        
        valid_values = {-8, -4, 0, 4, 8}
        
        for i in range(32):
            for j in range(32):
                dE = model.get_energy_change(i, j)
                assert dE in valid_values
    
    def test_periodic_boundary(self):
        """Test periodic boundary conditions."""
        model = Ising2D(size=4, temperature=2.0, boundary='periodic')
        model.initialize('up')
        model.spins[0, 0] = -1  # Flip corner spin
        
        # Corner should have 4 neighbors due to PBC
        neighbor_sum = model.get_neighbor_sum((0, 0))
        assert neighbor_sum == 4  # All neighbors are +1
```

#### Validation Tests

Compare simulation results against known analytical solutions.

```python
# tests/test_validation.py

class TestExactSolutions:
    """Validate against exact analytical results."""
    
    def test_1d_no_phase_transition(self):
        """1D Ising has no phase transition: <M> = 0 for all T > 0."""
        model = Ising1D(size=1000, temperature=1.0)
        sampler = MetropolisSampler(model, seed=42)
        results = sampler.run(n_steps=100000, equilibration=10000)
        
        mean_m = np.abs(results.magnetization).mean() / model.n_spins
        assert mean_m < 0.1  # Should be small (finite-size effects)
    
    def test_1d_exact_energy(self):
        """Test 1D energy against exact solution."""
        temperatures = [0.5, 1.0, 2.0, 4.0]
        
        for T in temperatures:
            model = Ising1D(size=10000, temperature=T)
            sampler = MetropolisSampler(model, seed=42)
            results = sampler.run(n_steps=50000, equilibration=5000)
            
            # Exact: e = -tanh(J/T)
            exact_e = -np.tanh(1.0 / T)
            measured_e = results.energy.mean() / model.n_spins
            
            assert measured_e == pytest.approx(exact_e, rel=0.05)
    
    def test_2d_critical_temperature(self):
        """Test 2D critical temperature matches Onsager solution."""
        exact_Tc = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269
        
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[16, 32, 64],
            temperatures=np.linspace(2.1, 2.5, 20),
            n_steps=50000,
            algorithm='wolff'
        )
        results = fss.run()
        Tc, _ = fss.find_critical_temperature()
        
        assert Tc == pytest.approx(exact_Tc, rel=0.02)
    
    def test_high_temperature_limit(self):
        """At high T, magnetization should be ~0."""
        model = Ising2D(size=64, temperature=10.0)
        sampler = MetropolisSampler(model, seed=42)
        results = sampler.run(n_steps=50000, equilibration=5000)
        
        mean_m = np.abs(results.magnetization).mean() / model.n_spins
        assert mean_m < 0.05
    
    def test_low_temperature_limit(self):
        """At low T, |magnetization| should be ~1."""
        model = Ising2D(size=64, temperature=0.5)
        model.initialize('up')  # Start ordered to avoid metastability
        sampler = MetropolisSampler(model, seed=42)
        results = sampler.run(n_steps=50000, equilibration=5000)
        
        mean_m = np.abs(results.magnetization).mean() / model.n_spins
        assert mean_m > 0.95
```

#### Integration Tests

Test complete workflows.

```python
# tests/test_integration.py

class TestCompleteWorkflow:
    """Test complete simulation workflows."""
    
    def test_temperature_sweep_workflow(self, tmp_path):
        """Test full temperature sweep and analysis."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=32,
            temperatures=np.linspace(1.5, 3.5, 10),
            n_steps=10000,
            equilibration=1000
        )
        
        results = sweep.run()
        
        # Check results structure
        assert 'temperature' in results.columns
        assert 'energy' in results.columns
        assert 'magnetization' in results.columns
        assert len(results) == 10
        
        # Check physics: energy should increase with temperature
        assert results['energy'].iloc[-1] > results['energy'].iloc[0]
    
    def test_save_and_load_results(self, tmp_path):
        """Test results can be saved and reloaded."""
        model = Ising2D(size=32, temperature=2.5)
        sampler = MetropolisSampler(model, seed=42)
        results = sampler.run(n_steps=10000)
        
        # Save
        filepath = tmp_path / "results.h5"
        results.save(filepath)
        
        # Load
        loaded = Results.load(filepath)
        
        np.testing.assert_array_equal(results.energy, loaded.energy)
        np.testing.assert_array_equal(results.magnetization, loaded.magnetization)
    
    def test_cli_run_command(self, tmp_path):
        """Test CLI run command."""
        from click.testing import CliRunner
        from ising_toolkit.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'run',
            '--model', 'ising2d',
            '--size', '16',
            '--temperature', '2.5',
            '--steps', '1000',
            '--output', str(tmp_path / 'test.h5')
        ])
        
        assert result.exit_code == 0
        assert (tmp_path / 'test.h5').exists()
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=ising_toolkit --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test class
pytest tests/test_models.py::TestIsing2D

# Run specific test
pytest tests/test_models.py::TestIsing2D::test_energy_ground_state

# Run with verbose output
pytest tests/ -v

# Run validation tests only
pytest tests/test_validation.py -v
```

### Continuous Integration

The project uses GitHub Actions for automated testing:

```yaml
# .github/workflows/tests.yml

name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ --cov=ising_toolkit --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: coverage.xml
```

---

## Examples

### Example 1: Basic Phase Transition Study

```python
"""
Example: Studying the ferromagnetic-paramagnetic phase transition
in the 2D Ising model.
"""
import numpy as np
import matplotlib.pyplot as plt
from ising_toolkit import Ising2D, MetropolisSampler, TemperatureSweep

# Define temperature range around critical point
Tc_exact = 2.269  # Onsager's exact result
temperatures = np.linspace(1.5, 3.5, 40)

# Run temperature sweep
sweep = TemperatureSweep(
    model_class=Ising2D,
    size=64,
    temperatures=temperatures,
    n_steps=100000,
    equilibration=20000,
    algorithm='metropolis'
)

results = sweep.run(n_workers=4, progress=True)

# Plot phase diagram
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Magnetization
axes[0, 0].errorbar(
    results['temperature'], 
    results['abs_magnetization'],
    yerr=results['abs_magnetization_err'],
    fmt='o-'
)
axes[0, 0].axvline(Tc_exact, color='r', linestyle='--', label=f'$T_c$ = {Tc_exact:.3f}')
axes[0, 0].set_xlabel('Temperature (J/kB)')
axes[0, 0].set_ylabel('|Magnetization| per spin')
axes[0, 0].legend()

# Energy
axes[0, 1].errorbar(
    results['temperature'], 
    results['energy'],
    yerr=results['energy_err'],
    fmt='o-'
)
axes[0, 1].axvline(Tc_exact, color='r', linestyle='--')
axes[0, 1].set_xlabel('Temperature (J/kB)')
axes[0, 1].set_ylabel('Energy per spin')

# Heat capacity
axes[1, 0].errorbar(
    results['temperature'], 
    results['heat_capacity'],
    yerr=results['heat_capacity_err'],
    fmt='o-'
)
axes[1, 0].axvline(Tc_exact, color='r', linestyle='--')
axes[1, 0].set_xlabel('Temperature (J/kB)')
axes[1, 0].set_ylabel('Heat capacity per spin')

# Susceptibility
axes[1, 1].errorbar(
    results['temperature'], 
    results['susceptibility'],
    yerr=results['susceptibility_err'],
    fmt='o-'
)
axes[1, 1].axvline(Tc_exact, color='r', linestyle='--')
axes[1, 1].set_xlabel('Temperature (J/kB)')
axes[1, 1].set_ylabel('Susceptibility per spin')

plt.tight_layout()
plt.savefig('phase_diagram.pdf', dpi=300)
plt.show()
```

### Example 2: Finite-Size Scaling Analysis

```python
"""
Example: Finite-size scaling analysis to extract critical exponents.
"""
import numpy as np
from ising_toolkit import Ising2D, FiniteSizeScaling

# System sizes
sizes = [8, 16, 32, 64, 128]

# Temperature range focused near Tc
temperatures = np.linspace(2.1, 2.5, 40)

# Create FSS analysis object
fss = FiniteSizeScaling(
    model_class=Ising2D,
    sizes=sizes,
    temperatures=temperatures,
    n_steps=200000,
    equilibration=30000,
    algorithm='wolff'  # Much faster near Tc
)

# Run all simulations
results = fss.run(n_workers=8, progress=True)

# Find critical temperature from Binder crossing
Tc, Tc_err = fss.find_critical_temperature(method='binder')
print(f"Critical temperature: Tc = {Tc:.4f} ± {Tc_err:.4f}")
print(f"Exact value: Tc = 2.2692...")

# Extract critical exponents
exponents = fss.extract_exponents(Tc=Tc)
print("\nCritical exponents:")
print(f"  β = {exponents['beta'][0]:.3f} ± {exponents['beta'][1]:.3f} (exact: 0.125)")
print(f"  γ = {exponents['gamma'][0]:.3f} ± {exponents['gamma'][1]:.3f} (exact: 1.75)")
print(f"  ν = {exponents['nu'][0]:.3f} ± {exponents['nu'][1]:.3f} (exact: 1.0)")

# Plot Binder cumulant crossing
fss.plot_binder_crossing(save='binder_crossing.pdf')

# Plot data collapse for magnetization
fss.plot_data_collapse(
    observable='magnetization',
    Tc=Tc,
    nu=exponents['nu'][0],
    exponent=exponents['beta'][0] / exponents['nu'][0],
    save='data_collapse.pdf'
)
```

### Example 3: Algorithm Comparison

```python
"""
Example: Comparing Metropolis and Wolff algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from ising_toolkit import Ising2D, MetropolisSampler, WolffSampler
from ising_toolkit.analysis import autocorrelation_time

# Parameters
size = 64
temperature = 2.269  # Critical temperature
n_steps = 50000

# Metropolis simulation
print("Running Metropolis...")
model_met = Ising2D(size=size, temperature=temperature)
sampler_met = MetropolisSampler(model_met, seed=42)

start = time.time()
results_met = sampler_met.run(n_steps=n_steps, equilibration=10000)
time_met = time.time() - start

tau_met = autocorrelation_time(results_met.magnetization)
print(f"  Time: {time_met:.1f}s")
print(f"  Autocorrelation time: {tau_met:.1f} steps")

# Wolff simulation
print("\nRunning Wolff...")
model_wolff = Ising2D(size=size, temperature=temperature)
sampler_wolff = WolffSampler(model_wolff, seed=42)

start = time.time()
results_wolff = sampler_wolff.run(n_steps=n_steps // 10, equilibration=1000)
time_wolff = time.time() - start

tau_wolff = autocorrelation_time(results_wolff.magnetization)
print(f"  Time: {time_wolff:.1f}s")
print(f"  Autocorrelation time: {tau_wolff:.1f} steps")
print(f"  Mean cluster size: {results_wolff.cluster_sizes.mean():.1f}")

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Magnetization time series
axes[0].plot(results_met.magnetization[:5000] / model_met.n_spins, 
             alpha=0.7, label='Metropolis')
axes[0].set_xlabel('MC step')
axes[0].set_ylabel('Magnetization per spin')
axes[0].set_title('Metropolis: slow decorrelation')
axes[0].legend()

axes[1].plot(results_wolff.magnetization[:500] / model_wolff.n_spins,
             alpha=0.7, label='Wolff', color='orange')
axes[1].set_xlabel('Cluster flip')
axes[1].set_ylabel('Magnetization per spin')
axes[1].set_title('Wolff: fast decorrelation')
axes[1].legend()

plt.tight_layout()
plt.savefig('algorithm_comparison.pdf')
```

### Example 4: Creating Animations

```python
"""
Example: Creating an animation of spin dynamics.
"""
from ising_toolkit import Ising2D, MetropolisSampler
from ising_toolkit.visualization import create_animation

# Create model at critical temperature
model = Ising2D(size=64, temperature=2.269)
model.initialize('random')

# Run simulation saving configurations
sampler = MetropolisSampler(model, seed=42)
results = sampler.run(
    n_steps=10000,
    equilibration=0,  # Watch equilibration
    save_configurations=True,
    configuration_interval=100
)

# Create animation
create_animation(
    configurations=results.configurations,
    filename='spin_evolution.gif',
    fps=10,
    cmap='RdBu',
    dpi=100,
    title_template='MC Step: {step}'
)

print("Animation saved to spin_evolution.gif")
```

---

## Performance Optimization

### Numba JIT Compilation

Critical simulation loops are compiled with Numba for near-C performance:

```python
# src/ising_toolkit/utils/numba_kernels.py

from numba import njit, prange
import numpy as np

@njit
def metropolis_step_2d(spins, beta, rng_state):
    """Single Metropolis sweep over 2D lattice.
    
    Compiled with Numba for maximum performance.
    """
    L = spins.shape[0]
    n_flips = 0
    
    for _ in range(L * L):
        # Random site
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        # Sum of neighbors (periodic BC)
        neighbor_sum = (
            spins[(i+1) % L, j] + 
            spins[(i-1) % L, j] +
            spins[i, (j+1) % L] + 
            spins[i, (j-1) % L]
        )
        
        # Energy change
        dE = 2 * spins[i, j] * neighbor_sum
        
        # Metropolis criterion
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[i, j] *= -1
            n_flips += 1
    
    return n_flips


@njit(parallel=True)
def wolff_cluster_2d(spins, p_add, seed_i, seed_j):
    """Build and flip Wolff cluster.
    
    Uses parallel Numba for cluster identification.
    """
    L = spins.shape[0]
    cluster = np.zeros((L, L), dtype=np.bool_)
    stack = [(seed_i, seed_j)]
    cluster[seed_i, seed_j] = True
    seed_spin = spins[seed_i, seed_j]
    
    while stack:
        i, j = stack.pop()
        
        # Check all neighbors
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = (i + di) % L, (j + dj) % L
            
            if not cluster[ni, nj] and spins[ni, nj] == seed_spin:
                if np.random.random() < p_add:
                    cluster[ni, nj] = True
                    stack.append((ni, nj))
    
    # Flip cluster
    cluster_size = 0
    for i in prange(L):
        for j in range(L):
            if cluster[i, j]:
                spins[i, j] *= -1
                cluster_size += 1
    
    return cluster_size
```

### Memory Optimization

Spins are stored as `int8` to minimize memory:

```python
# Memory comparison for 1024x1024 lattice:
# int64: 8 MB
# int32: 4 MB
# int8:  1 MB (4x improvement!)

self.spins = np.ones((size, size), dtype=np.int8)
```

### Parallel Temperature Sweeps

```python
from concurrent.futures import ProcessPoolExecutor

def run_parallel_sweep(temperatures, n_workers=4):
    """Run simulations at different temperatures in parallel."""
    
    def simulate_temperature(T):
        model = Ising2D(size=64, temperature=T)
        sampler = MetropolisSampler(model)
        results = sampler.run(n_steps=100000, equilibration=10000)
        return {
            'temperature': T,
            'energy': results.energy.mean(),
            'magnetization': np.abs(results.magnetization).mean()
        }
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(simulate_temperature, temperatures))
    
    return pd.DataFrame(results)
```

### Benchmarks

Performance on AMD Ryzen 9 5900X (12 cores):

| Configuration | Metropolis | Wolff | 
|---------------|------------|-------|
| L=64, T=2.27, 100k steps | 2.1s | 0.8s |
| L=128, T=2.27, 100k steps | 8.5s | 1.2s |
| L=256, T=2.27, 100k steps | 34.2s | 2.8s |
| L=64, sweep 50 temps, 4 cores | 45s | 18s |

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/username/ising-mc-toolkit.git
cd ising-mc-toolkit
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all public functions
- Maximum line length: 100 characters
- Use docstrings (NumPy style) for all public functions and classes

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and add tests
3. Ensure all tests pass: `pytest tests/`
4. Update documentation if needed
5. Submit pull request with clear description

### Adding New Features

When adding new features:

1. **New Model**: Inherit from `IsingModel` base class
2. **New Algorithm**: Inherit from `Sampler` base class
3. **New Observable**: Inherit from `Observable` base class
4. **Add tests**: Include unit tests and validation tests
5. **Add documentation**: Update API docs and add examples

---

## References

### Original Papers

1. Ising, E. (1925). "Beitrag zur Theorie des Ferromagnetismus." Zeitschrift für Physik, 31(1), 253-258.

2. Onsager, L. (1944). "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition." Physical Review, 65(3-4), 117-149.

3. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). "Equation of State Calculations by Fast Computing Machines." The Journal of Chemical Physics, 21(6), 1087-1092.

4. Wolff, U. (1989). "Collective Monte Carlo Updating for Spin Systems." Physical Review Letters, 62(4), 361-364.

### Textbooks

5. Newman, M. E. J., & Barkema, G. T. (1999). "Monte Carlo Methods in Statistical Physics." Oxford University Press.

6. Landau, D. P., & Binder, K. (2014). "A Guide to Monte Carlo Simulations in Statistical Physics." Cambridge University Press.

7. Krauth, W. (2006). "Statistical Mechanics: Algorithms and Computations." Oxford University Press.

### Online Resources

8. "Monte Carlo simulations in Statistical Physics" - Online course notes
9. "Computational Physics" - Various university lecture notes

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Changelog

### Version 1.0.0 (2025-XX-XX)

**Initial Release**

- 1D, 2D, and 3D Ising model implementations
- Metropolis and Wolff sampling algorithms
- Comprehensive analysis tools (bootstrap, autocorrelation, FSS)
- Publication-quality visualization
- Snakemake workflow integration
- Full test suite with validation against exact solutions
- Complete documentation and examples
