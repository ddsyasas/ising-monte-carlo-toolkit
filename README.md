<h1 align="center">Ising Monte Carlo Toolkit</h1>

<p align="center">
  <strong>A high-performance Monte Carlo simulation framework for studying magnetic phase transitions in the Ising model across 1D, 2D, and 3D lattices.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/version-0.1.0-green?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/physics-statistical_mechanics-red?style=for-the-badge" alt="Physics">
  <img src="https://img.shields.io/badge/numba-JIT_accelerated-orange?style=for-the-badge&logo=numba" alt="Numba">
  <a href="https://github.com/ddsyasas/ising-monte-carlo-toolkit/actions/workflows/ci.yml"><img src="https://github.com/ddsyasas/ising-monte-carlo-toolkit/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &nbsp;&bull;&nbsp;
  <a href="#features">Features</a> &nbsp;&bull;&nbsp;
  <a href="#results">Results</a> &nbsp;&bull;&nbsp;
  <a href="#cli-reference">CLI Reference</a> &nbsp;&bull;&nbsp;
  <a href="#python-api">Python API</a> &nbsp;&bull;&nbsp;
  <a href="docs/USAGE.md">Full Documentation</a>
</p>

---

<p align="center">
  <img src="docs/images/cli_phase_diagram.png" width="48%" alt="Phase Diagram">
  &nbsp;
  <img src="docs/images/cli_snapshot.png" width="48%" alt="Spin Configuration">
</p>

<p align="center"><em>Left: Phase diagram of the 2D Ising model showing the ferromagnetic-paramagnetic transition at T<sub>c</sub> ≈ 2.269. Right: Spin configuration snapshot at the critical temperature (32×32 lattice).</em></p>

---

## About The Project

The Ising model is one of the most important models in statistical mechanics — a lattice of interacting spins that exhibits a phase transition between ordered (ferromagnetic) and disordered (paramagnetic) states. This toolkit provides a complete framework for simulating, analyzing, and visualizing Ising model behavior across all three physical dimensions.

**The Hamiltonian:**

$$H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j$$

where $J > 0$ is the ferromagnetic coupling and $\sigma_i = \pm 1$ are spin variables on each lattice site.

### Why This Toolkit?

- **Multi-dimensional**: Simulate 1D chains, 2D square lattices, and 3D cubic lattices
- **Publication-ready**: Generate high-quality plots suitable for academic papers
- **Educational**: Learn Monte Carlo methods with clear, well-documented code
- **Performant**: Optional Numba JIT compilation for significant speedups
- **Complete**: From simulation to analysis to visualization — all in one toolkit

---

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Results Gallery](#results)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Architecture](#architecture)
- [Model Compatibility](#model-compatibility)
- [Physical Background](#physical-background)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)

---

## Quick Start

### Prerequisites

- Python 3.9+
- NumPy, Matplotlib, Click, SciPy

### Installation

```bash
git clone https://github.com/ddsyasas/ising-monte-carlo-toolkit.git
cd ising-monte-carlo-toolkit
pip install -e .
```

### Run Your First Simulation

```bash
# Simulate the 2D Ising model at the critical temperature
python -m src.ising_toolkit.cli run -m ising2d -L 32 -T 2.269 -n 1000 --save-spins -o my_simulation.npz
```

```
Running ising2d simulation at T=2.2690...
Simulating  [####################################]  100%

==================================================
Simulation Results
==================================================
Model:            ising2d
Size:             32
Temperature:      2.2690
Algorithm:        metropolis
Steps:            1000
--------------------------------------------------
Energy/spin:      -1.464766 ± 0.087105
|Magnetization|:  0.727715 ± 0.087465
Heat capacity:    1.5091
Susceptibility:   3.4525
==================================================
```

### Generate a Phase Diagram in 2 Commands

```bash
# Step 1: Temperature sweep
python -m src.ising_toolkit.cli sweep -m ising2d -L 16 \
    --temp-start 1.5 --temp-end 3.5 --temp-steps 15 \
    --steps 5000 -o sweep_results/

# Step 2: Plot
python -m src.ising_toolkit.cli plot sweep_results/ \
    --type phase_diagram -f png -o phase_diagram.png
```

---

## Features

### Simulation Engine

| Feature | Description |
|---------|-------------|
| **3 Lattice Dimensions** | 1D chain, 2D square lattice, 3D cubic lattice |
| **2 Algorithms** | Metropolis single-spin flip, Wolff cluster algorithm |
| **Boundary Conditions** | Periodic and fixed boundary conditions |
| **Initial States** | Random, all-up, all-down, checkerboard configurations |
| **Reproducibility** | Seed-based random number generation |
| **Numba Acceleration** | Optional JIT-compiled kernels for 5-10x speedup |

### Analysis Tools

| Feature | Description |
|---------|-------------|
| **Bootstrap Error** | Bootstrap resampling with confidence intervals |
| **Jackknife Error** | Leave-one-out error estimation |
| **Blocking Analysis** | Error estimation for correlated data with automatic plateau detection |
| **Autocorrelation** | FFT-based autocorrelation with integrated correlation time |
| **Finite-Size Scaling** | Extract critical exponents and locate T<sub>c</sub> |
| **Temperature Sweeps** | Parallel sweep across temperature ranges |

### Visualization

| Feature | Description |
|---------|-------------|
| **Phase Diagrams** | Energy, magnetization, heat capacity, susceptibility vs temperature |
| **Spin Snapshots** | 1D bar, 2D heatmap, 3D cross-sectional slices |
| **Time Series** | Observable fluctuations during simulation |
| **Autocorrelation Plots** | Correlation function with exponential and integrated times |
| **Binder Cumulant** | Crossing analysis for finite-size scaling |
| **Animations** | GIF/MP4 of spin evolution during simulation |
| **Publication Styles** | Pre-configured styles for papers, presentations, and notebooks |

### I/O & Configuration

| Feature | Description |
|---------|-------------|
| **Output Formats** | CSV, NPZ (NumPy), HDF5 |
| **YAML Config** | Load simulation parameters from configuration files |
| **Spin Compression** | 8x compression via bit-packing for large lattices |
| **Parallel Execution** | Multi-worker temperature sweeps |

---

## Results

### 2D Ising Model — Phase Transition at T<sub>c</sub> ≈ 2.269

The 2D Ising model on a square lattice has an exact analytical solution (Onsager, 1944). The phase transition is clearly visible in both simulation output and plots.

**Simulation at the critical temperature:**
```
==================================================
Simulation Results
==================================================
Model:            ising2d
Size:             32
Temperature:      2.2690
Algorithm:        metropolis
Steps:            1000
--------------------------------------------------
Energy/spin:      -1.464766 ± 0.087105
|Magnetization|:  0.727715 ± 0.087465
Heat capacity:    1.5091
Susceptibility:   3.4525
==================================================
```

**Wolff cluster algorithm** produces consistent results with better decorrelation near T<sub>c</sub>:
```
==================================================
Simulation Results (Wolff)
==================================================
Model:            ising2d
Size:             32
Temperature:      2.2690
Algorithm:        wolff
Steps:            2000
--------------------------------------------------
Energy/spin:      -1.427520 ± 0.103856
|Magnetization|:  0.639385 ± 0.182454
Heat capacity:    2.1453
Susceptibility:   15.0235
==================================================
```

**Temperature sweep** automatically identifies the transition:
```
============================================================
Sweep Complete!
============================================================

  Size    T_peak(χ)        χ_max    T_peak(C)        C_max
------------------------------------------------------------
    16       2.5000       5.8370       2.3000       1.5346
============================================================
```

<p align="center">
  <img src="docs/images/cli_phase_diagram.png" width="70%" alt="2D Phase Diagram">
</p>

**Key observations:**
- **Magnetization** drops sharply from ~1 to ~0 at T<sub>c</sub> — the order parameter vanishes
- **Susceptibility** diverges at T<sub>c</sub> (sharp peak) — fluctuations are maximal
- **Heat capacity** peaks at T<sub>c</sub> — the system absorbs maximum energy during the transition
- **Energy** rises smoothly from the ground state (-2J per spin) toward 0

<p align="center">
  <img src="docs/images/cli_snapshot.png" width="45%" alt="2D Snapshot">
  &nbsp;&nbsp;
  <img src="docs/images/timeseries_energy.png" width="45%" alt="2D Timeseries">
</p>

<p align="center"><em>Left: Spin configuration at T<sub>c</sub> — large correlated domains of both up (white) and down (black) spins coexist. Right: Energy time series showing large fluctuations characteristic of critical slowing down.</em></p>

<p align="center">
  <img src="docs/images/autocorr_energy.png" width="60%" alt="2D Autocorrelation">
</p>

<p align="center"><em>Autocorrelation function of energy at T<sub>c</sub>, with integrated autocorrelation time τ<sub>int</sub> ≈ 1.8 MC steps and exponential decay time τ<sub>exp</sub> ≈ 3 MC steps.</em></p>

**Statistical analysis with bootstrap error estimation:**
```
======================================================================
Analysis Results
======================================================================
Source: snapshot_run.npz
Model: ising2d | Size: 32 | Temperature: 2.2690 | Samples: 100
----------------------------------------------------------------------
Observable                   Mean          Std        Error
----------------------------------------------------------------------
energy                  -1.464766     0.087105     0.008958
magnetization            0.727715     0.087465     0.008698
======================================================================
```

---

### 1D Ising Model — No Phase Transition

The 1D Ising model has **no phase transition at finite temperature** (Ising, 1925). The system is paramagnetic for all T > 0 — a fundamental result in statistical mechanics.

**Simulation output:**
```
==================================================
Simulation Results
==================================================
Model:            ising1d
Size:             100
Temperature:      1.5000
Algorithm:        metropolis
Steps:            2000
--------------------------------------------------
Energy/spin:      -0.582400 ± 0.077216
|Magnetization|:  0.150600 ± 0.111945
Heat capacity:    0.2650
Susceptibility:   0.8354
==================================================
```

> Note: Even at T=1.5 (relatively low), the magnetization is only ~0.15 — there is no long-range order. Compare with the 2D model which has |M| ≈ 0.73 at a similar distance below T<sub>c</sub>.

**Temperature sweep** confirms the absence of a sharp transition:
```
============================================================
Sweep Complete!
============================================================

  Size    T_peak(χ)        χ_max    T_peak(C)        C_max
------------------------------------------------------------
    64       0.5000       9.7299       1.0000       0.4025
============================================================
```

The susceptibility peaks at the lowest measured temperature (T=0.5) rather than at a finite T<sub>c</sub>, and the heat capacity maximum (Schottky anomaly) is broad — both signatures of a system with no true phase transition.

<p align="center">
  <img src="docs/images/ising1d_phase_diagram.png" width="48%" alt="1D Phase Diagram">
  &nbsp;
  <img src="docs/images/ising1d_snapshot.png" width="48%" alt="1D Snapshot">
</p>

<p align="center"><em>Left: 1D phase diagram — all observables change gradually with no sharp transition. Right: 1D spin chain at T=1.5 showing short-range domains but no long-range alignment.</em></p>

---

### 3D Ising Model — Phase Transition at T<sub>c</sub> ≈ 4.511

The 3D cubic lattice model has no exact analytical solution but shows a well-studied phase transition. Each spin has 6 nearest neighbors (vs 4 in 2D), producing a higher critical temperature.

**Simulation at the critical temperature:**
```
==================================================
Simulation Results
==================================================
Model:            ising3d
Size:             10
Temperature:      4.5110
Algorithm:        metropolis
Steps:            1000
--------------------------------------------------
Energy/spin:      -1.089360 ± 0.189763
|Magnetization|:  0.346940 ± 0.163790
Heat capacity:    1.7696
Susceptibility:   5.9471
==================================================
```

**Wolff algorithm** also works for 3D, producing consistent results:
```
==================================================
Simulation Results (Wolff)
==================================================
Model:            ising3d
Size:             10
Temperature:      4.5110
Algorithm:        wolff
Steps:            1000
--------------------------------------------------
Energy/spin:      -1.069520 ± 0.170066
|Magnetization|:  0.317860 ± 0.162530
Heat capacity:    1.4213
Susceptibility:   5.8559
==================================================
```

**Temperature sweep** locates the transition:
```
============================================================
Sweep Complete!
============================================================

  Size    T_peak(χ)        χ_max    T_peak(C)        C_max
------------------------------------------------------------
     8       4.5000       3.5030       4.3000       1.3644
============================================================
```

Both susceptibility (peak at T=4.5) and heat capacity (peak at T=4.3) correctly identify the transition near the known T<sub>c</sub> ≈ 4.511.

<p align="center">
  <img src="docs/images/ising3d_phase_diagram.png" width="48%" alt="3D Phase Diagram">
  &nbsp;
  <img src="docs/images/ising3d_snapshot.png" width="48%" alt="3D Snapshot">
</p>

<p align="center"><em>Left: 3D phase diagram showing a clear transition near T<sub>c</sub> ≈ 4.5 — magnetization drops sharply, susceptibility and heat capacity peak. Right: Cross-sectional slices through a 10×10×10 cubic lattice at the critical temperature, showing large mixed-spin domains.</em></p>

### Comparing All Three Models

| Property | 1D | 2D | 3D |
|----------|:--:|:--:|:--:|
| T<sub>c</sub> (J/k<sub>B</sub>) | 0 (none) | 2.269 | ≈ 4.511 |
| \|M\| at T<sub>c</sub> | — | ~0.73 | ~0.35 |
| χ<sub>max</sub> | 9.73 (at T→0) | 5.84 | 3.50 |
| C<sub>max</sub> | 0.40 | 1.53 | 1.36 |
| Exact solution? | Yes | Yes | No |

---

## CLI Reference

The toolkit provides a full-featured command-line interface:

```bash
python -m src.ising_toolkit.cli [COMMAND] [OPTIONS]
```

### Commands Overview

| Command | Description |
|---------|-------------|
| `run` | Run a single Monte Carlo simulation |
| `sweep` | Temperature sweep for phase diagrams |
| `plot` | Generate visualizations from results |
| `analyze` | Statistical analysis with bootstrap errors |
| `info` | Display model theoretical properties |
| `benchmark` | Compare algorithm performance |

### `run` — Single Simulation

```bash
python -m src.ising_toolkit.cli run -m MODEL -L SIZE -T TEMP [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model` | `-m` | `ising1d`, `ising2d`, `ising3d` | Required |
| `--size` | `-L` | Lattice size | Required |
| `--temperature` | `-T` | Temperature (J/k<sub>B</sub>) | Required |
| `--steps` | `-n` | MC steps | 100000 |
| `--equilibration` | `-e` | Equilibration steps | 10000 |
| `--algorithm` | `-a` | `metropolis` or `wolff` | metropolis |
| `--save-spins` | | Save final spin configuration | False |
| `--output` | `-o` | Output file (.npz) | None |
| `--seed` | `-s` | Random seed | None |
| `--config` | `-c` | YAML config file | None |

**Examples:**
```bash
# 2D at critical temperature with Wolff algorithm
python -m src.ising_toolkit.cli run -m ising2d -L 64 -T 2.269 -a wolff -n 5000 -o results.npz

# 3D with spin snapshot
python -m src.ising_toolkit.cli run -m ising3d -L 10 -T 4.511 --save-spins -o snapshot_3d.npz

# 1D chain
python -m src.ising_toolkit.cli run -m ising1d -L 200 -T 1.5 -n 10000 -o chain.npz
```

### `sweep` — Temperature Sweep

```bash
python -m src.ising_toolkit.cli sweep -m MODEL -L SIZE --temp-start T1 --temp-end T2 [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--size` | `-L` | Lattice size(s) — repeatable | Required |
| `--temp-start` | | Start temperature | Required |
| `--temp-end` | | End temperature | Required |
| `--temp-steps` | | Number of temperature points | 50 |
| `--parallel` | `-p` | Parallel workers | 1 |
| `--format` | `-f` | `csv`, `npz`, `hdf5` | csv |
| `--output` | `-o` | Output directory | Required |

**Examples:**
```bash
# Basic phase diagram
python -m src.ising_toolkit.cli sweep -m ising2d -L 32 \
    --temp-start 1.5 --temp-end 3.5 --temp-steps 20 \
    --steps 5000 -o sweep_results/

# Finite-size scaling (multiple sizes)
python -m src.ising_toolkit.cli sweep -m ising2d -L 8 -L 16 -L 32 -L 64 \
    --temp-start 2.0 --temp-end 2.5 --temp-steps 20 -o fss_results/

# Parallel execution
python -m src.ising_toolkit.cli sweep -m ising2d -L 32 \
    --temp-start 1.5 --temp-end 3.5 -p 4 -o parallel_results/
```

### `plot` — Visualization

```bash
python -m src.ising_toolkit.cli plot INPUT --type TYPE [OPTIONS]
```

| Plot Type | Input | Description |
|-----------|-------|-------------|
| `phase_diagram` | Sweep directory | All observables vs temperature |
| `snapshot` | Run .npz (with `--save-spins`) | Spin configuration visualization |
| `timeseries` | Run .npz | Observable vs MC step |
| `autocorrelation` | Run .npz | Correlation function with τ<sub>int</sub> |
| `binder` | Multi-size sweep directory | Binder cumulant crossing |
| `animation` | Run .npz (2D only) | Spin evolution GIF/MP4 |

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--observable` | `-O` | Observable for timeseries/autocorrelation | magnetization |
| `--format` | `-f` | `png`, `pdf`, `svg` | pdf |
| `--style` | `-s` | `publication`, `presentation`, `default` | publication |
| `--dpi` | | Resolution | 300 |
| `--output` | `-o` | Output file | Auto |

**Examples:**
```bash
# Phase diagram
python -m src.ising_toolkit.cli plot sweep_results/ --type phase_diagram -f png -o phase.png

# Spin snapshot
python -m src.ising_toolkit.cli plot snapshot.npz --type snapshot -f png -o spins.png

# Energy time series
python -m src.ising_toolkit.cli plot results.npz --type timeseries -O energy -f png -o energy.png

# Autocorrelation
python -m src.ising_toolkit.cli plot results.npz --type autocorrelation -O energy -f png -o acf.png
```

### `analyze` — Statistical Analysis

```bash
python -m src.ising_toolkit.cli analyze INPUT [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--observables` | `-O` | Observables (comma-separated or `all`) | all |
| `--bootstrap` | `-b` | Bootstrap samples | 1000 |
| `--format` | `-f` | `csv`, `json`, `hdf5` | csv |
| `--output` | `-o` | Output file | None (print only) |

**Example:**
```bash
python -m src.ising_toolkit.cli analyze results.npz
```

```
======================================================================
Analysis Results
======================================================================
Source: results.npz
Model: ising2d
Size: 32
Temperature: 2.2690
Samples: 100
----------------------------------------------------------------------
Observable                   Mean          Std        Error
----------------------------------------------------------------------
energy                  -1.464766     0.087105     0.008958
magnetization            0.727715     0.087465     0.008698
======================================================================
```

### `info` — Model Information

```bash
python -m src.ising_toolkit.cli info --model ising2d
```

```
==================================================
Model Information: ising2d
==================================================

2D Ising Model (Square Lattice)
--------------------------------------------------
Critical temperature:  Tc = 2.269185 J/kB
                          = 2/ln(1+√2)

Critical exponents (exact):
  β  = 0.125000  (magnetization)
  γ  = 1.750000  (susceptibility)
  ν  = 1.000000  (correlation length)
  α  = 0.000000  (heat capacity, log divergence)

Exact solution: Onsager (1944)

==================================================
```

### `benchmark` — Performance Test

```bash
python -m src.ising_toolkit.cli benchmark
```

```
==================================================
Performance Benchmark
==================================================

Temperature: T = 2.269 (critical)
Steps: 10000 per test

  Size   Metropolis (s)        Wolff (s)    Speedup
--------------------------------------------------
    16            0.404            2.547       0.16x
    32            0.315            8.379       0.04x
    64            1.281           29.123       0.04x
==================================================
```

---

## Python API

For programmatic access and advanced workflows:

```python
from src.ising_toolkit.models import Ising2D
from src.ising_toolkit.samplers import MetropolisSampler, WolffSampler
import numpy as np

# Create model at the critical temperature
model = Ising2D(size=32, temperature=2.269)
sampler = MetropolisSampler(model)

# Equilibrate
for _ in range(1000):
    sampler.step()

# Collect measurements
energies, magnetizations = [], []
for _ in range(5000):
    sampler.step()
    energies.append(model.get_energy() / model.n_spins)
    magnetizations.append(np.abs(model.get_magnetization()) / model.n_spins)

print(f"Energy:        {np.mean(energies):.4f} +/- {np.std(energies):.4f}")
print(f"Magnetization: {np.mean(magnetizations):.4f} +/- {np.std(magnetizations):.4f}")
```

### Available Models

```python
from src.ising_toolkit.models import Ising1D, Ising2D, Ising3D

chain  = Ising1D(size=200, temperature=1.5)         # 200-site chain
square = Ising2D(size=64,  temperature=2.269)        # 64x64 lattice
cube   = Ising3D(size=16,  temperature=4.511)        # 16x16x16 lattice
```

### Analysis Tools

```python
from src.ising_toolkit.analysis import (
    calculate_heat_capacity,
    calculate_susceptibility,
    bootstrap_error,
    autocorrelation_function_fft,
    integrated_autocorrelation_time,
)

# Bootstrap error estimation
mean, error = bootstrap_error(energies, n_bootstrap=1000)

# Autocorrelation analysis
acf = autocorrelation_function_fft(energies)
tau_int = integrated_autocorrelation_time(energies)
```

### Visualization

```python
from src.ising_toolkit.visualization import (
    plot_phase_diagram,
    plot_spin_configuration,
    plot_time_series,
    plot_autocorrelation,
)
```

---

## Architecture

```
ising-monte-carlo-toolkit/
├── src/ising_toolkit/
│   ├── models/                 # Lattice models
│   │   ├── base.py             #   Base model interface
│   │   ├── ising1d.py          #   1D chain (N spins, 2 neighbors)
│   │   ├── ising2d.py          #   2D square lattice (N² spins, 4 neighbors)
│   │   └── ising3d.py          #   3D cubic lattice (N³ spins, 6 neighbors)
│   │
│   ├── samplers/               # Monte Carlo algorithms
│   │   ├── base.py             #   Base sampler interface
│   │   ├── metropolis.py       #   Metropolis single-spin flip
│   │   └── wolff.py            #   Wolff cluster algorithm
│   │
│   ├── analysis/               # Statistical analysis
│   │   ├── observables.py      #   Heat capacity, susceptibility, Binder cumulant
│   │   ├── statistics.py       #   Bootstrap, jackknife, blocking methods
│   │   ├── sweep.py            #   Temperature sweep utilities
│   │   └── finite_size.py      #   Finite-size scaling & critical exponents
│   │
│   ├── visualization/          # Plotting & animation
│   │   ├── plots.py            #   Phase diagrams, snapshots, timeseries, ACF
│   │   ├── animation.py        #   Spin evolution animations
│   │   └── styles.py           #   Publication, presentation, notebook styles
│   │
│   ├── io/                     # Input/output
│   │   ├── results.py          #   NPZ/CSV/HDF5 reading & writing
│   │   ├── compression.py      #   8x bit-packing for spin configurations
│   │   └── config.py           #   YAML configuration loading
│   │
│   ├── utils/                  # Utilities
│   │   ├── constants.py        #   Physical constants & critical values
│   │   ├── exceptions.py       #   Custom exception hierarchy
│   │   ├── validation.py       #   Input validation
│   │   ├── numba_kernels.py    #   JIT-compiled MC kernels
│   │   └── parallel.py         #   Multi-process execution
│   │
│   └── cli.py                  # Command-line interface (Click)
│
├── tests/                      # 18 test modules
├── benchmarks/                 # Performance benchmarks
├── docs/
│   ├── USAGE.md                # Comprehensive usage guide
│   └── images/                 # Result images for documentation
└── workflow/                   # Snakemake workflow
```

---

## Model Compatibility

| Feature | 1D (`ising1d`) | 2D (`ising2d`) | 3D (`ising3d`) |
|---------|:-:|:-:|:-:|
| Metropolis algorithm | ✅ | ✅ | ✅ |
| Wolff cluster algorithm | — | ✅ | ✅ |
| `plot snapshot` | ✅ bar | ✅ heatmap | ✅ z-slices |
| `plot animation` | — | ✅ | — |
| `benchmark` | — | ✅ | — |
| All other commands | ✅ | ✅ | ✅ |

> **Why no Wolff for 1D?** The Wolff cluster algorithm requires ≥2 dimensions to form meaningful clusters. For 1D, use Metropolis (the default).

---

## Physical Background

### Critical Temperatures

| Dimension | T<sub>c</sub> (J/k<sub>B</sub>) | Solution | Reference |
|:---------:|:---:|:---:|:---:|
| **1D** | 0 (no transition) | Exact | Ising (1925) |
| **2D** | 2.269185... = 2/ln(1+√2) | Exact | Onsager (1944) |
| **3D** | ≈ 4.511 | Numerical | Monte Carlo studies |

### Critical Exponents

| Exponent | Meaning | 2D (exact) | 3D (numerical) |
|:--------:|---------|:----------:|:---------------:|
| β | Magnetization: M ~ (T<sub>c</sub>-T)<sup>β</sup> | 1/8 | 0.326 |
| γ | Susceptibility: χ ~ \|T-T<sub>c</sub>\|<sup>-γ</sup> | 7/4 | 1.237 |
| ν | Correlation length: ξ ~ \|T-T<sub>c</sub>\|<sup>-ν</sup> | 1 | 0.630 |
| α | Heat capacity: C ~ \|T-T<sub>c</sub>\|<sup>-α</sup> | 0 (log) | 0.110 |

### Observables

| Observable | Formula | Physical Meaning |
|------------|---------|-----------------|
| Energy per spin | ⟨E⟩/N | Average interaction energy |
| Magnetization | ⟨\|M\|⟩/N | Net spin alignment (order parameter) |
| Heat capacity | C = (⟨E²⟩ - ⟨E⟩²) / (k<sub>B</sub>T²) | Energy fluctuations |
| Susceptibility | χ = (⟨M²⟩ - ⟨M⟩²) / (k<sub>B</sub>T) | Magnetization fluctuations |

---

## Benchmarks

The toolkit includes a benchmark suite comparing algorithm performance and testing Numba acceleration:

```bash
# Built-in CLI benchmark (Metropolis vs Wolff)
python -m src.ising_toolkit.cli benchmark

# Full benchmark suite
python benchmarks/benchmark_suite.py

# Numba kernel benchmarks
python benchmarks/benchmark_numba_kernels.py

# Memory optimization benchmarks
python benchmarks/benchmark_memory.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| **NumPy** | Array operations, random numbers |
| **Matplotlib** | Visualization and plotting |
| **Click** | Command-line interface |
| **SciPy** | Statistical functions |
| **h5py** | HDF5 file format support |
| **PyYAML** | Configuration file parsing |
| **Numba** *(optional)* | JIT compilation for 5-10x speedup |

---

## Contributing

Contributions are welcome! Areas where help is needed:

- **Additional models**: XY model, Heisenberg model, Potts model
- **Advanced algorithms**: Swendsen-Wang, parallel tempering
- **GPU acceleration**: CUDA kernels for large-scale simulations
- **Visualization**: Interactive 3D spin visualization

---

## References

1. Ising, E. (1925). Beitrag zur Theorie des Ferromagnetismus. *Zeitschrift fur Physik*, 31, 253-258.
2. Onsager, L. (1944). Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition. *Physical Review*, 65(3-4), 117-149.
3. Metropolis, N. et al. (1953). Equation of State Calculations by Fast Computing Machines. *J. Chem. Phys.*, 21, 1087-1092.
4. Wolff, U. (1989). Collective Monte Carlo Updating for Spin Systems. *Physical Review Letters*, 62(4), 361-364.

---

<p align="center">
  <em>Built for physicists, by an upcoming physicist — <strong>Sajana Yasas</strong></em>
</p>
