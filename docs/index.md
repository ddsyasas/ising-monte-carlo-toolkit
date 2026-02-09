# Ising Monte Carlo Toolkit

A high-performance Monte Carlo simulation framework for studying magnetic phase transitions in the Ising model across 1D, 2D, and 3D lattices.

## Features

- **Multiple dimensions**: 1D chain, 2D square lattice, 3D cubic lattice
- **Sampling algorithms**: Metropolis single-spin flip, Wolff cluster algorithm
- **Numba JIT acceleration**: Up to 100x speedup over pure Python
- **Analysis tools**: Heat capacity, susceptibility, Binder cumulant, finite-size scaling
- **Statistical error estimation**: Bootstrap, jackknife, and blocking methods
- **Visualization**: Phase diagrams, spin configurations, animations
- **CLI interface**: Full-featured command-line tool for running simulations

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
from ising_toolkit.models import Ising2D
from ising_toolkit.samplers import MetropolisSampler

model = Ising2D(size=32)
sampler = MetropolisSampler(model, seed=42)
results = sampler.sample(temperature=2.269, n_steps=10000)
```

## Navigation

- [Usage Guide](USAGE.md) — detailed examples and tutorials
- [API Reference](api/models.md) — auto-generated from docstrings
