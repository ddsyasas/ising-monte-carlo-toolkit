# Contributing

Contributions to the Ising Monte Carlo Toolkit are welcome!

---

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/ising-monte-carlo-toolkit.git
   cd ising-monte-carlo-toolkit
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run the test suite:
   ```bash
   pytest
   ```

---

## Areas Where Help Is Needed

### New Models

- **XY Model**: Continuous spin angles on a 2D lattice (Kosterlitz-Thouless transition)
- **Heisenberg Model**: 3D spin vectors
- **Potts Model**: q-state generalization of the Ising model
- **Ising Model on Other Lattices**: Triangular, honeycomb, kagome

### Advanced Algorithms

- **Swendsen-Wang**: Multi-cluster algorithm
- **Parallel Tempering**: Replica exchange for better sampling
- **Wang-Landau**: Flat-histogram sampling for density of states
- **Multicanonical Methods**: Enhanced sampling near critical temperature

### Performance

- **GPU Acceleration**: CUDA kernels for large-scale simulations
- **Improved Numba Kernels**: Better vectorization and parallelism
- **Memory Optimization**: Further compression for very large lattices

### Visualization

- **Interactive 3D**: Rotate and explore 3D spin configurations
- **Streamlit Dashboard**: Web-based interactive simulation interface
- **Domain Wall Visualization**: Highlight interfaces between spin domains

### Analysis

- **Finite-Size Scaling Automation**: Automated extraction of critical exponents
- **Error Propagation**: Systematic uncertainty tracking through analysis pipeline
- **Comparison with Exact Results**: Automated validation against known solutions

---

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write your code with clear docstrings
4. Add tests if applicable
5. Run the linter: `flake8 src/ising_toolkit/`
6. Submit a pull request

---

## Repository Structure

```
ising-monte-carlo-toolkit/
├── src/ising_toolkit/
│   ├── models/           # Lattice models (add new models here)
│   ├── samplers/         # MC algorithms (add new algorithms here)
│   ├── analysis/         # Statistical tools
│   ├── visualization/    # Plotting
│   ├── io/               # File I/O
│   ├── utils/            # Utilities
│   └── cli.py            # CLI entry point
├── tests/                # Test suite
├── benchmarks/           # Performance benchmarks
├── docs/                 # Documentation
└── workflow/             # Snakemake workflow
```

---

## Code Style

- Follow PEP 8 (enforced via `flake8` with max line length 99)
- Use type hints for function signatures
- Write docstrings in NumPy style
- Format with `black` (line length 88)

---

## Contact

This project is maintained by **Sajana Yasas**.

- GitHub: [@ddsyasas](https://github.com/ddsyasas)
- Repository: [ising-monte-carlo-toolkit](https://github.com/ddsyasas/ising-monte-carlo-toolkit)
