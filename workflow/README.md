# Ising Model Simulation Workflow

## Overview

This Snakemake workflow automates parameter sweeps and analysis for Ising model Monte Carlo simulations. It handles:

- Running simulations across multiple system sizes and temperatures
- Calculating observables with bootstrap error estimates
- Finding the critical temperature via Binder cumulant crossing
- Extracting critical exponents from finite-size scaling
- Generating publication-quality figures

## Directory Structure

```
workflow/
├── Snakefile              # Main workflow definition
├── config/
│   └── config.yaml        # Configuration parameters
├── rules/
│   ├── analyze.smk        # Analysis rules (observables, Tc, exponents)
│   └── figures.smk        # Visualization rules (plots, animations)
├── scripts/
│   ├── collect_results.py        # Aggregate simulation results
│   ├── calculate_observables.py  # Compute observables with errors
│   ├── aggregate_sweep.py        # Combine temperature sweep data
│   ├── find_Tc.py                # Critical temperature estimation
│   ├── extract_exponents.py      # Critical exponent extraction
│   ├── compute_binder.py         # Binder cumulant calculation
│   ├── compute_binder_single.py  # Single simulation Binder
│   ├── aggregate_binder.py       # Aggregate Binder data
│   ├── compute_xi.py             # Correlation length estimation
│   ├── analyze_critical.py       # Critical point analysis
│   ├── plot_phase_diagram.py     # Phase diagram figure
│   ├── plot_binder.py            # Binder crossing plot
│   ├── plot_collapse.py          # FSS collapse plot
│   ├── plot_fss.py               # Finite-size scaling plot
│   ├── plot_susceptibility.py    # Susceptibility plot
│   ├── plot_specific_heat.py     # Specific heat plot
│   ├── plot_correlation.py       # Correlation function plot
│   ├── plot_observable.py        # Generic observable plot
│   ├── plot_snapshots.py         # Spin configuration snapshots
│   ├── create_animation.py       # Spin evolution animation
│   └── generate_report.py        # HTML summary report
├── logs/                  # Snakemake log files (created at runtime)
├── benchmarks/            # Timing benchmarks (created at runtime)
└── README.md              # This file
```

## Configuration

Edit `config/config.yaml` to customize simulation parameters:

```yaml
# =============================================================================
# Model Configuration
# =============================================================================

# Model type: ising2d, ising3d, ising1d
model: ising2d

# System sizes to simulate (list of integers)
# For finite-size scaling, use at least 3-4 sizes with ratio ~2
sizes: [8, 16, 32, 64]

# =============================================================================
# Temperature Sweep
# =============================================================================

temperatures:
  start: 2.0      # Starting temperature
  end: 2.5        # Ending temperature
  steps: 25       # Number of temperature points

# For 2D Ising, Tc = 2/ln(1+sqrt(2)) ≈ 2.269
# Ensure your range brackets Tc for critical analysis

# =============================================================================
# Simulation Parameters
# =============================================================================

# Monte Carlo steps per temperature point
n_steps: 100000

# Equilibration steps (discarded before measurement)
equilibration: 10000

# Update algorithm: metropolis, wolff, swendsen_wang
algorithm: wolff

# Random seed (null for random, integer for reproducibility)
seed: null

# =============================================================================
# Analysis Parameters
# =============================================================================

# Bootstrap samples for error estimation
bootstrap_samples: 1000

# Method for Tc estimation: binder_crossing, susceptibility_peak
tc_method: binder_crossing

# Method for correlation length: second_moment, exponential_fit
xi_method: second_moment

# =============================================================================
# Output Configuration
# =============================================================================

# Directory for simulation results
results_dir: results

# Directory for figures
figures_dir: figures

# Plot style: publication, presentation
plot_style: publication

# Animation settings
animation_fps: 10
animation_duration: 5
```

### Parameter Guidelines

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `sizes` | System sizes L | [8, 16, 32, 64] for quick runs; [16, 32, 64, 128] for publication |
| `n_steps` | MC steps | 10000 (test), 100000 (standard), 1000000 (high precision) |
| `equilibration` | Warmup steps | 10-20% of n_steps |
| `algorithm` | Update method | `wolff` near Tc, `metropolis` away from Tc |
| `bootstrap_samples` | Error estimation | 1000 (standard), 10000 (high precision) |

## Usage

### Basic Commands

```bash
# Navigate to workflow directory
cd workflow

# Dry run - see what would be executed
snakemake -n

# Run with 4 cores
snakemake --cores 4

# Run with all available cores
snakemake --cores all

# Run specific target
snakemake --cores 4 results/observables.csv
snakemake --cores 4 figures/ising2d_phase_diagram.pdf

# Force re-run of a rule
snakemake --cores 4 --forcerun calculate_observables

# Clean all outputs
snakemake --delete-all-output

# Clean and rerun
snakemake --delete-all-output && snakemake --cores 4
```

### Available Targets

```bash
# Default: all main outputs
snakemake --cores 4

# Run all simulations only
snakemake --cores 4 simulations

# Generate all figures only
snakemake --cores 4 all_figures

# Critical analysis only
snakemake --cores 4 critical_analysis

# Generate HTML report
snakemake --cores 4 results/report.html
```

### Cluster Execution

For HPC clusters with SLURM:

```bash
# Using Snakemake's SLURM executor
snakemake --executor slurm --jobs 100

# With custom profile
snakemake --profile slurm_profile/
```

Example SLURM profile (`slurm_profile/config.yaml`):

```yaml
executor: slurm
jobs: 100
default-resources:
  mem_mb: 4000
  time: "01:00:00"
  partition: "standard"
```

### Visualization

```bash
# Generate workflow DAG (requires graphviz)
snakemake --dag | dot -Tpng > dag.png
snakemake --dag | dot -Tpdf > dag.pdf

# Generate rule graph (simplified)
snakemake --rulegraph | dot -Tpng > rulegraph.png

# Generate file graph
snakemake --filegraph | dot -Tpng > filegraph.png
```

## Workflow DAG

The workflow has the following dependency structure:

```
                    ┌─────────────────┐
                    │    simulate     │
                    │  (per L, T)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐ ┌─────────┐ ┌─────────────────┐
    │ calculate_obs   │ │ compute │ │   compute_xi    │
    │   (per L, T)    │ │ binder  │ │   (per L, T)    │
    └────────┬────────┘ └────┬────┘ └────────┬────────┘
             │               │               │
             ▼               │               │
    ┌─────────────────┐      │               │
    │ aggregate_sweep │      │               │
    │    (per L)      │      │               │
    └────────┬────────┘      │               │
             │               │               │
    ┌────────┴───────────────┴───────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              find_critical_temperature              │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  extract_exponents  │
              └──────────┬──────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  phase   │    │    binder    │    │   scaling    │
│ diagram  │    │   crossing   │    │   collapse   │
└──────────┘    └──────────────┘    └──────────────┘
```

To generate the actual DAG:

```bash
snakemake --dag | dot -Tpng > workflow_dag.png
```

## Output Files

### Results Directory

| File | Description |
|------|-------------|
| `{model}_L{size}_T{temp}.npz` | Raw simulation data |
| `analysis/{model}_L{size}_T{temp}_obs.csv` | Per-point observables |
| `{model}_L{size}_sweep.csv` | Temperature sweep for one size |
| `{model}_critical_temp.json` | Critical temperature estimate |
| `{model}_exponents.json` | Critical exponents |
| `{model}_binder_all.csv` | Aggregated Binder data |
| `observables.csv` | All observables combined |
| `report.html` | Summary report |

### Figures Directory

| File | Description |
|------|-------------|
| `{model}_phase_diagram.pdf` | 2x2 phase diagram |
| `{model}_binder_crossing.pdf` | Binder cumulant vs T |
| `{model}_scaling_collapse.pdf` | FSS data collapse |
| `{model}_susceptibility.pdf` | Susceptibility vs T |
| `{model}_specific_heat.pdf` | Specific heat vs T |
| `{model}_snapshots.pdf` | Spin configurations |
| `animations/{model}_L{size}_T{temp}.gif` | Spin evolution |

## Adding Custom Analysis

### Adding a New Observable

1. Create a script in `scripts/`:

```python
# scripts/compute_my_observable.py
"""Compute custom observable."""

import sys
from pathlib import Path
import numpy as np

def log(message):
    print(message, file=sys.stderr)

def main():
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]

    log(f"Computing custom observable for {input_file}")

    # Load data
    data = dict(np.load(input_file, allow_pickle=True))

    # Your calculation here
    result = compute_something(data)

    # Save result
    np.savez(output_file, result=result)
    log(f"Saved to {output_file}")

if __name__ == '__main__':
    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]
    snakemake = MockSnakemake()
    main()
```

2. Add a rule to `rules/analyze.smk`:

```python
rule compute_my_observable:
    input:
        f"{RESULTS_DIR}/{{model}}_L{{size}}_T{{temp}}.npz"
    output:
        f"{RESULTS_DIR}/analysis/{{model}}_L{{size}}_T{{temp}}_myobs.npz"
    script:
        "../scripts/compute_my_observable.py"
```

### Adding a New Plot

1. Create a plotting script in `scripts/`:

```python
# scripts/plot_my_figure.py
"""Generate custom figure."""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def log(message):
    print(message, file=sys.stderr)

def main():
    input_files = snakemake.input
    output_file = snakemake.output[0]

    log(f"Generating figure from {len(input_files)} files")

    # Load and plot data
    fig, ax = plt.subplots()
    for f in input_files:
        df = pd.read_csv(f)
        ax.plot(df['x'], df['y'])

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    log(f"Figure saved to {output_file}")

if __name__ == '__main__':
    class MockSnakemake:
        def __init__(self):
            self.input = sys.argv[2:]
            self.output = [sys.argv[1]]
    snakemake = MockSnakemake()
    main()
```

2. Add a rule to `rules/figures.smk`:

```python
rule plot_my_figure:
    input:
        expand(f"{RESULTS_DIR}/{{{{model}}}}_L{{size}}_sweep.csv", size=SIZES)
    output:
        f"{FIGURES_DIR}/{{model}}_my_figure.pdf"
    script:
        "../scripts/plot_my_figure.py"
```

### Creating a New Rule File

1. Create `rules/my_rules.smk`:

```python
"""Custom rules for extended analysis."""

rule my_custom_rule:
    input:
        ...
    output:
        ...
    script:
        "../scripts/my_script.py"
```

2. Include in main `Snakefile`:

```python
include: "rules/my_rules.smk"
```

## Troubleshooting

### Common Issues

**Simulation fails with memory error:**
```bash
# Reduce system size or use cluster resources
snakemake --cores 4 --resources mem_mb=8000
```

**Lock file exists:**
```bash
# Remove stale locks
snakemake --unlock
```

**Script import error:**
```bash
# Ensure ising_toolkit is installed
pip install -e .
```

**Missing dependencies:**
```bash
# Install workflow dependencies
pip install snakemake pandas numpy scipy matplotlib h5py
```

### Debugging

```bash
# Print shell commands without executing
snakemake -n -p

# Print reason for each job
snakemake -n --reason

# Run with verbose output
snakemake --cores 4 --verbose

# Keep incomplete outputs for debugging
snakemake --cores 4 --keep-incomplete
```

## Performance Tips

1. **Use cluster algorithms near Tc**: The Wolff algorithm has much smaller autocorrelation times near the critical point.

2. **Parallel execution**: Run with `--cores all` to maximize parallelism.

3. **Incremental runs**: Snakemake only re-runs rules with changed inputs. Modify config and re-run to add more temperatures.

4. **Checkpointing**: For long simulations, implement checkpointing in the simulation code.

## References

- [Snakemake Documentation](https://snakemake.readthedocs.io/)
- [Ising Model Wikipedia](https://en.wikipedia.org/wiki/Ising_model)
- Newman & Barkema, "Monte Carlo Methods in Statistical Physics"
- Landau & Binder, "A Guide to Monte Carlo Simulations in Statistical Physics"
