"""
Generate phase diagram plots from simulation results.

Creates a 2x2 figure with:
- Energy vs Temperature
- Magnetization vs Temperature
- Specific Heat vs Temperature
- Susceptibility vs Temperature
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def log(message):
    """Log message to stderr."""
    print(message, file=sys.stderr)


def apply_publication_style():
    """Apply publication-quality plot settings."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.figsize': (10, 8),
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.dpi': 300,
    })


def load_sweep_data(input_files):
    """Load sweep data from multiple files.

    Parameters
    ----------
    input_files : list
        List of CSV file paths

    Returns
    -------
    dict
        Dictionary mapping size to DataFrame
    """
    data_by_size = {}

    for filepath in input_files:
        try:
            df = pd.read_csv(filepath)
            if len(df) == 0:
                log(f"  WARNING: Empty file {Path(filepath).name}")
                continue

            size = int(df['size'].iloc[0])
            data_by_size[size] = df
            log(f"  Loaded L={size}: {len(df)} points")
        except Exception as e:
            log(f"  ERROR loading {Path(filepath).name}: {e}")

    return data_by_size


def main():
    """Generate phase diagram figure."""
    input_files = snakemake.input
    output_file = snakemake.output[0]
    style = snakemake.params.get('style', 'publication')
    sizes = snakemake.params.get('sizes', [])

    log(f"Generating phase diagram from {len(input_files)} files...")

    # Apply style
    if style == 'publication':
        apply_publication_style()

    # Load data
    data_by_size = load_sweep_data(input_files)

    if not data_by_size:
        raise ValueError("No valid data files loaded")

    available_sizes = sorted(data_by_size.keys())
    log(f"Available sizes: {available_sizes}")

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Color map for different sizes
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(available_sizes)))

    # 2D Ising critical temperature
    Tc_exact = 2 / np.log(1 + np.sqrt(2))  # 2.269185...

    # Observable configurations: (column_name, ylabel, error_column, ax)
    observables = [
        ('energy_mean', 'Energy per spin $E/N$', 'energy_err', axes[0, 0]),
        ('abs_magnetization_mean', 'Magnetization $|M|/N$', 'abs_magnetization_err', axes[0, 1]),
        ('specific_heat', 'Specific Heat $C$', 'specific_heat_err', axes[1, 0]),
        ('susceptibility', 'Susceptibility $\\chi$', 'susceptibility_err', axes[1, 1]),
    ]

    for col, ylabel, err_col, ax in observables:
        has_data = False

        for i, size in enumerate(available_sizes):
            df = data_by_size[size].sort_values('temperature')
            temps = df['temperature'].values

            # Check if column exists
            if col not in df.columns:
                # Try alternative column names
                alt_col = col.replace('_mean', '')
                if alt_col in df.columns:
                    values = df[alt_col].values
                else:
                    continue
            else:
                values = df[col].values

            has_data = True

            # Plot with error bars if available
            if err_col in df.columns:
                errors = df[err_col].values
                ax.errorbar(temps, values, yerr=errors, fmt='o-',
                            color=colors[i], label=f'L={size}',
                            markersize=3, capsize=2, capthick=0.5,
                            linewidth=1)
            else:
                ax.plot(temps, values, 'o-', color=colors[i],
                        label=f'L={size}', markersize=3, linewidth=1)

        if has_data:
            ax.set_xlabel('Temperature $T$')
            ax.set_ylabel(ylabel)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)

            # Add critical temperature line
            xlim = ax.get_xlim()
            if xlim[0] < Tc_exact < xlim[1]:
                ax.axvline(Tc_exact, color='red', linestyle='--',
                           alpha=0.5, linewidth=1, label='$T_c$')
        else:
            ax.text(0.5, 0.5, f'No data for\n{col}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Temperature $T$')
            ax.set_ylabel(ylabel)

    # Add overall title
    model = 'Ising 2D'
    if data_by_size:
        first_df = list(data_by_size.values())[0]
        if 'model' in first_df.columns:
            model = first_df['model'].iloc[0]

    fig.suptitle(f'Phase Diagram: {model}', fontsize=14, y=1.02)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        log(f"Phase diagram saved to {output_file}")
    except Exception as e:
        log(f"ERROR saving figure: {e}")
        raise


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python plot_phase_diagram.py output.pdf input1.csv input2.csv ...", file=sys.stderr)
        sys.exit(1)

    class MockSnakemake:
        def __init__(self):
            self.input = sys.argv[2:]
            self.output = [sys.argv[1]]
            self.params = {'style': 'publication', 'sizes': []}

    snakemake = MockSnakemake()
    main()
