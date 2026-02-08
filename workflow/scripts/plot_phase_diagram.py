"""
Generate phase diagram plots from simulation results.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_publication_style():
    """Apply publication-quality plot settings."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (10, 8),
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def main():
    """Generate phase diagram figure."""
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    style = snakemake.params.get('style', 'publication')

    # Apply style
    if style == 'publication':
        apply_publication_style()

    # Load data
    df = pd.read_csv(input_file)
    sizes = sorted(df['size'].unique())

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Color map for different sizes
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sizes)))

    # Observable configurations
    observables = [
        ('energy_mean', 'Energy per spin (E/N)', axes[0, 0]),
        ('magnetization_mean', '|M|/N', axes[0, 1]),
        ('heat_capacity', 'Heat capacity (C)', axes[1, 0]),
        ('susceptibility', 'Susceptibility (χ)', axes[1, 1]),
    ]

    for col, ylabel, ax in observables:
        for i, size in enumerate(sizes):
            size_data = df[df['size'] == size].sort_values('temperature')
            temps = size_data['temperature']
            values = size_data[col]

            # Plot with error bars if available
            err_col = col.replace('_mean', '_err')
            if err_col in df.columns:
                errors = size_data[err_col]
                ax.errorbar(temps, values, yerr=errors, fmt='o-',
                           color=colors[i], label=f'L={size}',
                           markersize=3, capsize=2, capthick=0.5)
            else:
                ax.plot(temps, values, 'o-', color=colors[i],
                       label=f'L={size}', markersize=3)

        ax.set_xlabel('Temperature (T)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # Add critical temperature line (for 2D Ising)
    Tc_2d = 2.269185
    for ax in axes.flat:
        xlim = ax.get_xlim()
        if xlim[0] < Tc_2d < xlim[1]:
            ax.axvline(Tc_2d, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.suptitle('Phase Diagram: 2D Ising Model', fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Phase diagram saved to {output_file}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]
            self.params = {'style': 'publication'}

    snakemake = MockSnakemake()
    main()
