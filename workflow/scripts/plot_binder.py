"""
Plot Binder cumulant crossing to estimate critical temperature.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """Generate Binder crossing plot."""
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    model = snakemake.params.get('model', 'ising2d')

    # Load data
    df = pd.read_csv(input_file)
    sizes = sorted(df['size'].unique())

    # Apply publication style
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (8, 6),
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
    })

    # Create figure
    fig, ax = plt.subplots()

    # Color map
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sizes)))

    # Plot Binder cumulant for each size
    for i, size in enumerate(sizes):
        size_data = df[df['size'] == size].sort_values('temperature')
        temps = size_data['temperature']
        binder = size_data['binder']

        if 'binder_error' in df.columns:
            errors = size_data['binder_error']
            ax.errorbar(temps, binder, yerr=errors, fmt='o-',
                       color=colors[i], label=f'L={size}',
                       markersize=4, capsize=2)
        else:
            ax.plot(temps, binder, 'o-', color=colors[i],
                   label=f'L={size}', markersize=4)

    # Add theoretical value at Tc
    if model == 'ising2d':
        Tc = 2.269185
        U_star = 0.6106  # Universal value for 2D Ising
        ax.axvline(Tc, color='red', linestyle='--', alpha=0.7,
                   label=f'Tc = {Tc:.3f}')
        ax.axhline(U_star, color='gray', linestyle=':', alpha=0.7,
                   label=f'U* = {U_star:.4f}')

    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Binder Cumulant (U)')
    ax.set_title('Binder Cumulant Crossing')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Binder crossing plot saved to {output_file}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]
            self.params = {'model': 'ising2d'}

    snakemake = MockSnakemake()
    main()
