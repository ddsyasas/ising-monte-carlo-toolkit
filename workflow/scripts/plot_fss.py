"""
Generate finite-size scaling collapse plots.

Finite-size scaling predicts that near Tc:
- M(T, L) = L^(-beta/nu) * f_M((T-Tc) * L^(1/nu))
- chi(T, L) = L^(gamma/nu) * f_chi((T-Tc) * L^(1/nu))

A good scaling collapse indicates correct exponents and Tc.
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """Generate finite-size scaling plot."""
    obs_file = snakemake.input[0]
    critical_file = snakemake.input[1]
    output_file = snakemake.output[0]
    model = snakemake.params.get('model', 'ising2d')

    # Load data
    df = pd.read_csv(obs_file)
    with open(critical_file) as f:
        critical = json.load(f)

    sizes = sorted(df['size'].unique())

    # Get critical parameters
    Tc = critical.get('Tc_estimate', 2.269)

    # Use known exponents for 2D Ising
    if model == 'ising2d':
        beta = 0.125
        gamma = 1.75
        nu = 1.0
    else:
        # Use estimated values
        exponents = critical.get('exponents', {})
        beta = exponents.get('beta', 0.326)
        gamma = exponents.get('gamma', 1.237)
        nu = exponents.get('nu', 0.630)

    # Apply publication style
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (10, 4),
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
    })

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Color map
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sizes)))

    # Magnetization scaling collapse
    for i, L in enumerate(sizes):
        size_data = df[df['size'] == L].sort_values('temperature')
        temps = size_data['temperature'].values
        mags = size_data['magnetization_mean'].values

        # Scaling variables
        x = (temps - Tc) * L**(1/nu)
        y = mags * L**(beta/nu)

        ax1.plot(x, y, 'o-', color=colors[i], label=f'L={L}', markersize=3)

    ax1.set_xlabel(r'$(T - T_c) \cdot L^{1/\nu}$')
    ax1.set_ylabel(r'$M \cdot L^{\beta/\nu}$')
    ax1.set_title('Magnetization Scaling')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)

    # Susceptibility scaling collapse
    for i, L in enumerate(sizes):
        size_data = df[df['size'] == L].sort_values('temperature')
        temps = size_data['temperature'].values
        chi = size_data['susceptibility'].values

        # Scaling variables
        x = (temps - Tc) * L**(1/nu)
        y = chi * L**(-gamma/nu)

        ax2.plot(x, y, 'o-', color=colors[i], label=f'L={L}', markersize=3)

    ax2.set_xlabel(r'$(T - T_c) \cdot L^{1/\nu}$')
    ax2.set_ylabel(r'$\chi \cdot L^{-\gamma/\nu}$')
    ax2.set_title('Susceptibility Scaling')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle(f'Finite-Size Scaling Collapse (Tc = {Tc:.4f})', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"FSS plot saved to {output_file}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1], sys.argv[2]]
            self.output = [sys.argv[3]]
            self.params = {'model': 'ising2d'}

    snakemake = MockSnakemake()
    main()
