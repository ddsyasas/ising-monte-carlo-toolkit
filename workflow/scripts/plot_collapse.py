"""
Generate finite-size scaling collapse plot.

The scaling collapse uses:
- x-axis: (T - Tc) * L^(1/nu)
- y-axis: M * L^(beta/nu) or chi * L^(-gamma/nu)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_style(style='publication'):
    """Configure matplotlib style."""
    if style == 'publication':
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 11,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.figsize': (8, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })
    else:
        plt.style.use('default')


def main():
    """Generate scaling collapse plot."""
    sweep_files = snakemake.input.sweeps
    exponents_file = snakemake.input.exponents
    output_file = snakemake.output[0]
    sizes = snakemake.params.get('sizes', [])
    style = snakemake.params.get('style', 'publication')

    setup_style(style)

    # Load exponents
    with open(exponents_file) as f:
        exp_data = json.load(f)

    Tc = exp_data.get('Tc', 2.269)
    nu = exp_data.get('nu', 1.0)
    beta = exp_data.get('beta', 0.125)
    gamma = exp_data.get('gamma', 1.75)

    # Load sweep data
    data_by_size = {}
    for filepath in sweep_files:
        try:
            df = pd.read_csv(filepath)
            size = int(df['size'].iloc[0])
            data_by_size[size] = df
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    if not data_by_size:
        raise ValueError("No valid data files")

    available_sizes = sorted(data_by_size.keys())

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color map
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(available_sizes)))

    # Left plot: Magnetization collapse
    ax1 = axes[0]
    for i, L in enumerate(available_sizes):
        df = data_by_size[L].sort_values('temperature')
        T = df['temperature'].values

        if 'abs_magnetization_mean' in df.columns:
            M = df['abs_magnetization_mean'].values
        elif 'magnetization_mean' in df.columns:
            M = np.abs(df['magnetization_mean'].values)
        else:
            continue

        # Scaled variables
        x_scaled = (T - Tc) * L ** (1 / nu)
        y_scaled = M * L ** (beta / nu)

        ax1.plot(x_scaled, y_scaled, 'o-', color=colors[i],
                 label=f'L={L}', markersize=4, linewidth=1)

    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_xlabel(r'$(T - T_c) L^{1/\nu}$')
    ax1.set_ylabel(r'$M L^{\beta/\nu}$')
    ax1.set_title('Magnetization Scaling Collapse')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Right plot: Susceptibility collapse
    ax2 = axes[1]
    for i, L in enumerate(available_sizes):
        df = data_by_size[L].sort_values('temperature')
        T = df['temperature'].values

        if 'susceptibility' not in df.columns:
            continue

        chi = df['susceptibility'].values

        # Scaled variables
        x_scaled = (T - Tc) * L ** (1 / nu)
        y_scaled = chi * L ** (-gamma / nu)

        ax2.plot(x_scaled, y_scaled, 'o-', color=colors[i],
                 label=f'L={L}', markersize=4, linewidth=1)

    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.set_xlabel(r'$(T - T_c) L^{1/\nu}$')
    ax2.set_ylabel(r'$\chi L^{-\gamma/\nu}$')
    ax2.set_title('Susceptibility Scaling Collapse')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Add exponent info
    info_text = (f"$T_c = {Tc:.4f}$\n"
                 f"$\\nu = {nu:.3f}$\n"
                 f"$\\beta = {beta:.3f}$\n"
                 f"$\\gamma = {gamma:.3f}$")
    fig.text(0.02, 0.02, info_text, fontsize=8,
             verticalalignment='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close()

    print(f"Scaling collapse plot saved to {output_file}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        class Input:
            def __init__(self, args):
                self.sweeps = args[:-1]
                self.exponents = args[-1]

        def __init__(self):
            self.input = self.Input(sys.argv[2:])
            self.output = [sys.argv[1]]
            self.params = {'sizes': [], 'style': 'publication'}

    snakemake = MockSnakemake()
    main()
