"""
Plot magnetic susceptibility vs temperature for multiple system sizes.
"""

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
    """Generate susceptibility plot."""
    input_files = snakemake.input
    output_file = snakemake.output[0]
    sizes = snakemake.params.get('sizes', [])
    style = snakemake.params.get('style', 'publication')

    setup_style(style)

    # Load data for each size
    data_by_size = {}
    for filepath in input_files:
        try:
            df = pd.read_csv(filepath)
            size = int(df['size'].iloc[0])
            data_by_size[size] = df
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    if not data_by_size:
        raise ValueError("No valid data files")

    available_sizes = sorted(data_by_size.keys())

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color map
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(available_sizes)))

    # 2D Ising critical temperature
    Tc_exact = 2 / np.log(1 + np.sqrt(2))

    # Plot susceptibility for each size
    for i, L in enumerate(available_sizes):
        df = data_by_size[L].sort_values('temperature')
        T = df['temperature'].values

        if 'susceptibility' not in df.columns:
            print(f"Warning: No susceptibility data for L={L}")
            continue

        chi = df['susceptibility'].values

        # Plot with error bars if available
        if 'susceptibility_err' in df.columns:
            chi_err = df['susceptibility_err'].values
            ax.errorbar(T, chi, yerr=chi_err, fmt='o-', color=colors[i],
                        label=f'L={L}', markersize=4, linewidth=1,
                        capsize=2, elinewidth=0.8)
        else:
            ax.plot(T, chi, 'o-', color=colors[i], label=f'L={L}',
                    markersize=4, linewidth=1)

    # Mark critical temperature
    ax.axvline(Tc_exact, color='red', linestyle='--', alpha=0.7,
               label=f'$T_c$ (exact) = {Tc_exact:.4f}')

    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'Susceptibility $\chi$')
    ax.set_title('Magnetic Susceptibility vs Temperature')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Log scale for y-axis to see peak scaling
    if any(data_by_size[L]['susceptibility'].max() > 100 for L in available_sizes
           if 'susceptibility' in data_by_size[L].columns):
        ax.set_yscale('log')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close()

    print(f"Susceptibility plot saved to {output_file}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = sys.argv[2:]
            self.output = [sys.argv[1]]
            self.params = {'sizes': [], 'style': 'publication'}

    snakemake = MockSnakemake()
    main()
