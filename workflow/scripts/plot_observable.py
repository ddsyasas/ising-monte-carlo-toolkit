"""
Plot a single observable vs temperature for a specific size.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    """Generate single observable plot."""
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]

    # Load data
    df = pd.read_csv(input_file)

    # Get size from filename
    size = df['size'].iloc[0]

    # Apply publication style
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'figure.figsize': (6, 4.5),
    })

    fig, ax = plt.subplots()

    temps = df['temperature']
    mags = df['magnetization_mean']

    if 'magnetization_err' in df.columns:
        errors = df['magnetization_err']
        ax.errorbar(temps, mags, yerr=errors, fmt='o-', markersize=4, capsize=2)
    else:
        ax.plot(temps, mags, 'o-', markersize=4)

    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('|M|/N')
    ax.set_title(f'Magnetization (L={size})')
    ax.grid(True, alpha=0.3)

    # Add Tc line
    ax.axvline(2.269, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]

    snakemake = MockSnakemake()
    main()
