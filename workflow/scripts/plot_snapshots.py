"""
Generate spin configuration snapshot figure.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    """Generate snapshot figure."""
    input_files = snakemake.input
    output_file = snakemake.output[0]

    # Load spin configurations
    configs = []
    for filepath in input_files:
        data = np.load(filepath, allow_pickle=True)
        if 'spins' in data:
            configs.append({
                'spins': data['spins'],
                'temperature': float(data.get('temperature', 0)),
                'size': int(data.get('size', 0))
            })

    if not configs:
        print("No spin configurations found")
        return

    # Sort by temperature
    configs.sort(key=lambda x: x['temperature'])

    # Create figure
    n_configs = len(configs)
    fig, axes = plt.subplots(1, n_configs, figsize=(4*n_configs, 4))

    if n_configs == 1:
        axes = [axes]

    for ax, config in zip(axes, configs):
        spins = config['spins']
        T = config['temperature']

        im = ax.imshow(spins, cmap='binary', interpolation='nearest',
                       vmin=-1, vmax=1)
        ax.set_title(f'T = {T:.3f}')
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['↓', '↑'])

    plt.suptitle(f'Spin Configurations (L={configs[0]["size"]})', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Snapshots saved to {output_file}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = sys.argv[2:]
            self.output = [sys.argv[1]]

    snakemake = MockSnakemake()
    main()
