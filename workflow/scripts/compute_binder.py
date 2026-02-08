"""
Compute Binder cumulant from simulation data.

The Binder cumulant U = 1 - <m^4>/(3<m^2>^2) is useful for
identifying the critical temperature through the crossing point
of curves for different system sizes.
"""

import csv
from pathlib import Path

import numpy as np


def compute_binder_from_file(filepath):
    """Compute Binder cumulant from a simulation file.

    Parameters
    ----------
    filepath : Path or str
        Path to simulation result file

    Returns
    -------
    dict
        Dictionary with size, temperature, binder, and error
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npz':
        data = dict(np.load(filepath, allow_pickle=True))
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Extract metadata
    size = int(data.get('size', 0))
    temperature = float(data.get('temperature', 0))
    model = str(data.get('model', ''))

    # Compute Binder cumulant
    if 'magnetizations' in data:
        mags = np.asarray(data['magnetizations'])

        # Compute moments
        m2 = mags ** 2
        m4 = mags ** 4

        m2_mean = np.mean(m2)
        m4_mean = np.mean(m4)

        # Binder cumulant
        if m2_mean > 1e-10:
            binder = 1 - m4_mean / (3 * m2_mean**2)
        else:
            binder = 0.0

        # Bootstrap error estimation
        n_bootstrap = 1000
        n_samples = len(mags)
        binder_samples = []

        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n_samples, n_samples)
            m2_boot = np.mean(mags[idx]**2)
            m4_boot = np.mean(mags[idx]**4)
            if m2_boot > 1e-10:
                binder_samples.append(1 - m4_boot / (3 * m2_boot**2))

        binder_error = np.std(binder_samples) if binder_samples else 0.0

    else:
        binder = float(data.get('binder', 0))
        binder_error = 0.0

    return {
        'model': model,
        'size': size,
        'temperature': temperature,
        'binder': binder,
        'binder_error': binder_error
    }


def main():
    """Main entry point."""
    input_files = snakemake.input
    output_file = snakemake.output[0]

    results = []
    for filepath in input_files:
        try:
            result = compute_binder_from_file(filepath)
            result['source_file'] = Path(filepath).name
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to process {filepath}: {e}")

    if not results:
        raise ValueError("No valid results")

    # Sort by size, then temperature
    results.sort(key=lambda x: (x['size'], x['temperature']))

    # Write CSV
    fieldnames = ['model', 'size', 'temperature', 'binder', 'binder_error', 'source_file']

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Binder cumulant data saved to {output_file}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = sys.argv[2:]
            self.output = [sys.argv[1]]

    snakemake = MockSnakemake()
    main()
