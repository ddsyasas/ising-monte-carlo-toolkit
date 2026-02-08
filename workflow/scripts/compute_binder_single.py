"""
Compute Binder cumulant for a single simulation with bootstrap error.
"""

import csv
from pathlib import Path

import numpy as np


def main():
    """Compute Binder cumulant for a single simulation."""
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    n_bootstrap = snakemake.params.bootstrap

    # Load data
    data = dict(np.load(input_file, allow_pickle=True))

    # Extract metadata
    model = str(data.get('model', ''))
    size = int(data.get('size', 0))
    temperature = float(data.get('temperature', 0))

    results = {
        'model': model,
        'size': size,
        'temperature': temperature,
    }

    if 'magnetizations' in data:
        mags = np.asarray(data['magnetizations'])

        # Compute Binder cumulant
        m2 = mags ** 2
        m4 = mags ** 4

        m2_mean = np.mean(m2)
        m4_mean = np.mean(m4)

        if m2_mean > 1e-10:
            binder = 1 - m4_mean / (3 * m2_mean ** 2)
        else:
            binder = 0.0

        results['binder'] = binder
        results['m2_mean'] = m2_mean
        results['m4_mean'] = m4_mean

        # Bootstrap error
        n_samples = len(mags)
        binder_samples = []

        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n_samples, n_samples)
            m2_boot = np.mean(mags[idx] ** 2)
            m4_boot = np.mean(mags[idx] ** 4)
            if m2_boot > 1e-10:
                binder_samples.append(1 - m4_boot / (3 * m2_boot ** 2))

        results['binder_err'] = np.std(binder_samples) if binder_samples else 0.0

    else:
        results['binder'] = 0.0
        results['binder_err'] = 0.0

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write results
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]

            class Params:
                bootstrap = 1000
            self.params = Params()

    snakemake = MockSnakemake()
    main()
