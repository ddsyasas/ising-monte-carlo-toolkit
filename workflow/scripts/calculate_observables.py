"""
Calculate observables with bootstrap error estimates for a single simulation.

This script computes:
- Energy per spin with error
- Magnetization per spin with error
- Heat capacity with error
- Magnetic susceptibility with error
- Autocorrelation time estimates
"""

import csv
from pathlib import Path

import numpy as np


def bootstrap_error(data, statistic=np.mean, n_bootstrap=1000):
    """Compute bootstrap error for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    statistic : callable
        Function to compute statistic (default: np.mean)
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    float
        Bootstrap standard error
    """
    n = len(data)
    if n == 0:
        return np.nan

    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        bootstrap_stats[i] = statistic(data[indices])

    return np.std(bootstrap_stats)


def compute_autocorrelation_time(data, max_lag=None):
    """Estimate integrated autocorrelation time using Sokal's method.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    max_lag : int, optional
        Maximum lag to consider

    Returns
    -------
    float
        Integrated autocorrelation time
    """
    n = len(data)
    if n < 10:
        return 1.0

    if max_lag is None:
        max_lag = min(n // 4, 1000)

    mean = np.mean(data)
    var = np.var(data)

    if var < 1e-10:
        return 1.0

    centered = data - mean
    tau_int = 0.5

    for t in range(1, max_lag):
        autocorr = np.mean(centered[:n-t] * centered[t:]) / var
        tau_int += autocorr

        # Sokal's automatic windowing
        if t >= 6 * tau_int:
            break

    return max(tau_int, 0.5)


def main():
    """Calculate observables for a single simulation."""
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    n_bootstrap = snakemake.params.bootstrap

    # Load data
    data = dict(np.load(input_file, allow_pickle=True))

    # Extract metadata
    model = str(data.get('model', ''))
    size = int(data.get('size', 0))
    temperature = float(data.get('temperature', 0))
    algorithm = str(data.get('algorithm', ''))
    steps = int(data.get('steps', 0))

    # Get dimension for normalization
    if '1d' in model.lower():
        n_spins = size
    elif '2d' in model.lower():
        n_spins = size ** 2
    elif '3d' in model.lower():
        n_spins = size ** 3
    else:
        n_spins = size ** 2  # Default to 2D

    results = {
        'model': model,
        'size': size,
        'temperature': temperature,
        'algorithm': algorithm,
        'steps': steps,
        'n_spins': n_spins,
    }

    # Energy analysis
    if 'energies' in data:
        energies = np.asarray(data['energies'])
        results['energy_mean'] = np.mean(energies)
        results['energy_std'] = np.std(energies)
        results['energy_err'] = bootstrap_error(energies, np.mean, n_bootstrap)
        results['energy_tau'] = compute_autocorrelation_time(energies)

        # Heat capacity: C/N = (1/T^2) * Var(E) * N
        var_E = np.var(energies)
        results['heat_capacity'] = var_E * n_spins / (temperature ** 2)

        # Bootstrap error for heat capacity
        def heat_cap_stat(e):
            return np.var(e) * n_spins / (temperature ** 2)
        results['heat_capacity_err'] = bootstrap_error(energies, heat_cap_stat, n_bootstrap)

    # Magnetization analysis
    if 'magnetizations' in data:
        mags = np.asarray(data['magnetizations'])

        results['magnetization_mean'] = np.mean(mags)
        results['magnetization_std'] = np.std(mags)
        results['magnetization_err'] = bootstrap_error(mags, np.mean, n_bootstrap)
        results['magnetization_tau'] = compute_autocorrelation_time(mags)

        # Absolute magnetization
        abs_mags = np.abs(mags)
        results['abs_magnetization_mean'] = np.mean(abs_mags)
        results['abs_magnetization_err'] = bootstrap_error(abs_mags, np.mean, n_bootstrap)

        # Susceptibility: chi/N = (1/T) * Var(M) * N
        var_M = np.var(mags)
        results['susceptibility'] = var_M * n_spins / temperature

        def chi_stat(m):
            return np.var(m) * n_spins / temperature
        results['susceptibility_err'] = bootstrap_error(mags, chi_stat, n_bootstrap)

        # Binder cumulant: U = 1 - <m^4>/(3<m^2>^2)
        m2 = mags ** 2
        m4 = mags ** 4
        m2_mean = np.mean(m2)
        m4_mean = np.mean(m4)

        if m2_mean > 1e-10:
            results['binder'] = 1 - m4_mean / (3 * m2_mean ** 2)
        else:
            results['binder'] = 0.0

        # Bootstrap error for Binder
        def binder_stat(m):
            m2_s = np.mean(m ** 2)
            m4_s = np.mean(m ** 4)
            if m2_s > 1e-10:
                return 1 - m4_s / (3 * m2_s ** 2)
            return 0.0
        results['binder_err'] = bootstrap_error(mags, binder_stat, n_bootstrap)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write results
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"Observables saved to {output_file}")


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
