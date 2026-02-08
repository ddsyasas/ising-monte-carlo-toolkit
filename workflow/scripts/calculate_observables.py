"""
Calculate observables with bootstrap error estimates for a single simulation.

This script computes:
- Energy per spin with error
- Magnetization per spin with error
- Heat capacity with error
- Magnetic susceptibility with error
- Binder cumulant with error
- Autocorrelation time estimates
"""

import csv
import sys
from pathlib import Path

import numpy as np


def log(message):
    """Log message to stderr."""
    print(message, file=sys.stderr)


def load_data(filepath):
    """Load simulation data from file.

    Parameters
    ----------
    filepath : str
        Path to data file (NPZ or HDF5)

    Returns
    -------
    dict
        Simulation data

    Raises
    ------
    IOError
        If file cannot be loaded
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npz':
        try:
            data = dict(np.load(filepath, allow_pickle=True))
            # Convert 0-d arrays to scalars
            for key in list(data.keys()):
                if isinstance(data[key], np.ndarray) and data[key].ndim == 0:
                    data[key] = data[key].item()
            return data
        except Exception as e:
            raise IOError(f"Failed to load NPZ file: {e}")

    elif filepath.suffix in ['.h5', '.hdf5']:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 files")

        try:
            with h5py.File(filepath, 'r') as f:
                data = {}
                for key in f.attrs:
                    data[key] = f.attrs[key]
                for key in f.keys():
                    data[key] = f[key][:]
                return data
        except Exception as e:
            raise IOError(f"Failed to load HDF5 file: {e}")

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


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

    rng = np.random.default_rng()
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_stats[i] = statistic(data[indices])

    return float(np.std(bootstrap_stats))


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

    return max(float(tau_int), 0.5)


def main():
    """Calculate observables for a single simulation."""
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    n_bootstrap = getattr(snakemake.params, 'bootstrap', 1000)

    log(f"Processing {Path(input_file).name}...")

    # Load data
    try:
        data = load_data(input_file)
    except Exception as e:
        log(f"ERROR loading data: {e}")
        raise

    # Extract metadata
    model = str(data.get('model', 'unknown'))
    size = int(data.get('size', 0))
    temperature = float(data.get('temperature', 0))
    algorithm = str(data.get('algorithm', 'unknown'))
    steps = int(data.get('steps', data.get('n_steps', 0)))

    log(f"  Model: {model}, L={size}, T={temperature:.4f}")

    # Get dimension for normalization
    if '1d' in model.lower():
        n_spins = size
    elif '3d' in model.lower():
        n_spins = size ** 3
    else:
        n_spins = size ** 2  # Default to 2D

    T = temperature if temperature > 0 else 1.0

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
        n_samples = len(energies)
        log(f"  Energy samples: {n_samples}")

        if n_samples > 0:
            results['energy_mean'] = float(np.mean(energies))
            results['energy_std'] = float(np.std(energies))
            results['energy_err'] = float(bootstrap_error(energies, np.mean, n_bootstrap))
            results['energy_tau'] = compute_autocorrelation_time(energies)

            # Specific heat: C = Var(E) / (T^2)
            var_E = np.var(energies)
            results['specific_heat'] = float(var_E / (T ** 2))

            # Bootstrap error for specific heat
            def specific_heat_stat(e):
                return np.var(e) / (T ** 2)
            results['specific_heat_err'] = float(bootstrap_error(energies, specific_heat_stat, n_bootstrap))
        else:
            log("  WARNING: Empty energies array")
    else:
        log("  WARNING: No energies found in data")

    # Magnetization analysis
    if 'magnetizations' in data:
        mags = np.asarray(data['magnetizations'])
        n_samples = len(mags)
        log(f"  Magnetization samples: {n_samples}")

        if n_samples > 0:
            results['magnetization_mean'] = float(np.mean(mags))
            results['magnetization_std'] = float(np.std(mags))
            results['magnetization_err'] = float(bootstrap_error(mags, np.mean, n_bootstrap))
            results['magnetization_tau'] = compute_autocorrelation_time(mags)

            # Absolute magnetization
            abs_mags = np.abs(mags)
            results['abs_magnetization_mean'] = float(np.mean(abs_mags))
            results['abs_magnetization_err'] = float(bootstrap_error(abs_mags, np.mean, n_bootstrap))

            # Susceptibility: chi = N * Var(M) / T
            var_M = np.var(mags)
            results['susceptibility'] = float(n_spins * var_M / T)

            def chi_stat(m):
                return n_spins * np.var(m) / T
            results['susceptibility_err'] = float(bootstrap_error(mags, chi_stat, n_bootstrap))

            # Binder cumulant: U = 1 - <m^4>/(3<m^2>^2)
            m2_mean = np.mean(mags ** 2)
            m4_mean = np.mean(mags ** 4)

            if m2_mean > 1e-10:
                results['binder'] = float(1 - m4_mean / (3 * m2_mean ** 2))
            else:
                results['binder'] = 0.0

            # Bootstrap error for Binder
            def binder_stat(m):
                m2_s = np.mean(m ** 2)
                m4_s = np.mean(m ** 4)
                if m2_s > 1e-10:
                    return 1 - m4_s / (3 * m2_s ** 2)
                return 0.0
            results['binder_err'] = float(bootstrap_error(mags, binder_stat, n_bootstrap))
        else:
            log("  WARNING: Empty magnetizations array")
    else:
        log("  WARNING: No magnetizations found in data")

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write results
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
        log(f"  Saved to {output_file}")
    except Exception as e:
        log(f"ERROR writing output: {e}")
        raise


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python calculate_observables.py input.npz output.csv", file=sys.stderr)
        sys.exit(1)

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]

            class Params:
                bootstrap = 1000
            self.params = Params()

    snakemake = MockSnakemake()
    main()
