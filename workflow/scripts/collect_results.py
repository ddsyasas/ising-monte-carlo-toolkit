"""
Collect simulation results into a single CSV file.

This script is called by Snakemake to aggregate individual simulation
results into a summary table for analysis.
"""

import csv
from pathlib import Path

import numpy as np


def load_simulation_result(filepath):
    """Load a single simulation result file.

    Parameters
    ----------
    filepath : Path or str
        Path to the simulation result file (.npz or .h5)

    Returns
    -------
    dict
        Dictionary containing simulation results
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npz':
        data = dict(np.load(filepath, allow_pickle=True))
        # Convert 0-d arrays to scalars
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].ndim == 0:
                data[key] = data[key].item()
        return data

    elif filepath.suffix in ['.h5', '.hdf5']:
        try:
            import h5py
            with h5py.File(filepath, 'r') as f:
                data = {}
                for key in f.attrs:
                    data[key] = f.attrs[key]
                for key in f.keys():
                    data[key] = f[key][:]
                return data
        except ImportError:
            raise ImportError("h5py required for HDF5 files")

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def extract_observables(data):
    """Extract observable statistics from simulation data.

    Parameters
    ----------
    data : dict
        Simulation data dictionary

    Returns
    -------
    dict
        Dictionary with observable means and errors
    """
    result = {}

    # Metadata
    result['model'] = str(data.get('model', ''))
    result['size'] = int(data.get('size', 0))
    result['temperature'] = float(data.get('temperature', 0))
    result['algorithm'] = str(data.get('algorithm', ''))
    result['steps'] = int(data.get('steps', 0))

    # Energy
    if 'energies' in data:
        energies = np.asarray(data['energies'])
        result['energy_mean'] = np.mean(energies)
        result['energy_std'] = np.std(energies)
        result['energy_err'] = np.std(energies) / np.sqrt(len(energies))
    elif 'energy_mean' in data:
        result['energy_mean'] = float(data['energy_mean'])
        result['energy_std'] = float(data.get('energy_std', 0))
        result['energy_err'] = result['energy_std']

    # Magnetization
    if 'magnetizations' in data:
        mags = np.asarray(data['magnetizations'])
        result['magnetization_mean'] = np.mean(mags)
        result['magnetization_std'] = np.std(mags)
        result['magnetization_err'] = np.std(mags) / np.sqrt(len(mags))

        # Binder cumulant: U = 1 - <m^4>/(3<m^2>^2)
        m2 = np.mean(mags**2)
        m4 = np.mean(mags**4)
        if m2 > 1e-10:
            result['binder'] = 1 - m4 / (3 * m2**2)
        else:
            result['binder'] = 0.0

    elif 'magnetization_mean' in data:
        result['magnetization_mean'] = float(data['magnetization_mean'])
        result['magnetization_std'] = float(data.get('magnetization_std', 0))
        result['magnetization_err'] = result['magnetization_std']
        result['binder'] = float(data.get('binder', 0))

    # Heat capacity
    if 'heat_capacity' in data:
        result['heat_capacity'] = float(data['heat_capacity'])
    elif 'energies' in data:
        T = result['temperature']
        N = result['size'] ** 2 if '2d' in result['model'] else result['size']
        var_E = np.var(data['energies'])
        result['heat_capacity'] = var_E * N / (T**2)

    # Susceptibility
    if 'susceptibility' in data:
        result['susceptibility'] = float(data['susceptibility'])
    elif 'magnetizations' in data:
        T = result['temperature']
        N = result['size'] ** 2 if '2d' in result['model'] else result['size']
        var_M = np.var(data['magnetizations'])
        result['susceptibility'] = var_M * N / T

    return result


def main():
    """Main entry point for Snakemake script."""
    # Get input/output from Snakemake
    input_files = snakemake.input
    output_file = snakemake.output[0]

    # Collect all results
    results = []
    for filepath in input_files:
        try:
            data = load_simulation_result(filepath)
            obs = extract_observables(data)
            obs['source_file'] = str(Path(filepath).name)
            results.append(obs)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    if not results:
        raise ValueError("No valid results to collect")

    # Sort by size, then temperature
    results.sort(key=lambda x: (x['size'], x['temperature']))

    # Write CSV
    fieldnames = [
        'model', 'size', 'temperature', 'algorithm', 'steps',
        'energy_mean', 'energy_std', 'energy_err',
        'magnetization_mean', 'magnetization_std', 'magnetization_err',
        'heat_capacity', 'susceptibility', 'binder',
        'source_file'
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Collected {len(results)} results to {output_file}")


if __name__ == '__main__':
    # When run directly (not via Snakemake), use command line args
    import sys
    if len(sys.argv) < 3:
        print("Usage: python collect_results.py output.csv input1.npz input2.npz ...")
        sys.exit(1)

    # Mock snakemake object
    class MockSnakemake:
        def __init__(self):
            self.output = [sys.argv[1]]
            self.input = sys.argv[2:]

    snakemake = MockSnakemake()
    main()
