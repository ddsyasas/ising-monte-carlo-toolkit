"""
Collect simulation results into a single CSV file.

This script is called by Snakemake to aggregate individual simulation
results into a summary table for analysis.
"""

import sys
from pathlib import Path

import numpy as np


def log(message):
    """Log message to stderr."""
    print(message, file=sys.stderr)


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

    Raises
    ------
    ValueError
        If file format is not supported
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npz':
        try:
            data = dict(np.load(filepath, allow_pickle=True))
            # Convert 0-d arrays to scalars
            for key in data:
                if isinstance(data[key], np.ndarray) and data[key].ndim == 0:
                    data[key] = data[key].item()
            return data
        except Exception as e:
            raise IOError(f"Failed to load NPZ file {filepath}: {e}")

    elif filepath.suffix in ['.h5', '.hdf5']:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 files. Install with: pip install h5py")

        try:
            with h5py.File(filepath, 'r') as f:
                data = {}
                for key in f.attrs:
                    data[key] = f.attrs[key]
                for key in f.keys():
                    data[key] = f[key][:]
                return data
        except Exception as e:
            raise IOError(f"Failed to load HDF5 file {filepath}: {e}")

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def extract_observables(data, filepath):
    """Extract observable statistics from simulation data.

    Parameters
    ----------
    data : dict
        Simulation data dictionary
    filepath : str
        Source file path for error messages

    Returns
    -------
    dict
        Dictionary with observable means and errors
    """
    result = {}

    # Metadata
    result['model'] = str(data.get('model', 'unknown'))
    result['size'] = int(data.get('size', 0))
    result['temperature'] = float(data.get('temperature', 0))
    result['algorithm'] = str(data.get('algorithm', 'unknown'))
    result['steps'] = int(data.get('steps', data.get('n_steps', 0)))

    # Determine number of spins
    if '1d' in result['model'].lower():
        n_spins = result['size']
    elif '3d' in result['model'].lower():
        n_spins = result['size'] ** 3
    else:  # Default to 2D
        n_spins = result['size'] ** 2

    T = result['temperature'] if result['temperature'] > 0 else 1.0

    # Energy
    if 'energies' in data:
        energies = np.asarray(data['energies'])
        if len(energies) > 0:
            result['energy_mean'] = float(np.mean(energies))
            result['energy_std'] = float(np.std(energies))
            result['energy_err'] = float(np.std(energies) / np.sqrt(len(energies)))

            # Specific heat: C = Var(E) / (T^2 * N)
            var_E = np.var(energies)
            result['specific_heat'] = float(var_E / (T ** 2))
        else:
            log(f"  Warning: Empty energies array in {filepath}")
    elif 'energy_mean' in data:
        result['energy_mean'] = float(data['energy_mean'])
        result['energy_std'] = float(data.get('energy_std', 0))
        result['energy_err'] = float(data.get('energy_err', result['energy_std']))

    # Magnetization
    if 'magnetizations' in data:
        mags = np.asarray(data['magnetizations'])
        if len(mags) > 0:
            result['magnetization_mean'] = float(np.mean(mags))
            result['magnetization_std'] = float(np.std(mags))
            result['magnetization_err'] = float(np.std(mags) / np.sqrt(len(mags)))

            # Absolute magnetization
            abs_mags = np.abs(mags)
            result['abs_magnetization_mean'] = float(np.mean(abs_mags))
            result['abs_magnetization_err'] = float(np.std(abs_mags) / np.sqrt(len(abs_mags)))

            # Susceptibility: chi = N * Var(M) / T
            var_M = np.var(mags)
            result['susceptibility'] = float(n_spins * var_M / T)

            # Binder cumulant: U = 1 - <m^4>/(3<m^2>^2)
            m2 = np.mean(mags ** 2)
            m4 = np.mean(mags ** 4)
            if m2 > 1e-10:
                result['binder'] = float(1 - m4 / (3 * m2 ** 2))
            else:
                result['binder'] = 0.0
        else:
            log(f"  Warning: Empty magnetizations array in {filepath}")

    elif 'magnetization_mean' in data:
        result['magnetization_mean'] = float(data['magnetization_mean'])
        result['magnetization_std'] = float(data.get('magnetization_std', 0))
        result['magnetization_err'] = float(data.get('magnetization_err', result['magnetization_std']))
        result['binder'] = float(data.get('binder', 0))

    # Copy pre-computed values if available
    for key in ['heat_capacity', 'susceptibility', 'specific_heat']:
        if key in data and key not in result:
            result[key] = float(data[key])

    return result


def main():
    """Main entry point for Snakemake script."""
    input_files = snakemake.input
    output_file = snakemake.output[0]

    log(f"Collecting results from {len(input_files)} simulation files...")

    # Collect all results
    results = []
    failed_files = []

    for i, filepath in enumerate(input_files):
        try:
            log(f"  [{i+1}/{len(input_files)}] {Path(filepath).name}")
            data = load_simulation_result(filepath)
            obs = extract_observables(data, filepath)
            obs['source_file'] = str(Path(filepath).name)
            results.append(obs)
        except Exception as e:
            log(f"    ERROR: {e}")
            failed_files.append((filepath, str(e)))

    if not results:
        raise ValueError(f"No valid results to collect. {len(failed_files)} files failed.")

    # Sort by size, then temperature
    results.sort(key=lambda x: (x.get('size', 0), x.get('temperature', 0)))

    # Determine all fieldnames from results
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    # Order fieldnames sensibly
    priority_keys = [
        'model', 'size', 'temperature', 'algorithm', 'steps',
        'energy_mean', 'energy_std', 'energy_err',
        'magnetization_mean', 'magnetization_std', 'magnetization_err',
        'abs_magnetization_mean', 'abs_magnetization_err',
        'specific_heat', 'susceptibility', 'binder',
        'source_file'
    ]
    fieldnames = [k for k in priority_keys if k in all_keys]
    fieldnames += sorted([k for k in all_keys if k not in fieldnames])

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    log(f"Successfully collected {len(results)} results to {output_file}")
    if failed_files:
        log(f"WARNING: {len(failed_files)} files failed to process:")
        for fp, err in failed_files[:5]:
            log(f"  - {Path(fp).name}: {err}")
        if len(failed_files) > 5:
            log(f"  ... and {len(failed_files) - 5} more")


if __name__ == '__main__':
    # When run directly (not via Snakemake), use command line args
    if len(sys.argv) < 3:
        print("Usage: python collect_results.py output.csv input1.npz input2.npz ...", file=sys.stderr)
        sys.exit(1)

    # Mock snakemake object
    class MockSnakemake:
        def __init__(self):
            self.output = [sys.argv[1]]
            self.input = sys.argv[2:]

    snakemake = MockSnakemake()
    main()
