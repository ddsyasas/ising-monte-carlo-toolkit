"""Command-line interface for the Ising Monte Carlo toolkit."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import numpy as np

from ising_toolkit import __version__


# =============================================================================
# Helper Functions
# =============================================================================

def load_results_from_path(path: Union[str, Path]) -> Dict[str, Any]:
    """Load simulation results from a file or directory.

    Parameters
    ----------
    path : str or Path
        Path to a single results file (.npz, .csv, .h5) or a directory
        containing multiple result files.

    Returns
    -------
    dict
        Dictionary containing loaded results with keys:
        - 'type': 'single' or 'sweep' or 'multi'
        - 'data': The actual data (dict for single, list of dicts for multi)
        - 'source': Original path

    Raises
    ------
    click.ClickException
        If the path doesn't exist or contains no valid result files.
    """
    path = Path(path)

    if not path.exists():
        raise click.ClickException(f"Path does not exist: {path}")

    if path.is_file():
        # Load single file
        return _load_single_file(path)
    elif path.is_dir():
        # Load all result files from directory
        return _load_directory(path)
    else:
        raise click.ClickException(f"Invalid path: {path}")


def _load_single_file(filepath: Path) -> Dict[str, Any]:
    """Load a single result file."""
    suffix = filepath.suffix.lower()

    if suffix == '.npz':
        data = dict(np.load(filepath, allow_pickle=True))
        # Convert numpy arrays to regular values where appropriate
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].ndim == 0:
                data[key] = data[key].item()
        result_type = 'sweep' if 'temperatures' in data else 'single'
        return {'type': result_type, 'data': data, 'source': filepath}

    elif suffix == '.csv':
        # Try to load as sweep data
        try:
            data = np.genfromtxt(filepath, delimiter=',', names=True)
            result = {name: data[name] for name in data.dtype.names}
            return {'type': 'sweep', 'data': result, 'source': filepath}
        except Exception as e:
            raise click.ClickException(f"Failed to load CSV: {e}")

    elif suffix in ['.h5', '.hdf5']:
        try:
            import h5py
        except ImportError:
            raise click.ClickException("h5py required for HDF5 files")

        with h5py.File(filepath, 'r') as f:
            data = {}
            # Load attributes
            for key in f.attrs:
                data[key] = f.attrs[key]
            # Load datasets
            for key in f.keys():
                data[key] = f[key][:]

        result_type = 'sweep' if 'temperatures' in data else 'single'
        return {'type': result_type, 'data': data, 'source': filepath}

    else:
        raise click.ClickException(f"Unsupported file format: {suffix}")


def _load_directory(dirpath: Path) -> Dict[str, Any]:
    """Load all result files from a directory."""
    # Find all result files
    patterns = ['*.npz', '*.csv', '*.h5', '*.hdf5']
    files = []
    for pattern in patterns:
        files.extend(dirpath.glob(pattern))

    # Filter out summary files
    files = [f for f in files if 'summary' not in f.name.lower()]

    if not files:
        raise click.ClickException(f"No result files found in: {dirpath}")

    # Load all files
    results = []
    for f in sorted(files):
        try:
            result = _load_single_file(f)
            results.append(result)
        except click.ClickException:
            continue  # Skip files that can't be loaded

    if not results:
        raise click.ClickException(f"No valid result files in: {dirpath}")

    return {
        'type': 'multi',
        'data': results,
        'source': dirpath,
        'count': len(results)
    }


def format_results_table(results: Dict[str, Any], observables: List[str]) -> str:
    """Format analysis results as a pretty-printed table.

    Parameters
    ----------
    results : dict
        Analysis results dictionary containing observable statistics.
    observables : list of str
        List of observable names to include in the table.

    Returns
    -------
    str
        Formatted table string ready for printing.
    """
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("Analysis Results")
    lines.append("=" * 70)

    # Source info
    if 'source' in results:
        lines.append(f"Source: {results['source']}")
    if 'model' in results:
        lines.append(f"Model: {results['model']}")
    if 'size' in results:
        lines.append(f"Size: {results['size']}")
    if 'temperature' in results:
        lines.append(f"Temperature: {results['temperature']:.4f}")
    if 'n_samples' in results:
        lines.append(f"Samples: {results['n_samples']}")

    lines.append("-" * 70)

    # Observable statistics
    lines.append(f"{'Observable':<20} {'Mean':>12} {'Std':>12} {'Error':>12}")
    lines.append("-" * 70)

    for obs in observables:
        if obs in results:
            stats = results[obs]
            if isinstance(stats, dict):
                mean = stats.get('mean', float('nan'))
                std = stats.get('std', float('nan'))
                error = stats.get('error', stats.get('bootstrap_error', float('nan')))
                lines.append(f"{obs:<20} {mean:>12.6f} {std:>12.6f} {error:>12.6f}")
            else:
                lines.append(f"{obs:<20} {stats:>12.6f}")

    lines.append("=" * 70)

    return '\n'.join(lines)


def _calculate_bootstrap_error(data: np.ndarray, n_bootstrap: int = 1000,
                                statistic: str = 'mean') -> float:
    """Calculate bootstrap error for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Data array to bootstrap.
    n_bootstrap : int
        Number of bootstrap samples.
    statistic : str
        Statistic to compute ('mean', 'std', 'var').

    Returns
    -------
    float
        Bootstrap standard error.
    """
    n = len(data)
    if n == 0:
        return float('nan')

    stat_func = {
        'mean': np.mean,
        'std': np.std,
        'var': np.var,
    }.get(statistic, np.mean)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        sample = data[indices]
        bootstrap_stats.append(stat_func(sample))

    return np.std(bootstrap_stats)


def _analyze_single_result(data: Dict[str, Any], observables: List[str],
                           n_bootstrap: int = 1000) -> Dict[str, Any]:
    """Analyze a single simulation result.

    Parameters
    ----------
    data : dict
        Result data dictionary.
    observables : list of str
        Observables to analyze.
    n_bootstrap : int
        Number of bootstrap samples for error estimation.

    Returns
    -------
    dict
        Analysis results with statistics for each observable.
    """
    results = {}

    # Copy metadata
    for key in ['model', 'size', 'temperature', 'algorithm', 'steps']:
        if key in data:
            results[key] = data[key]

    # Analyze observables
    observable_keys = {
        'energy': 'energies',
        'magnetization': 'magnetizations',
        'heat_capacity': 'heat_capacities',
        'susceptibility': 'susceptibilities',
    }

    for obs in observables:
        key = observable_keys.get(obs, obs)

        # Check for time series data
        if key in data:
            values = np.asarray(data[key])
            if values.ndim > 0 and len(values) > 1:
                results[obs] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'bootstrap_error': _calculate_bootstrap_error(values, n_bootstrap),
                }
                results['n_samples'] = len(values)
            else:
                results[obs] = float(values)

        # Check for pre-computed mean/std
        elif f'{obs}_mean' in data:
            results[obs] = {
                'mean': float(data[f'{obs}_mean']),
                'std': float(data.get(f'{obs}_std', 0)),
                'error': float(data.get(f'{obs}_std', 0)),
            }

    return results


def _analyze_sweep_result(data: Dict[str, Any], observables: List[str],
                          n_bootstrap: int = 1000) -> Dict[str, Any]:
    """Analyze temperature sweep results.

    Parameters
    ----------
    data : dict
        Sweep result data dictionary.
    observables : list of str
        Observables to analyze.
    n_bootstrap : int
        Number of bootstrap samples for error estimation.

    Returns
    -------
    dict
        Analysis results including peak locations.
    """
    results = {}

    # Copy metadata
    for key in ['model', 'size', 'algorithm', 'steps']:
        if key in data:
            results[key] = data[key]

    temperatures = np.asarray(data.get('temperatures', []))
    results['n_temperatures'] = len(temperatures)
    results['temp_range'] = (float(np.min(temperatures)), float(np.max(temperatures)))

    # Observable analysis
    observable_keys = {
        'energy': 'energies',
        'magnetization': 'magnetizations',
        'heat_capacity': 'heat_capacities',
        'susceptibility': 'susceptibilities',
    }

    for obs in observables:
        key = observable_keys.get(obs, obs)

        if key in data:
            values = np.asarray(data[key])

            results[obs] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

            # Find peak for susceptibility and heat capacity
            if obs in ['susceptibility', 'heat_capacity']:
                idx_peak = np.argmax(values)
                results[obs]['peak_value'] = float(values[idx_peak])
                results[obs]['peak_temperature'] = float(temperatures[idx_peak])

            # Find transition for magnetization
            if obs == 'magnetization' and len(values) > 2:
                # Find steepest descent
                dM_dT = np.gradient(values, temperatures)
                idx_transition = np.argmin(dM_dT)
                results[obs]['transition_temperature'] = float(temperatures[idx_transition])

    return results


@click.group()
@click.version_option(version=__version__, prog_name='ising-sim')
def main():
    """Ising Monte Carlo Toolkit - Simulate magnetic phase transitions.

    Run Monte Carlo simulations of Ising models in 1D, 2D, and 3D.
    Compute thermodynamic observables and analyze phase transitions.

    Examples:

        # Run a 2D simulation at critical temperature
        ising-sim run --model ising2d --size 32 --temperature 2.269

        # Temperature sweep for phase diagram
        ising-sim sweep -m ising2d -L 16 -L 32 --temp-start 1.5 --temp-end 3.5 -o results/

        # Show info about a model
        ising-sim info --model ising2d
    """
    pass


@main.command()
@click.option('--model', '-m',
              type=click.Choice(['ising1d', 'ising2d', 'ising3d']),
              required=True,
              help='Model type')
@click.option('--size', '-L',
              type=int,
              required=True,
              help='Lattice size')
@click.option('--temperature', '-T',
              type=float,
              required=True,
              help='Temperature (in units of J/kB)')
@click.option('--steps', '-n',
              type=int,
              default=100000,
              show_default=True,
              help='Number of MC steps')
@click.option('--equilibration', '-e',
              type=int,
              default=10000,
              show_default=True,
              help='Equilibration steps (discarded)')
@click.option('--algorithm', '-a',
              type=click.Choice(['metropolis', 'wolff']),
              default='metropolis',
              show_default=True,
              help='Sampling algorithm')
@click.option('--seed', '-s',
              type=int,
              default=None,
              help='Random seed for reproducibility')
@click.option('--output', '-o',
              type=click.Path(),
              default=None,
              help='Output file path')
@click.option('--config', '-c',
              type=click.Path(exists=True),
              default=None,
              help='Configuration file (YAML)')
@click.option('--save-spins/--no-save-spins',
              default=False,
              help='Save final spin configuration')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Verbose output')
def run(model, size, temperature, steps, equilibration, algorithm,
        seed, output, config, save_spins, verbose):
    """Run a Monte Carlo simulation.

    Examples:

        # Basic simulation
        ising-sim run --model ising2d --size 32 --temperature 2.269

        # With Wolff algorithm and seed
        ising-sim run -m ising2d -L 64 -T 2.269 -a wolff -s 42

        # Save results to file
        ising-sim run -m ising2d -L 32 -T 2.269 -o results.npz
    """
    # Import here to avoid slow startup
    from ising_toolkit.models import Ising1D, Ising2D, Ising3D
    from ising_toolkit.samplers import MetropolisSampler, WolffSampler

    # Load config file if provided
    if config is not None:
        try:
            from ising_toolkit.io import load_config
            cfg = load_config(config)
            # CLI arguments override config file
            if model is None:
                model = cfg.simulation.model
            if size is None:
                size = cfg.simulation.size
            if temperature is None:
                temperature = cfg.simulation.temperature
            if verbose:
                click.echo(f"Loaded configuration from {config}")
        except Exception as e:
            raise click.ClickException(f"Failed to load config: {e}")

    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        if verbose:
            click.echo(f"Random seed: {seed}")

    # Create model
    model_classes = {
        'ising1d': Ising1D,
        'ising2d': Ising2D,
        'ising3d': Ising3D,
    }

    if verbose:
        click.echo(f"Creating {model} model with L={size}...")

    try:
        model_instance = model_classes[model](size=size, temperature=temperature)
    except Exception as e:
        raise click.ClickException(f"Failed to create model: {e}")

    # Create sampler
    if algorithm == 'metropolis':
        sampler = MetropolisSampler(model_instance)
    else:
        sampler = WolffSampler(model_instance)

    if verbose:
        click.echo(f"Using {algorithm} algorithm")
        click.echo(f"Temperature: {temperature}")
        click.echo(f"Steps: {steps} (equilibration: {equilibration})")

    # Run simulation with progress bar
    click.echo(f"\nRunning {model} simulation at T={temperature:.4f}...")

    with click.progressbar(length=steps, label='Simulating') as bar:
        # Equilibration
        for _ in range(equilibration):
            sampler.step()

        # Production
        energies = []
        magnetizations = []

        for i in range(steps):
            sampler.step()
            if i % 10 == 0:  # Measure every 10 steps
                energies.append(model_instance.get_energy() / model_instance.n_spins)
                magnetizations.append(
                    np.abs(model_instance.get_magnetization()) / model_instance.n_spins
                )
            bar.update(1)

    # Compute statistics
    energies = np.array(energies)
    magnetizations = np.array(magnetizations)

    E_mean = np.mean(energies)
    E_std = np.std(energies)
    M_mean = np.mean(magnetizations)
    M_std = np.std(magnetizations)

    # Heat capacity: C = (1/T^2) * Var(E) * N
    C = (1 / temperature**2) * np.var(energies) * model_instance.n_spins

    # Susceptibility: chi = (1/T) * Var(M) * N
    chi = (1 / temperature) * np.var(magnetizations) * model_instance.n_spins

    # Print results
    click.echo("\n" + "="*50)
    click.echo("Simulation Results")
    click.echo("="*50)
    click.echo(f"Model:            {model}")
    click.echo(f"Size:             {size}")
    click.echo(f"Temperature:      {temperature:.4f}")
    click.echo(f"Algorithm:        {algorithm}")
    click.echo(f"Steps:            {steps}")
    click.echo("-"*50)
    click.echo(f"Energy/spin:      {E_mean:.6f} ± {E_std:.6f}")
    click.echo(f"|Magnetization|:  {M_mean:.6f} ± {M_std:.6f}")
    click.echo(f"Heat capacity:    {C:.4f}")
    click.echo(f"Susceptibility:   {chi:.4f}")
    click.echo("="*50)

    # Save results
    if output is not None:
        output_path = Path(output)

        results = {
            'model': model,
            'size': size,
            'temperature': temperature,
            'algorithm': algorithm,
            'steps': steps,
            'equilibration': equilibration,
            'seed': seed,
            'energy_mean': E_mean,
            'energy_std': E_std,
            'magnetization_mean': M_mean,
            'magnetization_std': M_std,
            'heat_capacity': C,
            'susceptibility': chi,
            'energies': energies,
            'magnetizations': magnetizations,
        }

        if save_spins:
            results['spins'] = model_instance.spins

        if output_path.suffix == '.npz':
            np.savez(output_path, **results)
        elif output_path.suffix == '.npy':
            np.save(output_path, results)
        else:
            # Default to npz
            np.savez(output_path.with_suffix('.npz'), **results)

        click.echo(f"\nResults saved to {output_path}")


def _generate_sweep_filename(model: str, size: int, temp_start: float,
                             temp_end: float, algorithm: str) -> str:
    """Generate default filename for sweep results.

    Parameters
    ----------
    model : str
        Model type (e.g., 'ising2d')
    size : int
        Lattice size
    temp_start : float
        Start temperature
    temp_end : float
        End temperature
    algorithm : str
        Sampling algorithm

    Returns
    -------
    str
        Default filename without extension
    """
    return f"{model}_L{size}_T{temp_start:.2f}-{temp_end:.2f}_{algorithm}"


def _run_single_sweep(args):
    """Run a single temperature sweep for one size (worker function for parallel).

    Parameters
    ----------
    args : tuple
        (model_name, size, temperatures, steps, equilibration, algorithm)

    Returns
    -------
    dict
        Results dictionary with temperatures and observables
    """
    from ising_toolkit.models import Ising1D, Ising2D, Ising3D
    from ising_toolkit.samplers import MetropolisSampler, WolffSampler

    model_name, size, temperatures, steps, equilibration, algorithm = args

    model_classes = {
        'ising1d': Ising1D,
        'ising2d': Ising2D,
        'ising3d': Ising3D,
    }

    results = {
        'temperatures': [],
        'energies': [],
        'energy_stds': [],
        'magnetizations': [],
        'magnetization_stds': [],
        'heat_capacities': [],
        'susceptibilities': [],
    }

    for T in temperatures:
        # Create model
        model_instance = model_classes[model_name](size=size, temperature=T)

        # Create sampler
        if algorithm == 'metropolis':
            sampler = MetropolisSampler(model_instance)
        else:
            sampler = WolffSampler(model_instance)

        # Equilibration
        for _ in range(equilibration):
            sampler.step()

        # Production
        energies = []
        magnetizations = []

        for i in range(steps):
            sampler.step()
            if i % 10 == 0:
                energies.append(model_instance.get_energy() / model_instance.n_spins)
                magnetizations.append(
                    np.abs(model_instance.get_magnetization()) / model_instance.n_spins
                )

        energies = np.array(energies)
        magnetizations = np.array(magnetizations)

        E_mean = np.mean(energies)
        E_std = np.std(energies)
        M_mean = np.mean(magnetizations)
        M_std = np.std(magnetizations)
        C = (1 / T**2) * np.var(energies) * model_instance.n_spins
        chi = (1 / T) * np.var(magnetizations) * model_instance.n_spins

        results['temperatures'].append(T)
        results['energies'].append(E_mean)
        results['energy_stds'].append(E_std)
        results['magnetizations'].append(M_mean)
        results['magnetization_stds'].append(M_std)
        results['heat_capacities'].append(C)
        results['susceptibilities'].append(chi)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


@main.command()
@click.option('--model', '-m',
              type=click.Choice(['ising1d', 'ising2d', 'ising3d']),
              required=True,
              help='Model type')
@click.option('--size', '-L',
              type=int,
              multiple=True,
              required=True,
              help='Lattice size(s) - can specify multiple')
@click.option('--temp-start',
              type=float,
              required=True,
              help='Start temperature')
@click.option('--temp-end',
              type=float,
              required=True,
              help='End temperature')
@click.option('--temp-steps',
              type=int,
              default=50,
              show_default=True,
              help='Number of temperature points')
@click.option('--steps', '-n',
              type=int,
              default=100000,
              show_default=True,
              help='MC steps per temperature')
@click.option('--equilibration', '-e',
              type=int,
              default=10000,
              show_default=True,
              help='Equilibration steps')
@click.option('--algorithm', '-a',
              type=click.Choice(['metropolis', 'wolff']),
              default='metropolis',
              show_default=True,
              help='Sampling algorithm')
@click.option('--parallel', '-p',
              type=int,
              default=1,
              show_default=True,
              help='Number of parallel workers')
@click.option('--output', '-o',
              type=click.Path(),
              required=True,
              help='Output directory')
@click.option('--format', '-f',
              type=click.Choice(['csv', 'npz', 'hdf5']),
              default='csv',
              show_default=True,
              help='Output file format')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Verbose output')
def sweep(model, size, temp_start, temp_end, temp_steps, steps, equilibration,
          algorithm, parallel, output, format, verbose):
    """Run temperature sweep for phase diagram.

    Performs simulations at multiple temperatures for one or more system sizes.
    Supports parallel execution and multiple output formats.

    Examples:

        # Single size sweep
        ising-sim sweep -m ising2d -L 32 --temp-start 1.5 --temp-end 3.5 -o results/

        # Multiple sizes for finite-size scaling
        ising-sim sweep -m ising2d -L 16 -L 32 -L 64 --temp-start 2.0 --temp-end 2.5 -o results/

        # Parallel execution with 4 workers
        ising-sim sweep -m ising2d -L 32 -L 64 --temp-start 2.0 --temp-end 2.5 -p 4 -o results/

        # Save as HDF5
        ising-sim sweep -m ising2d -L 32 --temp-start 2.0 --temp-end 2.5 -f hdf5 -o results/
    """
    import os
    from datetime import datetime

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate temperature array
    temperatures = np.linspace(temp_start, temp_end, temp_steps)
    sizes = list(size)  # Convert tuple to list

    click.echo("\n" + "="*60)
    click.echo("Temperature Sweep for Phase Diagram")
    click.echo("="*60)
    click.echo(f"Model:         {model}")
    click.echo(f"Sizes:         {sizes}")
    click.echo(f"T range:       [{temp_start}, {temp_end}], {temp_steps} points")
    click.echo(f"Algorithm:     {algorithm}")
    click.echo(f"Steps:         {steps} (equilibration: {equilibration})")
    click.echo(f"Workers:       {parallel}")
    click.echo(f"Output:        {output_dir}")
    click.echo(f"Format:        {format}")
    click.echo("="*60 + "\n")

    all_results = {}

    if parallel > 1 and len(sizes) > 1:
        # Parallel execution using multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Prepare arguments for each size
        tasks = [
            (model, L, temperatures, steps, equilibration, algorithm)
            for L in sizes
        ]

        click.echo(f"Running {len(sizes)} sweeps in parallel...")

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            future_to_size = {
                executor.submit(_run_single_sweep, task): task[1]
                for task in tasks
            }

            with click.progressbar(length=len(sizes), label='Sweeping') as bar:
                for future in as_completed(future_to_size):
                    L = future_to_size[future]
                    try:
                        result = future.result()
                        all_results[L] = result
                        if verbose:
                            click.echo(f"\n  Completed L={L}")
                    except Exception as e:
                        click.echo(f"\n  Error for L={L}: {e}", err=True)
                    bar.update(1)
    else:
        # Sequential execution with progress bar
        for L in sizes:
            click.echo(f"\nProcessing L={L}...")

            results = {
                'temperatures': [],
                'energies': [],
                'energy_stds': [],
                'magnetizations': [],
                'magnetization_stds': [],
                'heat_capacities': [],
                'susceptibilities': [],
            }

            from ising_toolkit.models import Ising1D, Ising2D, Ising3D
            from ising_toolkit.samplers import MetropolisSampler, WolffSampler

            model_classes = {
                'ising1d': Ising1D,
                'ising2d': Ising2D,
                'ising3d': Ising3D,
            }

            with click.progressbar(temperatures, label=f'L={L}') as temps:
                for T in temps:
                    # Create model
                    model_instance = model_classes[model](size=L, temperature=T)

                    # Create sampler
                    if algorithm == 'metropolis':
                        sampler = MetropolisSampler(model_instance)
                    else:
                        sampler = WolffSampler(model_instance)

                    # Equilibration
                    for _ in range(equilibration):
                        sampler.step()

                    # Production
                    energies = []
                    magnetizations = []

                    for i in range(steps):
                        sampler.step()
                        if i % 10 == 0:
                            energies.append(model_instance.get_energy() / model_instance.n_spins)
                            magnetizations.append(
                                np.abs(model_instance.get_magnetization()) / model_instance.n_spins
                            )

                    energies = np.array(energies)
                    magnetizations = np.array(magnetizations)

                    E_mean = np.mean(energies)
                    E_std = np.std(energies)
                    M_mean = np.mean(magnetizations)
                    M_std = np.std(magnetizations)
                    C = (1 / T**2) * np.var(energies) * model_instance.n_spins
                    chi = (1 / T) * np.var(magnetizations) * model_instance.n_spins

                    results['temperatures'].append(T)
                    results['energies'].append(E_mean)
                    results['energy_stds'].append(E_std)
                    results['magnetizations'].append(M_mean)
                    results['magnetization_stds'].append(M_std)
                    results['heat_capacities'].append(C)
                    results['susceptibilities'].append(chi)

            # Convert to arrays
            for key in results:
                results[key] = np.array(results[key])

            all_results[L] = results

    # Save results for each size
    click.echo("\n" + "-"*60)
    click.echo("Saving results...")

    saved_files = []

    for L, results in all_results.items():
        filename = _generate_sweep_filename(model, L, temp_start, temp_end, algorithm)

        if format == 'csv':
            # Save as CSV
            filepath = output_dir / f"{filename}.csv"

            # Create header
            header = "temperature,energy,energy_std,magnetization,magnetization_std,heat_capacity,susceptibility"

            # Stack data
            data = np.column_stack([
                results['temperatures'],
                results['energies'],
                results['energy_stds'],
                results['magnetizations'],
                results['magnetization_stds'],
                results['heat_capacities'],
                results['susceptibilities'],
            ])

            np.savetxt(filepath, data, delimiter=',', header=header, comments='')
            saved_files.append(filepath)

        elif format == 'npz':
            # Save as NPZ
            filepath = output_dir / f"{filename}.npz"

            np.savez(
                filepath,
                model=model,
                size=L,
                algorithm=algorithm,
                steps=steps,
                equilibration=equilibration,
                **results
            )
            saved_files.append(filepath)

        elif format == 'hdf5':
            # Save as HDF5
            try:
                import h5py
            except ImportError:
                raise click.ClickException("h5py required for HDF5 format. Install with: pip install h5py")

            filepath = output_dir / f"{filename}.h5"

            with h5py.File(filepath, 'w') as f:
                # Metadata
                f.attrs['model'] = model
                f.attrs['size'] = L
                f.attrs['algorithm'] = algorithm
                f.attrs['steps'] = steps
                f.attrs['equilibration'] = equilibration
                f.attrs['created'] = datetime.now().isoformat()

                # Data
                for key, value in results.items():
                    f.create_dataset(key, data=value)

            saved_files.append(filepath)

        if verbose:
            click.echo(f"  Saved: {filepath}")

    # Save combined summary
    summary_file = output_dir / f"sweep_summary_{model}_{algorithm}.csv"
    with open(summary_file, 'w') as f:
        f.write("# Temperature Sweep Summary\n")
        f.write(f"# Model: {model}\n")
        f.write(f"# Algorithm: {algorithm}\n")
        f.write(f"# Temperature range: {temp_start} to {temp_end} ({temp_steps} points)\n")
        f.write(f"# Steps: {steps}, Equilibration: {equilibration}\n")
        f.write(f"# Sizes: {sizes}\n")
        f.write(f"# Created: {datetime.now().isoformat()}\n")
        f.write("#\n")
        f.write("size,T_peak_chi,chi_max,T_peak_C,C_max\n")

        for L in sorted(all_results.keys()):
            results = all_results[L]
            idx_chi = np.argmax(results['susceptibilities'])
            idx_C = np.argmax(results['heat_capacities'])
            f.write(f"{L},{results['temperatures'][idx_chi]:.6f},"
                    f"{results['susceptibilities'][idx_chi]:.6f},"
                    f"{results['temperatures'][idx_C]:.6f},"
                    f"{results['heat_capacities'][idx_C]:.6f}\n")

    saved_files.append(summary_file)

    # Print summary
    click.echo("\n" + "="*60)
    click.echo("Sweep Complete!")
    click.echo("="*60)

    click.echo(f"\n{'Size':>6} {'T_peak(χ)':>12} {'χ_max':>12} {'T_peak(C)':>12} {'C_max':>12}")
    click.echo("-"*60)

    for L in sorted(all_results.keys()):
        results = all_results[L]
        idx_chi = np.argmax(results['susceptibilities'])
        idx_C = np.argmax(results['heat_capacities'])
        T_peak_chi = results['temperatures'][idx_chi]
        chi_max = results['susceptibilities'][idx_chi]
        T_peak_C = results['temperatures'][idx_C]
        C_max = results['heat_capacities'][idx_C]

        click.echo(f"{L:>6} {T_peak_chi:>12.4f} {chi_max:>12.4f} {T_peak_C:>12.4f} {C_max:>12.4f}")

    click.echo("="*60)
    click.echo(f"\nSaved {len(saved_files)} files to {output_dir}/")

    for f in saved_files:
        click.echo(f"  - {f.name}")


@main.command()
@click.option('--model', '-m',
              type=click.Choice(['ising1d', 'ising2d', 'ising3d']),
              required=True,
              help='Model type')
def info(model):
    """Show information about a model.

    Displays theoretical values, critical temperature, and
    critical exponents for the specified model.

    Examples:

        ising-sim info --model ising2d
    """
    from ising_toolkit.utils.constants import (
        CRITICAL_TEMP_2D, CRITICAL_TEMP_3D,
        CRITICAL_EXPONENT_BETA_2D, CRITICAL_EXPONENT_GAMMA_2D,
        CRITICAL_EXPONENT_NU_2D, CRITICAL_EXPONENT_ALPHA_2D,
        CRITICAL_EXPONENT_BETA_3D, CRITICAL_EXPONENT_GAMMA_3D,
        CRITICAL_EXPONENT_NU_3D,
    )

    click.echo("\n" + "="*50)
    click.echo(f"Model Information: {model}")
    click.echo("="*50)

    if model == 'ising1d':
        click.echo("\n1D Ising Model")
        click.echo("-"*50)
        click.echo("No phase transition at finite temperature.")
        click.echo("The system is paramagnetic for all T > 0.")
        click.echo("\nGround state energy: E/N = -J")
        click.echo("Exact solution: Ising (1925)")

    elif model == 'ising2d':
        click.echo("\n2D Ising Model (Square Lattice)")
        click.echo("-"*50)
        click.echo(f"Critical temperature:  Tc = {CRITICAL_TEMP_2D:.6f} J/kB")
        click.echo(f"                          = 2/ln(1+√2)")
        click.echo("\nCritical exponents (exact):")
        click.echo(f"  β  = {CRITICAL_EXPONENT_BETA_2D:.6f}  (magnetization)")
        click.echo(f"  γ  = {CRITICAL_EXPONENT_GAMMA_2D:.6f}  (susceptibility)")
        click.echo(f"  ν  = {CRITICAL_EXPONENT_NU_2D:.6f}  (correlation length)")
        click.echo(f"  α  = {CRITICAL_EXPONENT_ALPHA_2D:.6f}  (heat capacity, log divergence)")
        click.echo("\nExact solution: Onsager (1944)")

    elif model == 'ising3d':
        click.echo("\n3D Ising Model (Cubic Lattice)")
        click.echo("-"*50)
        click.echo(f"Critical temperature:  Tc ≈ {CRITICAL_TEMP_3D:.6f} J/kB")
        click.echo("\nCritical exponents (numerical):")
        click.echo(f"  β  ≈ {CRITICAL_EXPONENT_BETA_3D:.6f}  (magnetization)")
        click.echo(f"  γ  ≈ {CRITICAL_EXPONENT_GAMMA_3D:.6f}  (susceptibility)")
        click.echo(f"  ν  ≈ {CRITICAL_EXPONENT_NU_3D:.6f}  (correlation length)")
        click.echo("\nNo exact solution known.")

    click.echo("\n" + "="*50)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--observable', '-O',
              type=click.Choice(['energy', 'magnetization', 'heat_capacity', 'susceptibility']),
              default='magnetization',
              help='Observable to plot')
@click.option('--output', '-o',
              type=click.Path(),
              default=None,
              help='Output image file')
def plot(input_file, observable, output):
    """Plot results from a sweep simulation.

    Examples:

        ising-sim plot sweep.npz --observable magnetization
        ising-sim plot sweep.npz -O susceptibility -o chi_vs_T.png
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        raise click.ClickException("matplotlib required for plotting")

    # Load data
    data = np.load(input_file, allow_pickle=True)

    if 'temperatures' not in data:
        raise click.ClickException("Input file must be from a sweep simulation")

    temps = data['temperatures']

    observable_map = {
        'energy': ('energies', 'Energy per spin (E/N)'),
        'magnetization': ('magnetizations', '|Magnetization| per spin (|M|/N)'),
        'heat_capacity': ('heat_capacities', 'Heat capacity (C/N)'),
        'susceptibility': ('susceptibilities', 'Susceptibility (χ/N)'),
    }

    key, ylabel = observable_map[observable]

    if key not in data:
        raise click.ClickException(f"Observable '{observable}' not found in file")

    values = data[key]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temps, values, 'o-', markersize=4)
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{observable.replace('_', ' ').title()} vs Temperature", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark peak for susceptibility/heat capacity
    if observable in ['susceptibility', 'heat_capacity']:
        idx_max = np.argmax(values)
        ax.axvline(temps[idx_max], color='red', linestyle='--', alpha=0.7,
                   label=f'Peak at T={temps[idx_max]:.3f}')
        ax.legend()

    plt.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=150)
        click.echo(f"Plot saved to {output}")
    else:
        # Save to default location
        default_output = Path(input_file).stem + f"_{observable}.png"
        plt.savefig(default_output, dpi=150)
        click.echo(f"Plot saved to {default_output}")

    plt.close()


@main.command()
def benchmark():
    """Run performance benchmark.

    Compares Metropolis and Wolff algorithms on the 2D Ising model.
    """
    from ising_toolkit.models import Ising2D
    from ising_toolkit.samplers import MetropolisSampler, WolffSampler
    import time

    click.echo("\n" + "="*50)
    click.echo("Performance Benchmark")
    click.echo("="*50)

    sizes = [16, 32, 64]
    T = 2.269  # Critical temperature

    click.echo(f"\nTemperature: T = {T} (critical)")
    click.echo(f"Steps: 10000 per test\n")

    click.echo(f"{'Size':>6} {'Metropolis (s)':>16} {'Wolff (s)':>16} {'Speedup':>10}")
    click.echo("-"*50)

    for L in sizes:
        # Metropolis
        model = Ising2D(size=L, temperature=T)
        sampler = MetropolisSampler(model)

        start = time.time()
        for _ in range(10000):
            sampler.step()
        metro_time = time.time() - start

        # Wolff
        model = Ising2D(size=L, temperature=T)
        sampler = WolffSampler(model)

        start = time.time()
        for _ in range(10000):
            sampler.step()
        wolff_time = time.time() - start

        speedup = metro_time / wolff_time if wolff_time > 0 else float('inf')

        click.echo(f"{L:>6} {metro_time:>16.3f} {wolff_time:>16.3f} {speedup:>10.2f}x")

    click.echo("="*50)
    click.echo("\nNote: Wolff is more efficient near Tc due to reduced")
    click.echo("critical slowing down.")


@main.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--observables', '-O',
              type=str,
              default='all',
              help='Comma-separated list of observables or "all"')
@click.option('--bootstrap', '-b',
              type=int,
              default=1000,
              show_default=True,
              help='Number of bootstrap samples for error estimation')
@click.option('--output', '-o',
              type=click.Path(),
              default=None,
              help='Output file path (optional)')
@click.option('--format', '-f',
              type=click.Choice(['csv', 'json', 'hdf5']),
              default='csv',
              show_default=True,
              help='Output file format')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Verbose output')
def analyze(input, observables, bootstrap, output, format, verbose):
    """Analyze simulation results.

    Loads simulation results from a file or directory and computes
    statistical analysis with bootstrap error estimates.

    INPUT can be:
    - A single .npz, .csv, or .h5 file from a simulation
    - A directory containing multiple result files

    Examples:

        # Analyze a single simulation result
        ising-sim analyze results.npz

        # Analyze all files in a directory
        ising-sim analyze results/

        # Analyze specific observables
        ising-sim analyze results.npz --observables energy,magnetization

        # Save analysis to file
        ising-sim analyze results.npz -o analysis.csv

        # Save as JSON
        ising-sim analyze results.npz -o analysis.json -f json
    """
    from datetime import datetime

    # Parse observables
    all_observables = ['energy', 'magnetization', 'heat_capacity', 'susceptibility']
    if observables.lower() == 'all':
        obs_list = all_observables
    else:
        obs_list = [o.strip() for o in observables.split(',')]
        # Validate observables
        for obs in obs_list:
            if obs not in all_observables:
                click.echo(f"Warning: Unknown observable '{obs}'", err=True)

    if verbose:
        click.echo(f"Loading results from: {input}")
        click.echo(f"Observables: {obs_list}")
        click.echo(f"Bootstrap samples: {bootstrap}")

    # Load results
    try:
        loaded = load_results_from_path(input)
    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Failed to load results: {e}")

    result_type = loaded['type']

    if verbose:
        click.echo(f"Result type: {result_type}")

    # Analyze based on result type
    all_analysis = []

    if result_type == 'single':
        # Single simulation result
        analysis = _analyze_single_result(loaded['data'], obs_list, bootstrap)
        analysis['source'] = str(loaded['source'])
        all_analysis.append(analysis)

    elif result_type == 'sweep':
        # Temperature sweep result
        analysis = _analyze_sweep_result(loaded['data'], obs_list, bootstrap)
        analysis['source'] = str(loaded['source'])
        all_analysis.append(analysis)

    elif result_type == 'multi':
        # Multiple files
        click.echo(f"\nAnalyzing {loaded['count']} result files...")

        for i, result in enumerate(loaded['data']):
            if verbose:
                click.echo(f"  Processing: {result['source'].name}")

            if result['type'] == 'single':
                analysis = _analyze_single_result(result['data'], obs_list, bootstrap)
            else:
                analysis = _analyze_sweep_result(result['data'], obs_list, bootstrap)

            analysis['source'] = str(result['source'])
            all_analysis.append(analysis)

    # Print results
    for analysis in all_analysis:
        click.echo("\n" + format_results_table(analysis, obs_list))

    # Additional summary for sweeps
    for analysis in all_analysis:
        if 'susceptibility' in analysis and isinstance(analysis['susceptibility'], dict):
            if 'peak_temperature' in analysis['susceptibility']:
                click.echo(f"\nEstimated Tc (χ peak): {analysis['susceptibility']['peak_temperature']:.4f}")
        if 'heat_capacity' in analysis and isinstance(analysis['heat_capacity'], dict):
            if 'peak_temperature' in analysis['heat_capacity']:
                click.echo(f"Estimated Tc (C peak): {analysis['heat_capacity']['peak_temperature']:.4f}")

    # Save results if output specified
    if output is not None:
        output_path = Path(output)

        if format == 'csv':
            # Flatten analysis for CSV
            with open(output_path, 'w') as f:
                # Write header
                headers = ['source', 'model', 'size', 'temperature']
                for obs in obs_list:
                    headers.extend([f'{obs}_mean', f'{obs}_std', f'{obs}_error'])
                f.write(','.join(headers) + '\n')

                # Write data rows
                for analysis in all_analysis:
                    row = [
                        str(analysis.get('source', '')),
                        str(analysis.get('model', '')),
                        str(analysis.get('size', '')),
                        str(analysis.get('temperature', '')),
                    ]
                    for obs in obs_list:
                        if obs in analysis and isinstance(analysis[obs], dict):
                            row.append(str(analysis[obs].get('mean', '')))
                            row.append(str(analysis[obs].get('std', '')))
                            row.append(str(analysis[obs].get('bootstrap_error',
                                                             analysis[obs].get('error', ''))))
                        else:
                            row.extend(['', '', ''])
                    f.write(','.join(row) + '\n')

            click.echo(f"\nAnalysis saved to {output_path}")

        elif format == 'json':
            # Convert numpy types to Python types for JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj

            output_data = {
                'analysis_date': datetime.now().isoformat(),
                'observables': obs_list,
                'bootstrap_samples': bootstrap,
                'results': [convert_numpy(a) for a in all_analysis]
            }

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            click.echo(f"\nAnalysis saved to {output_path}")

        elif format == 'hdf5':
            try:
                import h5py
            except ImportError:
                raise click.ClickException("h5py required for HDF5 format")

            with h5py.File(output_path, 'w') as f:
                f.attrs['analysis_date'] = datetime.now().isoformat()
                f.attrs['observables'] = obs_list
                f.attrs['bootstrap_samples'] = bootstrap

                for i, analysis in enumerate(all_analysis):
                    grp = f.create_group(f'result_{i}')

                    for key, value in analysis.items():
                        if isinstance(value, dict):
                            subgrp = grp.create_group(key)
                            for k, v in value.items():
                                if isinstance(v, (int, float, np.number)):
                                    subgrp.attrs[k] = float(v)
                                elif isinstance(v, str):
                                    subgrp.attrs[k] = v
                        elif isinstance(value, (int, float, np.number)):
                            grp.attrs[key] = float(value) if isinstance(value, np.floating) else value
                        elif isinstance(value, str):
                            grp.attrs[key] = value
                        elif isinstance(value, np.ndarray):
                            grp.create_dataset(key, data=value)

            click.echo(f"\nAnalysis saved to {output_path}")


if __name__ == '__main__':
    main()
