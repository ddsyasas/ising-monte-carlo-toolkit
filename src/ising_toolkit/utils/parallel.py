"""
Parallel execution utilities for Monte Carlo simulations.

This module provides functions for running simulations in parallel
across multiple CPU cores, with proper error handling and progress tracking.

Examples
--------
>>> from ising_toolkit.utils.parallel import parallel_map, run_temperature_sweep_parallel
>>>
>>> # Generic parallel map
>>> results = parallel_map(expensive_function, items, n_workers=4)
>>>
>>> # Parallel temperature sweep
>>> results = run_temperature_sweep_parallel(
...     model_class=Ising2D,
...     size=32,
...     temperatures=[2.0, 2.1, 2.2, 2.3],
...     n_steps=10000,
...     n_workers=4
... )
"""

import multiprocessing as mp
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):
        """Fallback when tqdm is not available."""
        return iterable


@dataclass
class ParallelResult:
    """Container for parallel execution results.

    Attributes
    ----------
    results : list
        Successful results in order
    errors : list
        List of (index, exception) tuples for failed items
    n_successful : int
        Number of successfully processed items
    n_failed : int
        Number of failed items
    """
    results: List[Any]
    errors: List[Tuple[int, Exception]]

    @property
    def n_successful(self) -> int:
        """Number of tasks that completed successfully."""
        return len([r for r in self.results if r is not None])

    @property
    def n_failed(self) -> int:
        """Number of tasks that raised exceptions."""
        return len(self.errors)

    @property
    def success_rate(self) -> float:
        """Fraction of tasks that completed successfully."""
        total = len(self.results)
        return self.n_successful / total if total > 0 else 0.0


def get_optimal_workers(n_items: int, max_workers: Optional[int] = None) -> int:
    """Determine optimal number of workers.

    Parameters
    ----------
    n_items : int
        Number of items to process
    max_workers : int, optional
        Maximum workers to use

    Returns
    -------
    int
        Optimal number of workers
    """
    cpu_count = mp.cpu_count()

    if max_workers is None:
        max_workers = cpu_count

    # Don't use more workers than items
    optimal = min(n_items, max_workers, cpu_count)

    # At least 1 worker
    return max(1, optimal)


def parallel_map(
    func: Callable,
    items: List[Any],
    n_workers: Optional[int] = None,
    progress: bool = True,
    desc: str = "Processing",
    use_threads: bool = False,
    timeout: Optional[float] = None,
    chunksize: int = 1,
) -> List[Any]:
    """Execute function on items in parallel with order preserved.

    Parameters
    ----------
    func : callable
        Function to apply to each item. Must be picklable for multiprocessing.
    items : list
        Items to process.
    n_workers : int, optional
        Number of parallel workers. Default: number of CPU cores.
    progress : bool, optional
        Show progress bar (requires tqdm). Default: True.
    desc : str, optional
        Progress bar description. Default: "Processing".
    use_threads : bool, optional
        Use threads instead of processes. Default: False.
        Use for I/O-bound tasks; processes are better for CPU-bound.
    timeout : float, optional
        Timeout in seconds for each task. Default: None (no timeout).
    chunksize : int, optional
        Number of items per worker batch. Default: 1.

    Returns
    -------
    results : list
        Results in same order as input items.
        Failed items will have None as their result.

    Raises
    ------
    RuntimeError
        If all items fail to process.

    Examples
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> parallel_map(square, [1, 2, 3, 4], n_workers=2)
    [1, 4, 9, 16]
    """
    if not items:
        return []

    n_items = len(items)
    n_workers = get_optimal_workers(n_items, n_workers)

    # For single worker or single item, use sequential execution
    if n_workers == 1 or n_items == 1:
        iterator = items
        if progress and TQDM_AVAILABLE:
            iterator = tqdm(items, desc=desc, file=sys.stderr)

        results = []
        for item in iterator:
            try:
                results.append(func(item))
            except Exception as e:
                warnings.warn(f"Task failed: {e}", RuntimeWarning)
                results.append(None)
        return results

    # Choose executor type
    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    # Initialize results with None
    results = [None] * n_items
    errors = []

    with ExecutorClass(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for i, item in enumerate(items):
            future = executor.submit(func, item)
            future_to_idx[future] = i

        # Collect results as they complete
        completed_futures = as_completed(future_to_idx, timeout=timeout)

        if progress and TQDM_AVAILABLE:
            completed_futures = tqdm(
                completed_futures,
                total=n_items,
                desc=desc,
                file=sys.stderr
            )

        for future in completed_futures:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result(timeout=timeout)
            except Exception as e:
                errors.append((idx, e))
                results[idx] = None

    if errors:
        n_errors = len(errors)
        warnings.warn(
            f"{n_errors}/{n_items} tasks failed. First error: {errors[0][1]}",
            RuntimeWarning
        )

    if all(r is None for r in results):
        raise RuntimeError("All parallel tasks failed")

    return results


def parallel_map_with_errors(
    func: Callable,
    items: List[Any],
    n_workers: Optional[int] = None,
    progress: bool = True,
    desc: str = "Processing",
) -> ParallelResult:
    """Execute function in parallel and return detailed results.

    Unlike parallel_map, this function returns a ParallelResult object
    that includes information about which items failed and why.

    Parameters
    ----------
    func : callable
        Function to apply to each item.
    items : list
        Items to process.
    n_workers : int, optional
        Number of parallel workers.
    progress : bool, optional
        Show progress bar.
    desc : str, optional
        Progress bar description.

    Returns
    -------
    ParallelResult
        Object containing results and error information.
    """
    if not items:
        return ParallelResult(results=[], errors=[])

    n_items = len(items)
    n_workers = get_optimal_workers(n_items, n_workers)

    results = [None] * n_items
    errors = []

    if n_workers == 1 or n_items == 1:
        iterator = items
        if progress and TQDM_AVAILABLE:
            iterator = tqdm(items, desc=desc, file=sys.stderr)

        for i, item in enumerate(iterator):
            try:
                results[i] = func(item)
            except Exception as e:
                errors.append((i, e))

        return ParallelResult(results=results, errors=errors)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(func, item): i for i, item in enumerate(items)}

        completed = as_completed(future_to_idx)
        if progress and TQDM_AVAILABLE:
            completed = tqdm(completed, total=n_items, desc=desc, file=sys.stderr)

        for future in completed:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                errors.append((idx, e))

    return ParallelResult(results=results, errors=errors)


# =============================================================================
# Temperature Sweep Worker Functions
# =============================================================================

def run_temperature_point(args: tuple) -> Dict[str, Any]:
    """Worker function for parallel temperature sweep.

    This function is designed to be picklable for multiprocessing.
    It creates a model, runs a simulation, and returns observables.

    Parameters
    ----------
    args : tuple
        (model_class, model_kwargs, sampler_class, sampler_kwargs,
         temperature, n_steps, equilibration, seed)

    Returns
    -------
    dict
        Dictionary containing:
        - temperature: float
        - energies: list of energy measurements
        - magnetizations: list of magnetization measurements
        - energy_mean, energy_std: statistics
        - magnetization_mean, magnetization_std: statistics
        - acceptance_rate: float
        - error: str or None if successful
    """
    (
        model_class_name,
        model_kwargs,
        sampler_class_name,
        sampler_kwargs,
        temperature,
        n_steps,
        equilibration,
        measurement_interval,
        seed,
    ) = args

    result = {
        'temperature': temperature,
        'error': None,
    }

    try:
        # Import classes dynamically to avoid pickling issues
        from ising_toolkit.models import Ising1D, Ising2D, Ising3D
        from ising_toolkit.samplers import MetropolisSampler, WolffSampler

        model_classes = {
            'Ising1D': Ising1D,
            'Ising2D': Ising2D,
            'Ising3D': Ising3D,
        }

        sampler_classes = {
            'MetropolisSampler': MetropolisSampler,
            'WolffSampler': WolffSampler,
        }

        model_class = model_classes.get(model_class_name)
        sampler_class = sampler_classes.get(sampler_class_name, MetropolisSampler)

        if model_class is None:
            raise ValueError(f"Unknown model class: {model_class_name}")

        # Create model
        model = model_class(temperature=temperature, **model_kwargs)
        model.initialize('random')

        if seed is not None:
            model.set_seed(seed)

        # Create sampler
        sampler = sampler_class(model, seed=seed, **sampler_kwargs)

        # Equilibration
        for _ in range(equilibration):
            sampler.step()

        # Measurement
        energies = []
        magnetizations = []

        for step in range(n_steps):
            sampler.step()

            if step % measurement_interval == 0:
                energies.append(model.get_energy())
                magnetizations.append(model.get_magnetization())

        # Compute statistics
        energies = np.array(energies)
        magnetizations = np.array(magnetizations)
        n_spins = model.n_spins

        result.update({
            'size': model_kwargs.get('size', model_kwargs.get('L', 0)),
            'n_spins': n_spins,
            'n_steps': n_steps,
            'equilibration': equilibration,
            'energies': energies.tolist(),
            'magnetizations': magnetizations.tolist(),
            'energy_mean': float(np.mean(energies)),
            'energy_std': float(np.std(energies)),
            'energy_per_spin': float(np.mean(energies) / n_spins),
            'magnetization_mean': float(np.mean(magnetizations)),
            'magnetization_std': float(np.std(magnetizations)),
            'abs_magnetization_mean': float(np.mean(np.abs(magnetizations))),
            'magnetization_per_spin': float(np.mean(np.abs(magnetizations)) / n_spins),
            'acceptance_rate': sampler.acceptance_rate,
        })

        # Compute derived quantities
        T = temperature
        var_E = np.var(energies)
        var_M = np.var(magnetizations)

        result['specific_heat'] = float(var_E / (T ** 2))
        result['susceptibility'] = float(n_spins * var_M / T)

        # Binder cumulant
        m2 = np.mean(magnetizations ** 2)
        m4 = np.mean(magnetizations ** 4)
        if m2 > 1e-10:
            result['binder'] = float(1 - m4 / (3 * m2 ** 2))
        else:
            result['binder'] = 0.0

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
        result['traceback'] = traceback.format_exc()

    return result


def run_temperature_sweep_parallel(
    model_class: str,
    size: int,
    temperatures: List[float],
    n_steps: int = 10000,
    equilibration: int = 1000,
    measurement_interval: int = 10,
    algorithm: str = 'metropolis',
    n_workers: Optional[int] = None,
    seed: Optional[int] = None,
    progress: bool = True,
    model_kwargs: Optional[Dict] = None,
    sampler_kwargs: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """Run temperature sweep in parallel.

    Parameters
    ----------
    model_class : str
        Model class name: 'Ising1D', 'Ising2D', or 'Ising3D'.
    size : int
        System size (L for 2D/3D, N for 1D).
    temperatures : list of float
        Temperatures to simulate.
    n_steps : int, optional
        Monte Carlo steps per temperature. Default: 10000.
    equilibration : int, optional
        Equilibration steps. Default: 1000.
    measurement_interval : int, optional
        Steps between measurements. Default: 10.
    algorithm : str, optional
        'metropolis' or 'wolff'. Default: 'metropolis'.
    n_workers : int, optional
        Number of parallel workers. Default: CPU count.
    seed : int, optional
        Base random seed. Each temperature gets seed + index.
    progress : bool, optional
        Show progress bar. Default: True.
    model_kwargs : dict, optional
        Additional model constructor arguments.
    sampler_kwargs : dict, optional
        Additional sampler constructor arguments.

    Returns
    -------
    list of dict
        Results for each temperature, sorted by temperature.

    Examples
    --------
    >>> results = run_temperature_sweep_parallel(
    ...     model_class='Ising2D',
    ...     size=32,
    ...     temperatures=np.linspace(2.0, 2.5, 20),
    ...     n_steps=10000,
    ...     n_workers=4
    ... )
    >>> for r in results:
    ...     print(f"T={r['temperature']:.3f}, E={r['energy_per_spin']:.4f}")
    """
    if model_kwargs is None:
        model_kwargs = {}
    if sampler_kwargs is None:
        sampler_kwargs = {}

    # Add size to model kwargs
    model_kwargs['size'] = size

    # Map algorithm to sampler class
    sampler_class_name = 'WolffSampler' if algorithm.lower() == 'wolff' else 'MetropolisSampler'

    # Prepare arguments for each temperature
    args_list = []
    for i, T in enumerate(temperatures):
        task_seed = seed + i if seed is not None else None
        args = (
            model_class,
            model_kwargs.copy(),
            sampler_class_name,
            sampler_kwargs.copy(),
            float(T),
            n_steps,
            equilibration,
            measurement_interval,
            task_seed,
        )
        args_list.append(args)

    # Run in parallel
    results = parallel_map(
        run_temperature_point,
        args_list,
        n_workers=n_workers,
        progress=progress,
        desc=f"Temperature sweep ({len(temperatures)} points)"
    )

    # Filter out failed results and sort by temperature
    valid_results = [r for r in results if r is not None and r.get('error') is None]
    failed_results = [r for r in results if r is not None and r.get('error') is not None]

    if failed_results:
        for r in failed_results[:3]:
            warnings.warn(f"T={r['temperature']}: {r['error']}", RuntimeWarning)

    valid_results.sort(key=lambda x: x['temperature'])

    return valid_results


def run_size_sweep_parallel(
    model_class: str,
    sizes: List[int],
    temperature: float,
    n_steps: int = 10000,
    equilibration: int = 1000,
    measurement_interval: int = 10,
    algorithm: str = 'metropolis',
    n_workers: Optional[int] = None,
    seed: Optional[int] = None,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Run system size sweep in parallel.

    Parameters
    ----------
    model_class : str
        Model class name.
    sizes : list of int
        System sizes to simulate.
    temperature : float
        Temperature for all simulations.
    n_steps : int, optional
        Monte Carlo steps per size.
    equilibration : int, optional
        Equilibration steps.
    measurement_interval : int, optional
        Steps between measurements.
    algorithm : str, optional
        'metropolis' or 'wolff'.
    n_workers : int, optional
        Number of parallel workers.
    seed : int, optional
        Base random seed.
    progress : bool, optional
        Show progress bar.

    Returns
    -------
    list of dict
        Results for each size, sorted by size.
    """
    sampler_class_name = 'WolffSampler' if algorithm.lower() == 'wolff' else 'MetropolisSampler'

    args_list = []
    for i, size in enumerate(sizes):
        task_seed = seed + i if seed is not None else None
        args = (
            model_class,
            {'size': size},
            sampler_class_name,
            {},
            float(temperature),
            n_steps,
            equilibration,
            measurement_interval,
            task_seed,
        )
        args_list.append(args)

    results = parallel_map(
        run_temperature_point,
        args_list,
        n_workers=n_workers,
        progress=progress,
        desc=f"Size sweep ({len(sizes)} sizes)"
    )

    valid_results = [r for r in results if r is not None and r.get('error') is None]
    valid_results.sort(key=lambda x: x['size'])

    return valid_results


# =============================================================================
# Batch simulation utilities
# =============================================================================

def run_replicas_parallel(
    model_class: str,
    size: int,
    temperature: float,
    n_replicas: int,
    n_steps: int = 10000,
    equilibration: int = 1000,
    measurement_interval: int = 10,
    algorithm: str = 'metropolis',
    n_workers: Optional[int] = None,
    base_seed: Optional[int] = None,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Run multiple independent replicas in parallel.

    Useful for computing error bars via replica averaging.

    Parameters
    ----------
    model_class : str
        Model class name.
    size : int
        System size.
    temperature : float
        Temperature.
    n_replicas : int
        Number of independent replicas.
    n_steps : int, optional
        Steps per replica.
    equilibration : int, optional
        Equilibration steps.
    measurement_interval : int, optional
        Measurement interval.
    algorithm : str, optional
        Algorithm choice.
    n_workers : int, optional
        Parallel workers.
    base_seed : int, optional
        Base seed (replicas get base_seed + replica_index).
    progress : bool, optional
        Show progress.

    Returns
    -------
    list of dict
        Results from each replica.
    """
    sampler_class_name = 'WolffSampler' if algorithm.lower() == 'wolff' else 'MetropolisSampler'

    args_list = []
    for i in range(n_replicas):
        task_seed = base_seed + i if base_seed is not None else None
        args = (
            model_class,
            {'size': size},
            sampler_class_name,
            {},
            float(temperature),
            n_steps,
            equilibration,
            measurement_interval,
            task_seed,
        )
        args_list.append(args)

    results = parallel_map(
        run_temperature_point,
        args_list,
        n_workers=n_workers,
        progress=progress,
        desc=f"Replicas ({n_replicas})"
    )

    valid_results = [r for r in results if r is not None and r.get('error') is None]
    return valid_results


def aggregate_replica_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple replicas.

    Computes mean and standard error for each observable.

    Parameters
    ----------
    results : list of dict
        Results from run_replicas_parallel.

    Returns
    -------
    dict
        Aggregated results with mean and error for each observable.
    """
    if not results:
        return {}

    # Observable keys to aggregate
    observable_keys = [
        'energy_mean', 'energy_per_spin',
        'magnetization_mean', 'abs_magnetization_mean', 'magnetization_per_spin',
        'specific_heat', 'susceptibility', 'binder',
        'acceptance_rate',
    ]

    aggregated = {
        'n_replicas': len(results),
        'temperature': results[0].get('temperature'),
        'size': results[0].get('size'),
    }

    for key in observable_keys:
        values = [r[key] for r in results if key in r]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_err'] = float(np.std(values) / np.sqrt(len(values)))

    return aggregated
