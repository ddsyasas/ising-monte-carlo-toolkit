"""Temperature sweep functionality for phase transition analysis."""

from typing import Dict, List, Optional, Type, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ising_toolkit.models.base import IsingModel
from ising_toolkit.samplers import create_sampler
from ising_toolkit.analysis.observables import calculate_all_observables
from ising_toolkit.analysis.statistics import bootstrap_mean_error
from ising_toolkit.utils.parallel import parallel_map


def _run_single_temperature(args: tuple) -> dict:
    """Worker function for parallel temperature sweep.

    This function is defined at module level to be picklable for
    multiprocessing.

    Parameters
    ----------
    args : tuple
        (model_class, size, temperature, n_steps, equilibration,
         measurement_interval, algorithm, seed)

    Returns
    -------
    dict
        Dictionary with temperature and all observables.
    """
    (model_class, size, temperature, n_steps, equilibration,
     measurement_interval, algorithm, seed) = args

    # Create model at this temperature
    model = model_class(size=size, temperature=temperature)

    # Create sampler with unique seed per temperature
    if seed is not None:
        # Create deterministic but unique seed for each temperature
        temp_seed = seed + int(temperature * 1000) % (2**31)
    else:
        temp_seed = None

    sampler = create_sampler(algorithm, model, seed=temp_seed)

    # Run simulation
    results = sampler.run(
        n_steps=n_steps,
        equilibration=equilibration,
        measurement_interval=measurement_interval,
        progress=False,  # Disable progress bar in workers
    )

    # Calculate observables
    observables = calculate_all_observables(results)

    # Add temperature to results
    observables['temperature'] = temperature

    # Compute bootstrap errors for key quantities
    n_spins = model.n_spins
    energy_per_spin = results.energy / n_spins
    mag_per_spin = results.magnetization / n_spins
    abs_mag_per_spin = np.abs(mag_per_spin)

    _, energy_err = bootstrap_mean_error(energy_per_spin, n_samples=200)
    _, mag_err = bootstrap_mean_error(mag_per_spin, n_samples=200)
    _, abs_mag_err = bootstrap_mean_error(abs_mag_per_spin, n_samples=200)

    observables['energy_err'] = energy_err
    observables['magnetization_err'] = mag_err
    observables['abs_magnetization_err'] = abs_mag_err

    return observables


class TemperatureSweep:
    """Run and analyze simulations across a temperature range.

    This class provides a convenient interface for running Monte Carlo
    simulations at multiple temperatures and analyzing the results,
    particularly for studying phase transitions.

    Parameters
    ----------
    model_class : type
        The Ising model class to use (Ising1D, Ising2D, or Ising3D).
    size : int
        System size (L for 2D/3D, N for 1D).
    temperatures : array-like
        Array of temperatures to simulate.
    n_steps : int
        Number of Monte Carlo steps per temperature.
    equilibration : int, optional
        Number of equilibration steps. Default is n_steps // 10.
    measurement_interval : int, optional
        Interval between measurements. Default is 1.
    algorithm : str, optional
        Sampling algorithm ('metropolis' or 'wolff'). Default is 'metropolis'.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from ising_toolkit.models import Ising2D
    >>> from ising_toolkit.analysis import TemperatureSweep
    >>>
    >>> # Define temperature range around critical point
    >>> Tc = 2.269
    >>> temps = np.linspace(1.5, 3.5, 21)
    >>>
    >>> # Create sweep
    >>> sweep = TemperatureSweep(
    ...     model_class=Ising2D,
    ...     size=16,
    ...     temperatures=temps,
    ...     n_steps=10000,
    ...     algorithm='wolff',
    ...     seed=42
    ... )
    >>>
    >>> # Run simulations (parallel)
    >>> df = sweep.run(n_workers=4)
    >>>
    >>> # Plot phase diagram
    >>> sweep.plot_phase_diagram()

    Notes
    -----
    For studying phase transitions, it's recommended to:
    - Use the Wolff algorithm near the critical temperature
    - Run longer simulations near Tc where correlations are strongest
    - Use multiple system sizes for finite-size scaling analysis
    """

    def __init__(
        self,
        model_class: Type[IsingModel],
        size: int,
        temperatures: Union[List[float], np.ndarray],
        n_steps: int,
        equilibration: Optional[int] = None,
        measurement_interval: int = 1,
        algorithm: str = 'metropolis',
        seed: Optional[int] = None,
    ):
        self.model_class = model_class
        self.size = size
        self.temperatures = np.asarray(temperatures)
        self.n_steps = n_steps
        self.equilibration = equilibration if equilibration is not None else n_steps // 10
        self.measurement_interval = measurement_interval
        self.algorithm = algorithm
        self.seed = seed

        # Results storage
        self._results_df = None
        self._results_list = None

    def run_single(self, temperature: float) -> dict:
        """Run simulation at a single temperature.

        Parameters
        ----------
        temperature : float
            Temperature to simulate.

        Returns
        -------
        dict
            Dictionary containing all computed observables:
            - temperature
            - energy_mean, energy_std, energy_err
            - magnetization_mean, magnetization_std, magnetization_err
            - abs_magnetization_mean, abs_magnetization_std, abs_magnetization_err
            - heat_capacity
            - susceptibility
            - binder_cumulant
        """
        args = (
            self.model_class,
            self.size,
            temperature,
            self.n_steps,
            self.equilibration,
            self.measurement_interval,
            self.algorithm,
            self.seed,
        )
        return _run_single_temperature(args)

    def run(
        self,
        n_workers: int = 1,
        progress: bool = True,
    ):
        """Run simulations at all temperatures.

        Parameters
        ----------
        n_workers : int, optional
            Number of parallel workers. Default is 1 (sequential).
            Use -1 for all available CPUs.
        progress : bool, optional
            Show progress bar. Default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for temperature and all observables.
            If pandas is not installed, returns a list of dictionaries.

        Notes
        -----
        Uses the parallel_map utility for robust parallel execution with
        proper error handling and progress tracking.
        """
        # Prepare arguments for each temperature
        args_list = [
            (
                self.model_class,
                self.size,
                temp,
                self.n_steps,
                self.equilibration,
                self.measurement_interval,
                self.algorithm,
                self.seed,
            )
            for temp in self.temperatures
        ]

        # Use parallel_map for both sequential and parallel execution
        results = parallel_map(
            func=_run_single_temperature,
            items=args_list,
            n_workers=n_workers,
            progress=progress,
            desc=f"Temperature sweep (L={self.size})",
        )

        # Sort by temperature
        results.sort(key=lambda x: x['temperature'])
        self._results_list = results

        # Convert to DataFrame if pandas available
        if HAS_PANDAS:
            self._results_df = pd.DataFrame(results)
            # Reorder columns for better readability
            column_order = [
                'temperature',
                'energy_mean', 'energy_std', 'energy_err',
                'magnetization_mean', 'magnetization_std', 'magnetization_err',
                'abs_magnetization_mean', 'abs_magnetization_std', 'abs_magnetization_err',
                'heat_capacity', 'susceptibility', 'binder_cumulant',
            ]
            # Only include columns that exist
            column_order = [c for c in column_order if c in self._results_df.columns]
            self._results_df = self._results_df[column_order]
            return self._results_df
        else:
            return results

    @property
    def results(self):
        """Get results as DataFrame (or list if pandas not available)."""
        if self._results_df is not None:
            return self._results_df
        return self._results_list

    def plot_phase_diagram(
        self,
        observables: Optional[List[str]] = None,
        ax=None,
        save: Optional[str] = None,
        show_Tc: bool = True,
        figsize: tuple = (10, 8),
    ):
        """Plot observables versus temperature.

        Parameters
        ----------
        observables : list of str, optional
            Which observables to plot. Default is
            ['abs_magnetization_mean', 'heat_capacity', 'susceptibility', 'binder_cumulant'].
        ax : matplotlib.axes.Axes or list of Axes, optional
            Axes to plot on. If None, creates new figure with subplots.
        save : str, optional
            Path to save figure. If None, figure is not saved.
        show_Tc : bool, optional
            Show vertical line at theoretical critical temperature.
            Default is True.
        figsize : tuple, optional
            Figure size. Default is (10, 8).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : list of matplotlib.axes.Axes
            List of axes objects.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        if self._results_list is None:
            raise RuntimeError("No results available. Run the sweep first.")

        # Default observables
        if observables is None:
            observables = [
                'abs_magnetization_mean',
                'heat_capacity',
                'susceptibility',
                'binder_cumulant',
            ]

        # Get data
        temps = np.array([r['temperature'] for r in self._results_list])

        # Observable display names and error keys
        display_names = {
            'energy_mean': 'Energy per spin',
            'magnetization_mean': 'Magnetization per spin',
            'abs_magnetization_mean': '|Magnetization| per spin',
            'heat_capacity': 'Heat capacity (C/N)',
            'susceptibility': 'Susceptibility (χ/N)',
            'binder_cumulant': 'Binder cumulant (U)',
        }

        error_keys = {
            'energy_mean': 'energy_err',
            'magnetization_mean': 'magnetization_err',
            'abs_magnetization_mean': 'abs_magnetization_err',
        }

        # Get theoretical Tc if available
        Tc = None
        if show_Tc:
            from ising_toolkit.utils.constants import CRITICAL_TEMP_2D, CRITICAL_TEMP_3D
            model_name = self.model_class.__name__.lower()
            if '2d' in model_name:
                Tc = CRITICAL_TEMP_2D
            elif '3d' in model_name:
                Tc = CRITICAL_TEMP_3D

        # Create figure
        n_plots = len(observables)
        if ax is None:
            if n_plots == 1:
                fig, axes = plt.subplots(1, 1, figsize=figsize)
                axes = [axes]
            else:
                n_cols = 2
                n_rows = (n_plots + 1) // 2
                fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
                axes = axes.flatten()
        else:
            if isinstance(ax, (list, np.ndarray)):
                axes = ax
            else:
                axes = [ax]
            fig = axes[0].get_figure()

        # Plot each observable
        for i, obs in enumerate(observables):
            if i >= len(axes):
                break

            ax = axes[i]
            values = np.array([r[obs] for r in self._results_list])

            # Check for error bars
            err_key = error_keys.get(obs)
            if err_key and err_key in self._results_list[0]:
                errors = np.array([r[err_key] for r in self._results_list])
                ax.errorbar(temps, values, yerr=errors, fmt='o-', capsize=3, markersize=4)
            else:
                ax.plot(temps, values, 'o-', markersize=4)

            # Add Tc line
            if Tc is not None and min(temps) < Tc < max(temps):
                ax.axvline(Tc, color='red', linestyle='--', alpha=0.5, label=f'Tc = {Tc:.3f}')
                ax.legend(loc='best', fontsize=9)

            # Labels
            ylabel = display_names.get(obs, obs)
            ax.set_xlabel('Temperature (T)')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for i in range(len(observables), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')

        return fig, axes[:len(observables)]

    def find_critical_temperature(
        self,
        method: str = 'susceptibility',
    ) -> float:
        """Estimate critical temperature from the data.

        Parameters
        ----------
        method : str, optional
            Method to use:
            - 'susceptibility': Temperature of maximum susceptibility
            - 'heat_capacity': Temperature of maximum heat capacity
            - 'binder': Where Binder cumulant has steepest slope
            Default is 'susceptibility'.

        Returns
        -------
        float
            Estimated critical temperature.

        Notes
        -----
        For accurate Tc estimation, use finite-size scaling with
        multiple system sizes. This method provides a rough estimate.
        """
        if self._results_list is None:
            raise RuntimeError("No results available. Run the sweep first.")

        temps = np.array([r['temperature'] for r in self._results_list])

        if method == 'susceptibility':
            values = np.array([r['susceptibility'] for r in self._results_list])
            idx = np.argmax(values)
        elif method == 'heat_capacity':
            values = np.array([r['heat_capacity'] for r in self._results_list])
            idx = np.argmax(values)
        elif method == 'binder':
            values = np.array([r['binder_cumulant'] for r in self._results_list])
            # Find steepest slope (largest |dU/dT|)
            slopes = np.abs(np.gradient(values, temps))
            idx = np.argmax(slopes)
        else:
            raise ValueError(f"Unknown method: {method}")

        return float(temps[idx])

    def save_results(self, path: str, format: str = 'csv'):
        """Save results to file.

        Parameters
        ----------
        path : str
            Output file path.
        format : str, optional
            Output format: 'csv', 'json', or 'hdf5'. Default is 'csv'.
        """
        if self._results_list is None:
            raise RuntimeError("No results available. Run the sweep first.")

        if format == 'csv':
            if HAS_PANDAS:
                self._results_df.to_csv(path, index=False)
            else:
                import csv
                with open(path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self._results_list[0].keys())
                    writer.writeheader()
                    writer.writerows(self._results_list)

        elif format == 'json':
            import json
            with open(path, 'w') as f:
                json.dump(self._results_list, f, indent=2)

        elif format == 'hdf5':
            if not HAS_PANDAS:
                raise ImportError("pandas required for HDF5 export")
            self._results_df.to_hdf(path, key='sweep_results', mode='w')

        else:
            raise ValueError(f"Unknown format: {format}")


def _run_size_sweep(args: tuple) -> tuple:
    """Worker function for parallel finite-size scaling.

    Parameters
    ----------
    args : tuple
        (model_class, size, temperatures, n_steps, algorithm, seed)

    Returns
    -------
    tuple
        (size, results_list)
    """
    model_class, size, temperatures, n_steps, algorithm, seed = args

    sweep = TemperatureSweep(
        model_class=model_class,
        size=size,
        temperatures=temperatures,
        n_steps=n_steps,
        algorithm=algorithm,
        seed=seed,
    )

    # Run sequentially within each size (parallelism is at size level)
    results = sweep.run(n_workers=1, progress=False)

    # Return as list for pickling
    if HAS_PANDAS:
        return (size, sweep._results_list)
    else:
        return (size, results)


def run_finite_size_scaling(
    model_class: Type[IsingModel],
    sizes: List[int],
    temperatures: Union[List[float], np.ndarray],
    n_steps: int,
    algorithm: str = 'wolff',
    n_workers: int = 1,
    parallel_sizes: bool = False,
    seed: Optional[int] = None,
    progress: bool = True,
):
    """Run temperature sweeps for multiple system sizes.

    This is useful for finite-size scaling analysis to accurately
    determine critical exponents and the critical temperature.

    Parameters
    ----------
    model_class : type
        The Ising model class to use.
    sizes : list of int
        System sizes to simulate.
    temperatures : array-like
        Temperatures to simulate.
    n_steps : int
        Number of Monte Carlo steps per simulation.
    algorithm : str, optional
        Sampling algorithm. Default is 'wolff'.
    n_workers : int, optional
        Number of parallel workers for temperature sweep. Default is 1.
    parallel_sizes : bool, optional
        If True, run different sizes in parallel instead of temperatures.
        Useful when you have many sizes but few temperatures. Default is False.
    seed : int, optional
        Random seed for reproducibility.
    progress : bool, optional
        Show progress. Default is True.

    Returns
    -------
    dict
        Dictionary mapping size -> DataFrame of results.

    Examples
    --------
    >>> sizes = [8, 16, 32, 64]
    >>> temps = np.linspace(2.0, 2.5, 21)
    >>> results = run_finite_size_scaling(
    ...     Ising2D, sizes, temps, n_steps=10000, n_workers=4
    ... )
    >>> # Binder cumulant crossing gives Tc
    >>> for L, df in results.items():
    ...     plt.plot(df['temperature'], df['binder_cumulant'], label=f'L={L}')

    Notes
    -----
    There are two parallelization strategies:
    1. parallel_sizes=False (default): Run each size sequentially, but
       parallelize temperatures within each size. Best when you have
       many temperatures.
    2. parallel_sizes=True: Run different sizes in parallel. Best when
       you have many sizes but few temperatures.
    """
    results = {}

    if parallel_sizes and n_workers > 1:
        # Parallelize across sizes
        args_list = [
            (model_class, size, temperatures, n_steps, algorithm, seed)
            for size in sizes
        ]

        size_results = parallel_map(
            func=_run_size_sweep,
            items=args_list,
            n_workers=min(n_workers, len(sizes)),
            progress=progress,
            desc="Finite-size scaling",
        )

        for size, result_list in size_results:
            if HAS_PANDAS:
                results[size] = pd.DataFrame(result_list)
            else:
                results[size] = result_list
    else:
        # Sequential sizes, parallel temperatures
        for size in sizes:
            if progress:
                print(f"Running L={size}...")

            sweep = TemperatureSweep(
                model_class=model_class,
                size=size,
                temperatures=temperatures,
                n_steps=n_steps,
                algorithm=algorithm,
                seed=seed,
            )

            df = sweep.run(n_workers=n_workers, progress=progress)
            results[size] = df

    return results
