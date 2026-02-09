"""Finite-size scaling analysis for critical phenomena."""

from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
from scipy import optimize  # noqa: F401

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ising_toolkit.models.base import IsingModel
from ising_toolkit.analysis.sweep import TemperatureSweep


class FiniteSizeScaling:
    """Finite-size scaling analysis tools.

    Finite-size scaling is a powerful technique for studying phase
    transitions in finite systems. Near the critical point, observables
    follow scaling laws that depend on the system size L:

    - Magnetization: M(T, L) = L^(-β/ν) * f_M((T-Tc) * L^(1/ν))
    - Susceptibility: χ(T, L) = L^(γ/ν) * f_χ((T-Tc) * L^(1/ν))
    - Binder cumulant: U(T, L) = f_U((T-Tc) * L^(1/ν))

    The Binder cumulant is special because it's size-independent at Tc,
    meaning curves for different L cross at the critical temperature.

    Parameters
    ----------
    model_class : type
        The Ising model class (Ising1D, Ising2D, or Ising3D).
    sizes : list of int
        System sizes to simulate.
    temperatures : array-like
        Temperature range to sweep.
    n_steps : int
        Number of Monte Carlo steps per simulation.
    equilibration : int, optional
        Equilibration steps. Default is n_steps // 10.
    algorithm : str, optional
        Sampling algorithm ('metropolis' or 'wolff'). Default is 'metropolis'.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from ising_toolkit.models import Ising2D
    >>> from ising_toolkit.analysis import FiniteSizeScaling
    >>>
    >>> # Setup finite-size scaling analysis
    >>> sizes = [8, 16, 32, 64]
    >>> temps = np.linspace(2.0, 2.5, 21)
    >>>
    >>> fss = FiniteSizeScaling(
    ...     model_class=Ising2D,
    ...     sizes=sizes,
    ...     temperatures=temps,
    ...     n_steps=10000,
    ...     algorithm='wolff'
    ... )
    >>>
    >>> # Run all simulations
    >>> df = fss.run(n_workers=4)
    >>>
    >>> # Find critical temperature from Binder crossing
    >>> Tc, Tc_err = fss.find_critical_temperature(method='binder')
    >>> print(f"Tc = {Tc:.4f} ± {Tc_err:.4f}")
    >>>
    >>> # Extract critical exponents
    >>> beta_nu, err = fss.extract_exponent_beta(Tc)
    >>> gamma_nu, err = fss.extract_exponent_gamma(Tc)

    Notes
    -----
    For the 2D Ising model, the exact critical exponents are:
    - β = 1/8, γ = 7/4, ν = 1, α = 0 (log)
    - β/ν = 1/8, γ/ν = 7/4

    For accurate results:
    - Use the Wolff algorithm near Tc (reduces critical slowing down)
    - Use sufficiently large system sizes (L ≥ 16)
    - Use enough temperatures near Tc for good resolution
    - Run long enough simulations for good statistics
    """

    def __init__(
        self,
        model_class: Type[IsingModel],
        sizes: List[int],
        temperatures: Union[List[float], np.ndarray],
        n_steps: int,
        equilibration: Optional[int] = None,
        algorithm: str = 'metropolis',
        seed: Optional[int] = None,
    ):
        self.model_class = model_class
        self.sizes = sorted(sizes)
        self.temperatures = np.asarray(temperatures)
        self.n_steps = n_steps
        self.equilibration = equilibration
        self.algorithm = algorithm
        self.seed = seed

        # Results storage
        self._results_df = None
        self._results_by_size: Dict[int, List[dict]] = {}

    def run(
        self,
        n_workers: int = 1,
        progress: bool = True,
    ):
        """Run temperature sweeps for all system sizes.

        Parameters
        ----------
        n_workers : int, optional
            Number of parallel workers per sweep. Default is 1.
        progress : bool, optional
            Show progress. Default is True.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with 'size' column added.
            If pandas is not available, returns list of dicts.
        """
        all_results = []

        for size in self.sizes:
            if progress:
                print(f"Running L = {size}...")

            sweep = TemperatureSweep(
                model_class=self.model_class,
                size=size,
                temperatures=self.temperatures,
                n_steps=self.n_steps,
                equilibration=self.equilibration,
                algorithm=self.algorithm,
                seed=self.seed,
            )

            sweep.run(n_workers=n_workers, progress=progress)

            # Add size to each result
            for result in sweep._results_list:
                result_with_size = result.copy()
                result_with_size['size'] = size
                all_results.append(result_with_size)

            self._results_by_size[size] = sweep._results_list

        if HAS_PANDAS:
            self._results_df = pd.DataFrame(all_results)
            return self._results_df
        else:
            return all_results

    @property
    def results(self):
        """Get results DataFrame."""
        return self._results_df

    def _get_observable_at_temp(
        self,
        size: int,
        temperature: float,
        observable: str,
    ) -> float:
        """Get observable value at specific size and temperature."""
        if size not in self._results_by_size:
            raise ValueError(f"No results for size {size}")

        results = self._results_by_size[size]
        temps = np.array([r['temperature'] for r in results])
        values = np.array([r[observable] for r in results])

        # Linear interpolation
        return float(np.interp(temperature, temps, values))

    def _get_observable_vs_temp(
        self,
        size: int,
        observable: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get temperature and observable arrays for a size."""
        if size not in self._results_by_size:
            raise ValueError(f"No results for size {size}")

        results = self._results_by_size[size]
        temps = np.array([r['temperature'] for r in results])
        values = np.array([r[observable] for r in results])

        # Sort by temperature
        idx = np.argsort(temps)
        return temps[idx], values[idx]

    def find_critical_temperature(
        self,
        method: str = 'binder',
        sizes: Optional[List[int]] = None,
    ) -> Tuple[float, float]:
        """Find critical temperature using finite-size scaling.

        Parameters
        ----------
        method : str, optional
            Method to use:
            - 'binder': Find crossing point of Binder cumulant curves
            - 'susceptibility': Find average peak position of χ
            Default is 'binder'.
        sizes : list of int, optional
            Which sizes to use. Default is all sizes.

        Returns
        -------
        Tc : float
            Estimated critical temperature.
        Tc_error : float
            Estimated uncertainty in Tc.

        Notes
        -----
        The 'binder' method finds where Binder cumulant curves for
        different sizes cross. This is the most accurate method as
        U is size-independent at Tc.

        The 'susceptibility' method finds the peak of χ(T) and
        averages over sizes. Less accurate but simpler.
        """
        if not self._results_by_size:
            raise RuntimeError("No results available. Run first.")

        if sizes is None:
            sizes = self.sizes

        if len(sizes) < 2:
            raise ValueError("Need at least 2 sizes for FSS analysis")

        if method == 'binder':
            return self._find_Tc_binder_crossing(sizes)
        elif method == 'susceptibility':
            return self._find_Tc_susceptibility_peak(sizes)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _find_Tc_binder_crossing(
        self,
        sizes: List[int],
    ) -> Tuple[float, float]:
        """Find Tc from Binder cumulant crossing."""
        crossings = []

        # Find pairwise crossings
        for i, L1 in enumerate(sizes[:-1]):
            for L2 in sizes[i + 1:]:
                temps1, U1 = self._get_observable_vs_temp(L1, 'binder_cumulant')
                temps2, U2 = self._get_observable_vs_temp(L2, 'binder_cumulant')

                # Interpolate to common temperature grid
                temp_min = max(temps1.min(), temps2.min())
                temp_max = min(temps1.max(), temps2.max())
                temp_grid = np.linspace(temp_min, temp_max, 200)

                U1_interp = np.interp(temp_grid, temps1, U1)
                U2_interp = np.interp(temp_grid, temps2, U2)

                # Find crossing (where U1 - U2 changes sign)
                diff = U1_interp - U2_interp
                sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

                if len(sign_changes) > 0:
                    # Take first crossing (usually the relevant one)
                    idx = sign_changes[0]
                    # Linear interpolation for precise crossing
                    t1, t2 = temp_grid[idx], temp_grid[idx + 1]
                    d1, d2 = diff[idx], diff[idx + 1]
                    Tc_crossing = t1 - d1 * (t2 - t1) / (d2 - d1)
                    crossings.append(Tc_crossing)

        if len(crossings) == 0:
            # Fallback: use susceptibility peak
            return self._find_Tc_susceptibility_peak(sizes)

        Tc = float(np.mean(crossings))
        Tc_err = float(np.std(crossings)) if len(crossings) > 1 else 0.05

        return Tc, Tc_err

    def _find_Tc_susceptibility_peak(
        self,
        sizes: List[int],
    ) -> Tuple[float, float]:
        """Find Tc from susceptibility peak."""
        Tc_values = []

        for size in sizes:
            temps, chi = self._get_observable_vs_temp(size, 'susceptibility')
            idx_max = np.argmax(chi)
            Tc_values.append(temps[idx_max])

        Tc = float(np.mean(Tc_values))
        Tc_err = float(np.std(Tc_values)) if len(Tc_values) > 1 else 0.1

        return Tc, Tc_err

    def extract_exponent_beta(
        self,
        Tc: float,
        sizes: Optional[List[int]] = None,
    ) -> Tuple[float, float]:
        """Extract β/ν from magnetization scaling at Tc.

        At the critical point:
            M(Tc, L) ~ L^(-β/ν)

        Parameters
        ----------
        Tc : float
            Critical temperature.
        sizes : list of int, optional
            Which sizes to use. Default is all sizes.

        Returns
        -------
        beta_nu : float
            Estimated β/ν exponent ratio.
        error : float
            Fitting error estimate.

        Notes
        -----
        For 2D Ising: β/ν = 1/8 = 0.125
        For 3D Ising: β/ν ≈ 0.518
        """
        if sizes is None:
            sizes = self.sizes

        log_L = np.log(sizes)
        log_M = []

        for size in sizes:
            M = self._get_observable_at_temp(size, Tc, 'abs_magnetization_mean')
            log_M.append(np.log(M))

        log_M = np.array(log_M)

        # Linear fit: log(M) = -β/ν * log(L) + const
        coeffs, cov = np.polyfit(log_L, log_M, 1, cov=True)
        beta_nu = -coeffs[0]
        error = np.sqrt(cov[0, 0])

        return float(beta_nu), float(error)

    def extract_exponent_gamma(
        self,
        Tc: float,
        sizes: Optional[List[int]] = None,
    ) -> Tuple[float, float]:
        """Extract γ/ν from susceptibility scaling at Tc.

        At the critical point:
            χ(Tc, L) ~ L^(γ/ν)

        Parameters
        ----------
        Tc : float
            Critical temperature.
        sizes : list of int, optional
            Which sizes to use. Default is all sizes.

        Returns
        -------
        gamma_nu : float
            Estimated γ/ν exponent ratio.
        error : float
            Fitting error estimate.

        Notes
        -----
        For 2D Ising: γ/ν = 7/4 = 1.75
        For 3D Ising: γ/ν ≈ 1.963
        """
        if sizes is None:
            sizes = self.sizes

        log_L = np.log(sizes)
        log_chi = []

        for size in sizes:
            chi = self._get_observable_at_temp(size, Tc, 'susceptibility')
            log_chi.append(np.log(chi))

        log_chi = np.array(log_chi)

        # Linear fit: log(χ) = γ/ν * log(L) + const
        coeffs, cov = np.polyfit(log_L, log_chi, 1, cov=True)
        gamma_nu = coeffs[0]
        error = np.sqrt(cov[0, 0])

        return float(gamma_nu), float(error)

    def extract_exponent_nu(
        self,
        Tc: float,
        sizes: Optional[List[int]] = None,
    ) -> Tuple[float, float]:
        """Extract 1/ν from Binder cumulant derivative scaling.

        The derivative of U at Tc scales as:
            dU/dT|_{Tc} ~ L^(1/ν)

        Parameters
        ----------
        Tc : float
            Critical temperature.
        sizes : list of int, optional
            Which sizes to use. Default is all sizes.

        Returns
        -------
        one_over_nu : float
            Estimated 1/ν exponent.
        error : float
            Fitting error estimate.

        Notes
        -----
        For 2D Ising: 1/ν = 1
        For 3D Ising: 1/ν ≈ 1.587
        """
        if sizes is None:
            sizes = self.sizes

        log_L = np.log(sizes)
        log_dU = []

        for size in sizes:
            temps, U = self._get_observable_vs_temp(size, 'binder_cumulant')

            # Compute derivative using central differences
            dU_dT = np.gradient(U, temps)

            # Get derivative at Tc
            dU_at_Tc = np.interp(Tc, temps, np.abs(dU_dT))
            log_dU.append(np.log(dU_at_Tc))

        log_dU = np.array(log_dU)

        # Linear fit: log(|dU/dT|) = (1/ν) * log(L) + const
        coeffs, cov = np.polyfit(log_L, log_dU, 1, cov=True)
        one_over_nu = coeffs[0]
        error = np.sqrt(cov[0, 0])

        return float(one_over_nu), float(error)

    def plot_binder_crossing(
        self,
        ax=None,
        save: Optional[str] = None,
        show_Tc: bool = True,
    ):
        """Plot Binder cumulant curves showing crossing at Tc.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        save : str, optional
            Path to save figure.
        show_Tc : bool, optional
            Show estimated Tc line. Default is True.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if not self._results_by_size:
            raise RuntimeError("No results available. Run first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot Binder cumulant for each size
        for size in self.sizes:
            temps, U = self._get_observable_vs_temp(size, 'binder_cumulant')
            ax.plot(temps, U, 'o-', markersize=4, label=f'L = {size}')

        # Show estimated Tc
        if show_Tc and len(self.sizes) >= 2:
            try:
                Tc, Tc_err = self.find_critical_temperature(method='binder')
                ax.axvline(Tc, color='red', linestyle='--', alpha=0.7,
                          label=f'Tc = {Tc:.4f}')
            except Exception:
                pass

        ax.set_xlabel('Temperature (T)')
        ax.set_ylabel('Binder cumulant (U)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Reference lines
        ax.axhline(2/3, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')

        return ax

    def plot_scaling_collapse(
        self,
        observable: str,
        Tc: float,
        nu: float,
        exponent: float,
        ax=None,
        save: Optional[str] = None,
    ):
        """Plot data collapse for scaling function.

        If the scaling hypothesis is correct, plotting
            L^(-exponent) * O(T, L)  vs  (T - Tc) * L^(1/ν)
        should collapse all curves onto a single scaling function.

        Parameters
        ----------
        observable : str
            Observable to collapse ('abs_magnetization_mean' or 'susceptibility').
        Tc : float
            Critical temperature.
        nu : float
            Correlation length exponent ν.
        exponent : float
            Scaling exponent (β/ν for magnetization, -γ/ν for susceptibility).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        save : str, optional
            Path to save figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.

        Notes
        -----
        For magnetization collapse:
            M * L^(β/ν) vs (T - Tc) * L^(1/ν)

        For susceptibility collapse:
            χ * L^(-γ/ν) vs (T - Tc) * L^(1/ν)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if not self._results_by_size:
            raise RuntimeError("No results available. Run first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        for size in self.sizes:
            temps, values = self._get_observable_vs_temp(size, observable)

            # Scaling variables
            x = (temps - Tc) * (size ** (1 / nu))
            y = values * (size ** exponent)

            ax.plot(x, y, 'o', markersize=4, label=f'L = {size}')

        ax.set_xlabel(r'$(T - T_c) \cdot L^{1/\nu}$')
        ax.set_ylabel(f'Scaled {observable}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')

        return ax

    def plot_exponent_fit(
        self,
        observable: str,
        Tc: float,
        ax=None,
        save: Optional[str] = None,
    ):
        """Plot log-log fit for exponent extraction.

        Parameters
        ----------
        observable : str
            Which observable to fit:
            - 'magnetization': fits β/ν
            - 'susceptibility': fits γ/ν
        Tc : float
            Critical temperature.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        save : str, optional
            Path to save figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        sizes = np.array(self.sizes)
        log_L = np.log(sizes)

        if observable == 'magnetization':
            values = [self._get_observable_at_temp(L, Tc, 'abs_magnetization_mean')
                     for L in sizes]
            exponent, error = self.extract_exponent_beta(Tc)
            ylabel = r'$\langle |M| \rangle$'
            title = f'β/ν = {exponent:.3f} ± {error:.3f}'
        elif observable == 'susceptibility':
            values = [self._get_observable_at_temp(L, Tc, 'susceptibility')
                     for L in sizes]
            exponent, error = self.extract_exponent_gamma(Tc)
            ylabel = r'$\chi$'
            title = f'γ/ν = {exponent:.3f} ± {error:.3f}'
        else:
            raise ValueError(f"Unknown observable: {observable}")

        log_values = np.log(values)

        # Plot data
        ax.plot(log_L, log_values, 'o', markersize=8, label='Data')

        # Plot fit
        if observable == 'magnetization':
            slope = -exponent
        else:
            slope = exponent

        fit_y = slope * log_L + (log_values[0] - slope * log_L[0])
        ax.plot(log_L, fit_y, '--', label='Fit')

        ax.set_xlabel(r'$\ln(L)$')
        ax.set_ylabel(rf'$\ln({ylabel})$')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')

        return ax

    def summary(self, Tc: Optional[float] = None) -> dict:
        """Get summary of finite-size scaling analysis.

        Parameters
        ----------
        Tc : float, optional
            Critical temperature. If None, will be estimated.

        Returns
        -------
        dict
            Dictionary with all extracted quantities.
        """
        if not self._results_by_size:
            raise RuntimeError("No results available. Run first.")

        results = {}

        # Find Tc
        if Tc is None:
            Tc, Tc_err = self.find_critical_temperature(method='binder')
        else:
            Tc_err = 0.0

        results['Tc'] = Tc
        results['Tc_error'] = Tc_err

        # Extract exponents
        try:
            beta_nu, beta_nu_err = self.extract_exponent_beta(Tc)
            results['beta_nu'] = beta_nu
            results['beta_nu_error'] = beta_nu_err
        except Exception:
            pass

        try:
            gamma_nu, gamma_nu_err = self.extract_exponent_gamma(Tc)
            results['gamma_nu'] = gamma_nu
            results['gamma_nu_error'] = gamma_nu_err
        except Exception:
            pass

        try:
            one_over_nu, one_over_nu_err = self.extract_exponent_nu(Tc)
            results['one_over_nu'] = one_over_nu
            results['one_over_nu_error'] = one_over_nu_err

            # Derive individual exponents
            if one_over_nu > 0:
                nu = 1.0 / one_over_nu
                results['nu'] = nu

                if 'beta_nu' in results:
                    results['beta'] = results['beta_nu'] * nu

                if 'gamma_nu' in results:
                    results['gamma'] = results['gamma_nu'] * nu
        except Exception:
            pass

        return results
