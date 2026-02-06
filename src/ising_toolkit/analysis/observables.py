"""Functions for calculating physical observables from simulation data."""

from typing import Dict

import numpy as np

from ising_toolkit.io.results import SimulationResults


def calculate_heat_capacity(
    energy: np.ndarray,
    temperature: float,
    n_spins: int,
) -> float:
    """Calculate heat capacity from energy fluctuations.

    The heat capacity is computed using the fluctuation-dissipation theorem:
        C = (⟨E²⟩ - ⟨E⟩²) / (kB * T²)

    Parameters
    ----------
    energy : np.ndarray
        Energy time series from Monte Carlo simulation.
    temperature : float
        Temperature in units of J/kB.
    n_spins : int
        Total number of spins in the system.

    Returns
    -------
    float
        Heat capacity per spin, C/N.

    Notes
    -----
    In natural units where kB = 1, the formula simplifies to:
        C/N = (⟨E²⟩ - ⟨E⟩²) / (N * T²)

    The heat capacity diverges logarithmically at the critical point
    in the 2D Ising model (α = 0).

    Examples
    --------
    >>> energy = np.array([-100, -98, -102, -99, -101])
    >>> C = calculate_heat_capacity(energy, temperature=2.0, n_spins=100)
    """
    energy = np.asarray(energy)

    # Calculate variance: ⟨E²⟩ - ⟨E⟩²
    e_mean = np.mean(energy)
    e2_mean = np.mean(energy**2)
    variance = e2_mean - e_mean**2

    # Heat capacity per spin: C/N = Var(E) / (N * T²)
    heat_capacity_per_spin = variance / (n_spins * temperature**2)

    return float(heat_capacity_per_spin)


def calculate_susceptibility(
    magnetization: np.ndarray,
    temperature: float,
    n_spins: int,
) -> float:
    """Calculate magnetic susceptibility from magnetization fluctuations.

    The susceptibility is computed using the fluctuation-dissipation theorem:
        χ = (⟨M²⟩ - ⟨M⟩²) / (kB * T)

    Parameters
    ----------
    magnetization : np.ndarray
        Magnetization time series from Monte Carlo simulation.
    temperature : float
        Temperature in units of J/kB.
    n_spins : int
        Total number of spins in the system.

    Returns
    -------
    float
        Magnetic susceptibility per spin, χ/N.

    Notes
    -----
    In natural units where kB = 1, the formula simplifies to:
        χ/N = (⟨M²⟩ - ⟨M⟩²) / (N * T)

    For simulations with zero external field, ⟨M⟩ ≈ 0 above Tc, so
    the susceptibility is approximately ⟨M²⟩ / (N * T).

    The susceptibility diverges at the critical point with exponent γ:
        χ ~ |T - Tc|^(-γ)

    Examples
    --------
    >>> magnetization = np.array([50, -48, 52, -50, 49])
    >>> chi = calculate_susceptibility(magnetization, temperature=2.0, n_spins=100)
    """
    magnetization = np.asarray(magnetization)

    # Calculate variance: ⟨M²⟩ - ⟨M⟩²
    m_mean = np.mean(magnetization)
    m2_mean = np.mean(magnetization**2)
    variance = m2_mean - m_mean**2

    # Susceptibility per spin: χ/N = Var(M) / (N * T)
    susceptibility_per_spin = variance / (n_spins * temperature)

    return float(susceptibility_per_spin)


def calculate_binder_cumulant(
    magnetization: np.ndarray,
    n_spins: int,
) -> float:
    """Calculate the Binder cumulant (fourth-order cumulant).

    The Binder cumulant is defined as:
        U = 1 - ⟨m⁴⟩ / (3⟨m²⟩²)

    where m = M/N is the magnetization per spin.

    Parameters
    ----------
    magnetization : np.ndarray
        Magnetization time series from Monte Carlo simulation.
    n_spins : int
        Total number of spins in the system.

    Returns
    -------
    float
        Binder cumulant U.

    Notes
    -----
    The Binder cumulant is useful for locating phase transitions:
    - U → 2/3 in the ordered phase (T < Tc)
    - U → 0 in the disordered phase (T > Tc)
    - Curves for different system sizes cross at Tc

    This quantity is size-independent at the critical point, making
    it useful for finite-size scaling analysis.

    Examples
    --------
    >>> magnetization = np.array([100, 98, 102, 99, 101])  # Ordered
    >>> U = calculate_binder_cumulant(magnetization, n_spins=100)
    >>> # U should be close to 2/3
    """
    magnetization = np.asarray(magnetization)

    # Magnetization per spin
    m = magnetization / n_spins

    # Calculate moments
    m2_mean = np.mean(m**2)
    m4_mean = np.mean(m**4)

    # Avoid division by zero
    if m2_mean == 0:
        return 0.0

    # Binder cumulant: U = 1 - ⟨m⁴⟩ / (3⟨m²⟩²)
    binder = 1.0 - m4_mean / (3.0 * m2_mean**2)

    return float(binder)


def calculate_correlation_time(
    observable: np.ndarray,
    max_lag: int = None,
) -> float:
    """Estimate integrated autocorrelation time.

    Parameters
    ----------
    observable : np.ndarray
        Time series of an observable.
    max_lag : int, optional
        Maximum lag to consider. Default is len(observable) // 4.

    Returns
    -------
    float
        Integrated autocorrelation time τ.

    Notes
    -----
    The integrated autocorrelation time is:
        τ = 1/2 + Σ_{t=1}^{∞} ρ(t)

    where ρ(t) is the normalized autocorrelation function.
    The effective number of independent samples is N / (2τ).
    """
    observable = np.asarray(observable)
    n = len(observable)

    if max_lag is None:
        max_lag = n // 4

    # Subtract mean
    x = observable - np.mean(observable)

    # Variance
    var = np.var(x)
    if var == 0:
        return 1.0

    # Compute autocorrelation function
    tau = 0.5
    for t in range(1, max_lag):
        # Autocorrelation at lag t
        autocorr = np.mean(x[:-t] * x[t:]) / var

        # Stop if autocorrelation becomes negative or very small
        if autocorr <= 0.0:
            break

        tau += autocorr

    return float(tau)


def calculate_all_observables(results: SimulationResults) -> Dict[str, float]:
    """Calculate all standard observables from simulation results.

    Parameters
    ----------
    results : SimulationResults
        Simulation results containing energy and magnetization time series.

    Returns
    -------
    dict
        Dictionary containing:
        - energy_mean: Mean energy per spin
        - energy_std: Standard deviation of energy per spin
        - magnetization_mean: Mean magnetization per spin
        - magnetization_std: Standard deviation of magnetization per spin
        - abs_magnetization_mean: Mean absolute magnetization per spin
        - abs_magnetization_std: Standard deviation of absolute magnetization per spin
        - heat_capacity: Heat capacity per spin
        - susceptibility: Magnetic susceptibility per spin
        - binder_cumulant: Binder cumulant

    Examples
    --------
    >>> results = sampler.run(n_steps=10000)
    >>> observables = calculate_all_observables(results)
    >>> print(f"Heat capacity: {observables['heat_capacity']:.4f}")
    """
    # Get parameters from metadata
    n_spins = results.metadata.get("n_spins", len(results.energy))
    temperature = results.metadata.get("temperature", 1.0)

    # Energy statistics (per spin)
    energy_per_spin = results.energy / n_spins
    energy_mean = float(np.mean(energy_per_spin))
    energy_std = float(np.std(energy_per_spin, ddof=1))

    # Magnetization statistics (per spin)
    mag_per_spin = results.magnetization / n_spins
    magnetization_mean = float(np.mean(mag_per_spin))
    magnetization_std = float(np.std(mag_per_spin, ddof=1))

    # Absolute magnetization statistics (per spin)
    abs_mag_per_spin = np.abs(mag_per_spin)
    abs_magnetization_mean = float(np.mean(abs_mag_per_spin))
    abs_magnetization_std = float(np.std(abs_mag_per_spin, ddof=1))

    # Thermodynamic quantities
    heat_capacity = calculate_heat_capacity(
        results.energy, temperature, n_spins
    )
    susceptibility = calculate_susceptibility(
        results.magnetization, temperature, n_spins
    )
    binder_cumulant = calculate_binder_cumulant(
        results.magnetization, n_spins
    )

    return {
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "magnetization_mean": magnetization_mean,
        "magnetization_std": magnetization_std,
        "abs_magnetization_mean": abs_magnetization_mean,
        "abs_magnetization_std": abs_magnetization_std,
        "heat_capacity": heat_capacity,
        "susceptibility": susceptibility,
        "binder_cumulant": binder_cumulant,
    }
