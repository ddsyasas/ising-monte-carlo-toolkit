"""Analysis tools for simulation results."""

from ising_toolkit.analysis.observables import (
    calculate_heat_capacity,
    calculate_susceptibility,
    calculate_binder_cumulant,
    calculate_correlation_time,
    calculate_all_observables,
)
from ising_toolkit.analysis.statistics import (
    bootstrap_error,
    bootstrap_confidence_interval,
    bootstrap_mean_error,
    jackknife_error,
    blocking_error,
    autocorrelation_function,
    autocorrelation_function_fft,
    integrated_autocorrelation_time,
    effective_sample_size,
    generate_ar1,
)

__all__ = [
    # Observables
    "calculate_heat_capacity",
    "calculate_susceptibility",
    "calculate_binder_cumulant",
    "calculate_correlation_time",
    "calculate_all_observables",
    # Statistical error estimation
    "bootstrap_error",
    "bootstrap_confidence_interval",
    "bootstrap_mean_error",
    "jackknife_error",
    "blocking_error",
    # Autocorrelation analysis
    "autocorrelation_function",
    "autocorrelation_function_fft",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "generate_ar1",
]
