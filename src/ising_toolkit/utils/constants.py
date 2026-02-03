"""Physical and numerical constants for Ising model simulations.

This module defines physical constants, exact analytical results, and
default simulation parameters used throughout the Ising Monte Carlo toolkit.

All physical constants are expressed in natural units where:
- J (coupling constant) = 1
- kB (Boltzmann constant) = 1

This means temperatures are measured in units of J/kB.
"""

import numpy as np

__all__ = [
    # Physical constants
    "DEFAULT_COUPLING",
    "DEFAULT_EXTERNAL_FIELD",
    # 2D critical values
    "CRITICAL_TEMP_2D",
    "CRITICAL_EXPONENT_BETA_2D",
    "CRITICAL_EXPONENT_GAMMA_2D",
    "CRITICAL_EXPONENT_NU_2D",
    "CRITICAL_EXPONENT_ALPHA_2D",
    # 3D critical values
    "CRITICAL_TEMP_3D",
    "CRITICAL_EXPONENT_BETA_3D",
    "CRITICAL_EXPONENT_GAMMA_3D",
    "CRITICAL_EXPONENT_NU_3D",
    # Simulation defaults
    "DEFAULT_MC_STEPS",
    "DEFAULT_EQUILIBRATION",
    "DEFAULT_MEASUREMENT_INTERVAL",
    # Valid values
    "VALID_BOUNDARIES",
    "VALID_INITIAL_STATES",
    "VALID_ALGORITHMS",
]

# =============================================================================
# Physical Constants (in natural units where J=1, kB=1)
# =============================================================================

DEFAULT_COUPLING: float = 1.0
"""Default nearest-neighbor coupling constant J.

The coupling constant determines the strength of interaction between
neighboring spins. Positive J favors ferromagnetic alignment (parallel spins),
while negative J favors antiferromagnetic alignment (antiparallel spins).
"""

DEFAULT_EXTERNAL_FIELD: float = 0.0
"""Default external magnetic field h.

The external field couples linearly to each spin, adding a term -h*s_i
to the Hamiltonian. Zero field is used to study spontaneous magnetization.
"""

# =============================================================================
# 2D Ising Model Exact Results
# =============================================================================

CRITICAL_TEMP_2D: float = 2.0 / np.log(1 + np.sqrt(2))
"""Exact critical temperature for the 2D square lattice Ising model.

Derived by Lars Onsager in 1944. This is one of the most celebrated
exact results in statistical mechanics.

Value: Tc = 2/ln(1 + sqrt(2)) ≈ 2.269185...

At this temperature, the system undergoes a continuous phase transition
from the disordered (paramagnetic) to the ordered (ferromagnetic) phase.
"""

CRITICAL_EXPONENT_BETA_2D: float = 0.125
"""Critical exponent beta for the 2D Ising model (exact: 1/8).

Describes how the order parameter (magnetization) vanishes as T -> Tc:
    M ~ (Tc - T)^beta  for T < Tc

The exact value beta = 1/8 was derived analytically.
"""

CRITICAL_EXPONENT_GAMMA_2D: float = 1.75
"""Critical exponent gamma for the 2D Ising model (exact: 7/4).

Describes the divergence of magnetic susceptibility at Tc:
    chi ~ |T - Tc|^(-gamma)

The exact value gamma = 7/4 was derived analytically.
"""

CRITICAL_EXPONENT_NU_2D: float = 1.0
"""Critical exponent nu for the 2D Ising model (exact: 1).

Describes the divergence of correlation length at Tc:
    xi ~ |T - Tc|^(-nu)

The exact value nu = 1 was derived analytically.
"""

CRITICAL_EXPONENT_ALPHA_2D: float = 0.0
"""Critical exponent alpha for the 2D Ising model (exact: 0, logarithmic).

Describes the divergence of specific heat at Tc:
    C ~ |T - Tc|^(-alpha)

For the 2D Ising model, alpha = 0 indicates a logarithmic divergence:
    C ~ -ln|T - Tc|

This is a special case where the power law is replaced by a logarithm.
"""

# =============================================================================
# 3D Ising Model Numerical Results
# =============================================================================

CRITICAL_TEMP_3D: float = 4.511
"""Critical temperature for the 3D simple cubic Ising model.

Unlike the 2D case, no exact solution exists for the 3D Ising model.
This value is determined from high-precision Monte Carlo simulations
and series expansions.

Value: Tc ≈ 4.511 (in units of J/kB)

More precise estimates: Tc/J = 4.5115232(16)
"""

CRITICAL_EXPONENT_BETA_3D: float = 0.326
"""Critical exponent beta for the 3D Ising model.

Numerical estimate from Monte Carlo and field theory calculations.
Describes magnetization: M ~ (Tc - T)^beta

More precise estimate: beta = 0.32653(10)
"""

CRITICAL_EXPONENT_GAMMA_3D: float = 1.237
"""Critical exponent gamma for the 3D Ising model.

Numerical estimate from Monte Carlo and field theory calculations.
Describes susceptibility divergence: chi ~ |T - Tc|^(-gamma)

More precise estimate: gamma = 1.2372(5)
"""

CRITICAL_EXPONENT_NU_3D: float = 0.630
"""Critical exponent nu for the 3D Ising model.

Numerical estimate from Monte Carlo and field theory calculations.
Describes correlation length divergence: xi ~ |T - Tc|^(-nu)

More precise estimate: nu = 0.62999(5)
"""

# =============================================================================
# Simulation Defaults
# =============================================================================

DEFAULT_MC_STEPS: int = 100000
"""Default number of Monte Carlo steps (sweeps) for production runs.

One Monte Carlo step typically consists of N attempted spin flips,
where N is the total number of spins in the system. This ensures
that on average each spin has one opportunity to flip per step.
"""

DEFAULT_EQUILIBRATION: int = 10000
"""Default number of equilibration steps before measurements begin.

Equilibration (thermalization) steps are discarded to ensure the
system has relaxed from its initial configuration to thermal
equilibrium at the target temperature.

The required equilibration time depends on system size and temperature,
especially near the critical point where critical slowing down occurs.
"""

DEFAULT_MEASUREMENT_INTERVAL: int = 10
"""Default interval between measurements in Monte Carlo steps.

Measurements taken at every step are highly correlated. Taking
measurements at intervals helps reduce autocorrelation and
provides more statistically independent samples.

The optimal interval depends on the autocorrelation time of the
observable being measured.
"""

# =============================================================================
# Valid Configuration Values
# =============================================================================

VALID_BOUNDARIES: tuple[str, ...] = ("periodic", "fixed")
"""Valid boundary condition types.

- 'periodic': Periodic boundary conditions (toroidal topology).
  Spins on opposite edges are neighbors, eliminating edge effects.
  Most commonly used for studying bulk properties.

- 'fixed': Fixed (open) boundary conditions.
  Spins at the boundary have fewer neighbors.
  Useful for studying surface effects or finite-size systems.
"""

VALID_INITIAL_STATES: tuple[str, ...] = ("random", "up", "down", "checkerboard")
"""Valid initial spin configurations.

- 'random': Each spin randomly set to +1 or -1 with equal probability.
  Good general-purpose initialization.

- 'up': All spins initialized to +1 (fully magnetized up).
  Useful for studying relaxation from ordered state.

- 'down': All spins initialized to -1 (fully magnetized down).
  Equivalent to 'up' by symmetry.

- 'checkerboard': Alternating +1/-1 pattern (Néel state).
  Ground state for antiferromagnetic coupling (J < 0).
"""

VALID_ALGORITHMS: tuple[str, ...] = ("metropolis", "wolff")
"""Valid Monte Carlo sampling algorithms.

- 'metropolis': Single-spin-flip Metropolis algorithm.
  Simple and robust, but suffers from critical slowing down near Tc.
  Computational cost: O(N) per sweep.

- 'wolff': Wolff cluster algorithm.
  Flips clusters of aligned spins, dramatically reducing critical
  slowing down. Highly efficient near the critical point.
  Computational cost varies with cluster size.
"""
