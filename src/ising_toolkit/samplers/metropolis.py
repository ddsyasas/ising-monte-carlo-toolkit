"""Metropolis single-spin flip sampler."""

from typing import Optional

import numpy as np

from ising_toolkit.samplers.base import Sampler
from ising_toolkit.models.base import IsingModel


class MetropolisSampler(Sampler):
    """Metropolis single-spin flip Monte Carlo sampler.

    Implements the classic Metropolis algorithm for the Ising model.
    Each Monte Carlo step consists of N single-spin flip attempts,
    where N is the total number of spins in the system.

    The acceptance criterion is:
        - If ΔE ≤ 0: always accept
        - If ΔE > 0: accept with probability exp(-β * ΔE)

    Parameters
    ----------
    model : IsingModel
        The Ising model to simulate.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    model : IsingModel
        The Ising model being simulated.
    rng : np.random.Generator
        NumPy random number generator.

    Examples
    --------
    >>> from ising_toolkit.models import Ising2D
    >>> model = Ising2D(size=32, temperature=2.269)
    >>> model.initialize('random')
    >>> sampler = MetropolisSampler(model, seed=42)
    >>> results = sampler.run(n_steps=10000, equilibration=1000)
    >>> print(f"Mean energy: {results.energy_mean:.2f}")

    Notes
    -----
    The Metropolis algorithm is simple and robust but suffers from
    critical slowing down near the phase transition. For simulations
    near Tc, consider using cluster algorithms like Wolff.

    References
    ----------
    Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N., Teller, A.H.,
    and Teller, E. (1953). "Equation of State Calculations by Fast
    Computing Machines". Journal of Chemical Physics, 21(6), 1087-1092.
    """

    def __init__(self, model: IsingModel, seed: Optional[int] = None) -> None:
        """Initialize the Metropolis sampler.

        Parameters
        ----------
        model : IsingModel
            The Ising model to simulate.
        seed : int, optional
            Random seed for reproducibility.
        """
        super().__init__(model, seed)

        # Also seed the model's RNG for consistent site selection
        if seed is not None:
            self.model.set_seed(seed)

    def step(self) -> int:
        """Perform one Monte Carlo step (N spin flip attempts).

        One step consists of N attempted single-spin flips, where N
        is the number of spins. This ensures that on average each
        spin has one opportunity to flip per step.

        Returns
        -------
        int
            Number of accepted spin flips in this step.

        Notes
        -----
        Uses precomputed acceptance probabilities from the model
        when available for improved efficiency.
        """
        n_spins = self.model.n_spins
        accepted = 0

        # Get precomputed acceptance probabilities if available
        acceptance_probs = getattr(self.model, "acceptance_probs", {})

        for _ in range(n_spins):
            self.n_attempted += 1

            # Select random site
            site = self.model.random_site()

            # Calculate energy change
            dE = self.model.get_energy_change(site)

            # Metropolis acceptance criterion
            if dE <= 0:
                # Always accept moves that lower energy
                self.model.flip_spin(site)
                accepted += 1
                self.n_accepted += 1
            else:
                # Accept with probability exp(-beta * dE)
                # Use precomputed probability if available
                dE_int = int(round(dE / self.model.coupling))

                if dE_int in acceptance_probs:
                    accept_prob = acceptance_probs[dE_int]
                else:
                    accept_prob = np.exp(-self.model.beta * dE)

                if self.rng.random() < accept_prob:
                    self.model.flip_spin(site)
                    accepted += 1
                    self.n_accepted += 1

        return accepted

    def get_acceptance_rate(self) -> float:
        """Get the current acceptance rate.

        Returns
        -------
        float
            Fraction of attempted moves that were accepted.
            Returns 0.0 if no moves have been attempted.

        Notes
        -----
        The acceptance rate depends strongly on temperature:
        - High T: acceptance rate approaches 1 (most moves accepted)
        - Low T: acceptance rate is low (system is frozen)
        - Near Tc: intermediate acceptance rate
        """
        return self.acceptance_rate

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MetropolisSampler("
            f"model={self.model.__class__.__name__}, "
            f"T={self.model.temperature:.4f}, "
            f"seed={self.seed})"
        )
