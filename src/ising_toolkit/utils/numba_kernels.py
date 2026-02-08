"""
Numba-optimized kernels for Ising model Monte Carlo simulations.

This module provides JIT-compiled versions of the core simulation loops
for significant performance improvements over pure Python implementations.

Requirements:
    pip install numba

Usage:
    from ising_toolkit.utils.numba_kernels import metropolis_sweep_2d

    # Precompute acceptance probabilities
    acceptance_probs = precompute_acceptance_probs_2d(beta)

    # Run sweep
    n_accepted = metropolis_sweep_2d(spins, beta, acceptance_probs)
"""

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator for when numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args):
        return range(*args)


# =============================================================================
# Precomputation helpers
# =============================================================================

def precompute_acceptance_probs_2d(beta):
    """Precompute Metropolis acceptance probabilities for 2D Ising.

    For 2D Ising, dE can be -8, -4, 0, 4, 8.
    We only need probabilities for dE > 0, i.e., dE = 4, 8.

    Parameters
    ----------
    beta : float
        Inverse temperature (1/T)

    Returns
    -------
    ndarray of shape (3,)
        acceptance_probs[0] = exp(-4*beta) for dE = 4
        acceptance_probs[1] = exp(-8*beta) for dE = 8
        acceptance_probs[2] = 1.0 (unused, for safety)
    """
    return np.array([np.exp(-4 * beta), np.exp(-8 * beta), 1.0], dtype=np.float64)


def precompute_acceptance_probs_1d(beta):
    """Precompute Metropolis acceptance probabilities for 1D Ising.

    For 1D Ising, dE can be -4, 0, 4.
    We only need probability for dE = 4.

    Parameters
    ----------
    beta : float
        Inverse temperature (1/T)

    Returns
    -------
    ndarray of shape (2,)
        acceptance_probs[0] = exp(-4*beta) for dE = 4
        acceptance_probs[1] = 1.0 (unused, for safety)
    """
    return np.array([np.exp(-4 * beta), 1.0], dtype=np.float64)


def precompute_acceptance_probs_3d(beta):
    """Precompute Metropolis acceptance probabilities for 3D Ising.

    For 3D Ising with 6 neighbors, dE can be -12, -8, -4, 0, 4, 8, 12.
    We need probabilities for dE = 4, 8, 12.

    Parameters
    ----------
    beta : float
        Inverse temperature (1/T)

    Returns
    -------
    ndarray of shape (4,)
        acceptance_probs[0] = exp(-4*beta) for dE = 4
        acceptance_probs[1] = exp(-8*beta) for dE = 8
        acceptance_probs[2] = exp(-12*beta) for dE = 12
        acceptance_probs[3] = 1.0 (unused, for safety)
    """
    return np.array([
        np.exp(-4 * beta),
        np.exp(-8 * beta),
        np.exp(-12 * beta),
        1.0
    ], dtype=np.float64)


# =============================================================================
# 2D Ising Model Kernels
# =============================================================================

@njit(cache=True)
def _metropolis_sweep_2d(spins, beta, acceptance_probs):
    """Numba-optimized Metropolis sweep for 2D Ising.

    Performs L*L single-spin flip attempts using the Metropolis algorithm.

    Parameters
    ----------
    spins : ndarray (L, L) of int8
        Spin configuration, modified in place
    beta : float
        Inverse temperature (unused here, probs are precomputed)
    acceptance_probs : ndarray
        Precomputed exp(-beta * dE) for dE = 4, 8

    Returns
    -------
    n_accepted : int
        Number of accepted flips
    """
    L = spins.shape[0]
    n_accepted = 0

    for _ in range(L * L):
        # Random site selection
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        # Neighbor sum with periodic boundary conditions
        nn_sum = (spins[(i + 1) % L, j] +
                  spins[(i - 1 + L) % L, j] +
                  spins[i, (j + 1) % L] +
                  spins[i, (j - 1 + L) % L])

        # Energy change: dE = 2 * s_i * sum(neighbors)
        dE = 2 * spins[i, j] * nn_sum

        # Metropolis acceptance criterion
        if dE <= 0:
            spins[i, j] *= -1
            n_accepted += 1
        elif dE == 4:
            if np.random.random() < acceptance_probs[0]:
                spins[i, j] *= -1
                n_accepted += 1
        elif dE == 8:
            if np.random.random() < acceptance_probs[1]:
                spins[i, j] *= -1
                n_accepted += 1

    return n_accepted


@njit(cache=True)
def _metropolis_sweep_2d_checkerboard(spins, beta, acceptance_probs, parity):
    """Checkerboard Metropolis sweep for 2D Ising (parallelizable).

    Updates only sites where (i + j) % 2 == parity.
    Call twice with parity=0 and parity=1 for full sweep.

    Parameters
    ----------
    spins : ndarray (L, L) of int8
        Spin configuration, modified in place
    beta : float
        Inverse temperature
    acceptance_probs : ndarray
        Precomputed acceptance probabilities
    parity : int
        0 or 1, selects checkerboard sublattice

    Returns
    -------
    n_accepted : int
        Number of accepted flips
    """
    L = spins.shape[0]
    n_accepted = 0

    for i in range(L):
        for j in range(L):
            if (i + j) % 2 != parity:
                continue

            nn_sum = (spins[(i + 1) % L, j] +
                      spins[(i - 1 + L) % L, j] +
                      spins[i, (j + 1) % L] +
                      spins[i, (j - 1 + L) % L])

            dE = 2 * spins[i, j] * nn_sum

            if dE <= 0:
                spins[i, j] *= -1
                n_accepted += 1
            elif dE == 4:
                if np.random.random() < acceptance_probs[0]:
                    spins[i, j] *= -1
                    n_accepted += 1
            elif dE == 8:
                if np.random.random() < acceptance_probs[1]:
                    spins[i, j] *= -1
                    n_accepted += 1

    return n_accepted


@njit(cache=True)
def _calculate_energy_2d(spins):
    """Calculate total energy of 2D Ising configuration.

    E = -J * sum_{<i,j>} s_i * s_j

    Only counts each pair once (right and down neighbors).

    Parameters
    ----------
    spins : ndarray (L, L) of int8
        Spin configuration

    Returns
    -------
    energy : int
        Total energy (in units of J)
    """
    L = spins.shape[0]
    energy = 0

    for i in range(L):
        for j in range(L):
            # Only count right and down neighbors to avoid double counting
            energy -= spins[i, j] * (spins[(i + 1) % L, j] +
                                      spins[i, (j + 1) % L])

    return energy


@njit(cache=True)
def _calculate_magnetization_2d(spins):
    """Calculate total magnetization of 2D configuration.

    Parameters
    ----------
    spins : ndarray (L, L) of int8
        Spin configuration

    Returns
    -------
    magnetization : int
        Total magnetization (sum of all spins)
    """
    return np.sum(spins)


@njit(cache=True)
def _wolff_cluster_2d(spins, p_add, seed_i, seed_j):
    """Build and flip a Wolff cluster using stack-based BFS.

    The Wolff algorithm builds clusters of aligned spins and flips them.
    Near Tc, this dramatically reduces critical slowing down.

    Parameters
    ----------
    spins : ndarray (L, L) of int8
        Spin configuration, modified in place
    p_add : float
        Probability to add aligned neighbor: 1 - exp(-2*beta*J)
    seed_i, seed_j : int
        Starting site coordinates

    Returns
    -------
    cluster_size : int
        Number of spins in the cluster
    """
    L = spins.shape[0]

    # Track which sites are in the cluster
    in_cluster = np.zeros((L, L), dtype=np.uint8)

    # Stack for BFS (pre-allocate to max possible size)
    stack_i = np.empty(L * L, dtype=np.int32)
    stack_j = np.empty(L * L, dtype=np.int32)
    stack_ptr = 0

    # Initialize with seed
    seed_spin = spins[seed_i, seed_j]
    in_cluster[seed_i, seed_j] = 1
    stack_i[stack_ptr] = seed_i
    stack_j[stack_ptr] = seed_j
    stack_ptr += 1

    cluster_size = 1

    # Neighbor offsets
    di = np.array([1, -1, 0, 0], dtype=np.int32)
    dj = np.array([0, 0, 1, -1], dtype=np.int32)

    while stack_ptr > 0:
        # Pop from stack
        stack_ptr -= 1
        i = stack_i[stack_ptr]
        j = stack_j[stack_ptr]

        # Check all neighbors
        for k in range(4):
            ni = (i + di[k] + L) % L
            nj = (j + dj[k] + L) % L

            # Add to cluster if: same spin, not already in cluster, random < p_add
            if (in_cluster[ni, nj] == 0 and
                spins[ni, nj] == seed_spin and
                np.random.random() < p_add):

                in_cluster[ni, nj] = 1
                stack_i[stack_ptr] = ni
                stack_j[stack_ptr] = nj
                stack_ptr += 1
                cluster_size += 1

    # Flip all spins in cluster
    for i in range(L):
        for j in range(L):
            if in_cluster[i, j]:
                spins[i, j] *= -1

    return cluster_size


@njit(cache=True)
def _correlation_function_2d(spins, max_r):
    """Compute radial correlation function C(r) for 2D configuration.

    C(r) = <s(0) * s(r)> averaged over all pairs at distance r.

    Parameters
    ----------
    spins : ndarray (L, L) of int8
        Spin configuration
    max_r : int
        Maximum distance to compute

    Returns
    -------
    C_r : ndarray of shape (max_r,)
        Correlation function values
    counts : ndarray of shape (max_r,)
        Number of pairs at each distance
    """
    L = spins.shape[0]
    C_r = np.zeros(max_r, dtype=np.float64)
    counts = np.zeros(max_r, dtype=np.int64)

    for i in range(L):
        for j in range(L):
            s0 = spins[i, j]
            for di in range(max_r):
                for dj in range(max_r):
                    if di == 0 and dj == 0:
                        continue

                    # Distance (using periodic minimum)
                    dx = min(di, L - di)
                    dy = min(dj, L - dj)
                    r = int(np.sqrt(dx * dx + dy * dy))

                    if r < max_r:
                        ni = (i + di) % L
                        nj = (j + dj) % L
                        C_r[r] += s0 * spins[ni, nj]
                        counts[r] += 1

    # Normalize
    for r in range(max_r):
        if counts[r] > 0:
            C_r[r] /= counts[r]

    return C_r, counts


# =============================================================================
# 1D Ising Model Kernels
# =============================================================================

@njit(cache=True)
def _metropolis_sweep_1d(spins, beta, acceptance_probs):
    """Numba-optimized Metropolis sweep for 1D Ising chain.

    Parameters
    ----------
    spins : ndarray (L,) of int8
        Spin configuration, modified in place
    beta : float
        Inverse temperature
    acceptance_probs : ndarray
        Precomputed exp(-beta * dE) for dE = 4

    Returns
    -------
    n_accepted : int
        Number of accepted flips
    """
    L = spins.shape[0]
    n_accepted = 0

    for _ in range(L):
        # Random site
        i = np.random.randint(0, L)

        # Neighbor sum with periodic BC
        nn_sum = spins[(i + 1) % L] + spins[(i - 1 + L) % L]

        # Energy change
        dE = 2 * spins[i] * nn_sum

        # Metropolis criterion
        if dE <= 0:
            spins[i] *= -1
            n_accepted += 1
        elif dE == 4:
            if np.random.random() < acceptance_probs[0]:
                spins[i] *= -1
                n_accepted += 1

    return n_accepted


@njit(cache=True)
def _calculate_energy_1d(spins):
    """Calculate total energy of 1D Ising chain.

    Parameters
    ----------
    spins : ndarray (L,) of int8
        Spin configuration

    Returns
    -------
    energy : int
        Total energy
    """
    L = spins.shape[0]
    energy = 0

    for i in range(L):
        # Only count right neighbor to avoid double counting
        energy -= spins[i] * spins[(i + 1) % L]

    return energy


@njit(cache=True)
def _calculate_magnetization_1d(spins):
    """Calculate total magnetization of 1D chain.

    Parameters
    ----------
    spins : ndarray (L,) of int8
        Spin configuration

    Returns
    -------
    magnetization : int
        Total magnetization
    """
    return np.sum(spins)


@njit(cache=True)
def _wolff_cluster_1d(spins, p_add, seed_i):
    """Build and flip a Wolff cluster in 1D.

    Parameters
    ----------
    spins : ndarray (L,) of int8
        Spin configuration, modified in place
    p_add : float
        Probability to add aligned neighbor
    seed_i : int
        Starting site

    Returns
    -------
    cluster_size : int
        Number of spins flipped
    """
    L = spins.shape[0]

    in_cluster = np.zeros(L, dtype=np.uint8)
    stack = np.empty(L, dtype=np.int32)
    stack_ptr = 0

    seed_spin = spins[seed_i]
    in_cluster[seed_i] = 1
    stack[stack_ptr] = seed_i
    stack_ptr += 1

    cluster_size = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        i = stack[stack_ptr]

        # Check left and right neighbors
        for di in [-1, 1]:
            ni = (i + di + L) % L

            if (in_cluster[ni] == 0 and
                spins[ni] == seed_spin and
                np.random.random() < p_add):

                in_cluster[ni] = 1
                stack[stack_ptr] = ni
                stack_ptr += 1
                cluster_size += 1

    # Flip cluster
    for i in range(L):
        if in_cluster[i]:
            spins[i] *= -1

    return cluster_size


# =============================================================================
# 3D Ising Model Kernels
# =============================================================================

@njit(cache=True)
def _metropolis_sweep_3d(spins, beta, acceptance_probs):
    """Numba-optimized Metropolis sweep for 3D Ising model.

    Parameters
    ----------
    spins : ndarray (L, L, L) of int8
        Spin configuration, modified in place
    beta : float
        Inverse temperature
    acceptance_probs : ndarray
        Precomputed exp(-beta * dE) for dE = 4, 8, 12

    Returns
    -------
    n_accepted : int
        Number of accepted flips
    """
    L = spins.shape[0]
    n_accepted = 0

    for _ in range(L * L * L):
        # Random site
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        k = np.random.randint(0, L)

        # 6 neighbors with periodic BC
        nn_sum = (spins[(i + 1) % L, j, k] + spins[(i - 1 + L) % L, j, k] +
                  spins[i, (j + 1) % L, k] + spins[i, (j - 1 + L) % L, k] +
                  spins[i, j, (k + 1) % L] + spins[i, j, (k - 1 + L) % L])

        # Energy change
        dE = 2 * spins[i, j, k] * nn_sum

        # Metropolis criterion
        if dE <= 0:
            spins[i, j, k] *= -1
            n_accepted += 1
        elif dE == 4:
            if np.random.random() < acceptance_probs[0]:
                spins[i, j, k] *= -1
                n_accepted += 1
        elif dE == 8:
            if np.random.random() < acceptance_probs[1]:
                spins[i, j, k] *= -1
                n_accepted += 1
        elif dE == 12:
            if np.random.random() < acceptance_probs[2]:
                spins[i, j, k] *= -1
                n_accepted += 1

    return n_accepted


@njit(cache=True)
def _calculate_energy_3d(spins):
    """Calculate total energy of 3D Ising configuration.

    Parameters
    ----------
    spins : ndarray (L, L, L) of int8
        Spin configuration

    Returns
    -------
    energy : int
        Total energy
    """
    L = spins.shape[0]
    energy = 0

    for i in range(L):
        for j in range(L):
            for k in range(L):
                # Only count +x, +y, +z neighbors to avoid double counting
                energy -= spins[i, j, k] * (
                    spins[(i + 1) % L, j, k] +
                    spins[i, (j + 1) % L, k] +
                    spins[i, j, (k + 1) % L]
                )

    return energy


@njit(cache=True)
def _calculate_magnetization_3d(spins):
    """Calculate total magnetization of 3D configuration.

    Parameters
    ----------
    spins : ndarray (L, L, L) of int8
        Spin configuration

    Returns
    -------
    magnetization : int
        Total magnetization
    """
    return np.sum(spins)


@njit(cache=True)
def _wolff_cluster_3d(spins, p_add, seed_i, seed_j, seed_k):
    """Build and flip a Wolff cluster in 3D.

    Parameters
    ----------
    spins : ndarray (L, L, L) of int8
        Spin configuration, modified in place
    p_add : float
        Probability to add aligned neighbor
    seed_i, seed_j, seed_k : int
        Starting site coordinates

    Returns
    -------
    cluster_size : int
        Number of spins flipped
    """
    L = spins.shape[0]

    in_cluster = np.zeros((L, L, L), dtype=np.uint8)

    # Stack for BFS
    stack_i = np.empty(L * L * L, dtype=np.int32)
    stack_j = np.empty(L * L * L, dtype=np.int32)
    stack_k = np.empty(L * L * L, dtype=np.int32)
    stack_ptr = 0

    seed_spin = spins[seed_i, seed_j, seed_k]
    in_cluster[seed_i, seed_j, seed_k] = 1
    stack_i[stack_ptr] = seed_i
    stack_j[stack_ptr] = seed_j
    stack_k[stack_ptr] = seed_k
    stack_ptr += 1

    cluster_size = 1

    # Neighbor offsets for 3D (6 neighbors)
    di = np.array([1, -1, 0, 0, 0, 0], dtype=np.int32)
    dj = np.array([0, 0, 1, -1, 0, 0], dtype=np.int32)
    dk = np.array([0, 0, 0, 0, 1, -1], dtype=np.int32)

    while stack_ptr > 0:
        stack_ptr -= 1
        i = stack_i[stack_ptr]
        j = stack_j[stack_ptr]
        k = stack_k[stack_ptr]

        for n in range(6):
            ni = (i + di[n] + L) % L
            nj = (j + dj[n] + L) % L
            nk = (k + dk[n] + L) % L

            if (in_cluster[ni, nj, nk] == 0 and
                spins[ni, nj, nk] == seed_spin and
                np.random.random() < p_add):

                in_cluster[ni, nj, nk] = 1
                stack_i[stack_ptr] = ni
                stack_j[stack_ptr] = nj
                stack_k[stack_ptr] = nk
                stack_ptr += 1
                cluster_size += 1

    # Flip cluster
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if in_cluster[i, j, k]:
                    spins[i, j, k] *= -1

    return cluster_size


# =============================================================================
# Parallel versions using prange (for multi-core)
# =============================================================================

@njit(parallel=True, cache=True)
def _calculate_energy_2d_parallel(spins):
    """Parallel energy calculation for 2D Ising.

    Uses OpenMP-style parallelization via numba.prange.
    """
    L = spins.shape[0]
    energy = 0

    for i in prange(L):
        row_energy = 0
        for j in range(L):
            row_energy -= spins[i, j] * (spins[(i + 1) % L, j] +
                                          spins[i, (j + 1) % L])
        energy += row_energy

    return energy


@njit(parallel=True, cache=True)
def _calculate_energy_3d_parallel(spins):
    """Parallel energy calculation for 3D Ising."""
    L = spins.shape[0]
    energy = 0

    for i in prange(L):
        plane_energy = 0
        for j in range(L):
            for k in range(L):
                plane_energy -= spins[i, j, k] * (
                    spins[(i + 1) % L, j, k] +
                    spins[i, (j + 1) % L, k] +
                    spins[i, j, (k + 1) % L]
                )
        energy += plane_energy

    return energy


# =============================================================================
# High-level wrappers with fallback to pure Python
# =============================================================================

def metropolis_sweep_2d(spins, beta, acceptance_probs=None):
    """Perform one Metropolis sweep on 2D Ising model.

    This is a wrapper that uses the Numba kernel if available.

    Parameters
    ----------
    spins : ndarray (L, L)
        Spin configuration, modified in place
    beta : float
        Inverse temperature
    acceptance_probs : ndarray, optional
        Precomputed acceptance probabilities

    Returns
    -------
    n_accepted : int
        Number of accepted flips
    """
    if acceptance_probs is None:
        acceptance_probs = precompute_acceptance_probs_2d(beta)

    return _metropolis_sweep_2d(spins.astype(np.int8), beta, acceptance_probs)


def metropolis_sweep_1d(spins, beta, acceptance_probs=None):
    """Perform one Metropolis sweep on 1D Ising chain."""
    if acceptance_probs is None:
        acceptance_probs = precompute_acceptance_probs_1d(beta)

    return _metropolis_sweep_1d(spins.astype(np.int8), beta, acceptance_probs)


def metropolis_sweep_3d(spins, beta, acceptance_probs=None):
    """Perform one Metropolis sweep on 3D Ising model."""
    if acceptance_probs is None:
        acceptance_probs = precompute_acceptance_probs_3d(beta)

    return _metropolis_sweep_3d(spins.astype(np.int8), beta, acceptance_probs)


def wolff_step_2d(spins, beta):
    """Perform one Wolff cluster update on 2D Ising model.

    Parameters
    ----------
    spins : ndarray (L, L)
        Spin configuration, modified in place
    beta : float
        Inverse temperature

    Returns
    -------
    cluster_size : int
        Number of spins flipped
    """
    L = spins.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta)
    seed_i = np.random.randint(0, L)
    seed_j = np.random.randint(0, L)

    return _wolff_cluster_2d(spins.astype(np.int8), p_add, seed_i, seed_j)


def wolff_step_1d(spins, beta):
    """Perform one Wolff cluster update on 1D Ising chain."""
    L = spins.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta)
    seed_i = np.random.randint(0, L)

    return _wolff_cluster_1d(spins.astype(np.int8), p_add, seed_i)


def wolff_step_3d(spins, beta):
    """Perform one Wolff cluster update on 3D Ising model."""
    L = spins.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta)
    seed_i = np.random.randint(0, L)
    seed_j = np.random.randint(0, L)
    seed_k = np.random.randint(0, L)

    return _wolff_cluster_3d(spins.astype(np.int8), p_add, seed_i, seed_j, seed_k)


# =============================================================================
# Pure Python reference implementations for benchmarking
# =============================================================================

def _metropolis_sweep_2d_python(spins, beta, acceptance_probs):
    """Pure Python Metropolis sweep for 2D Ising (reference implementation)."""
    L = spins.shape[0]
    n_accepted = 0

    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        nn_sum = (spins[(i + 1) % L, j] + spins[(i - 1) % L, j] +
                  spins[i, (j + 1) % L] + spins[i, (j - 1) % L])

        dE = 2 * spins[i, j] * nn_sum

        if dE <= 0:
            spins[i, j] *= -1
            n_accepted += 1
        elif np.random.random() < np.exp(-beta * dE):
            spins[i, j] *= -1
            n_accepted += 1

    return n_accepted


def _calculate_energy_2d_python(spins):
    """Pure Python energy calculation for 2D Ising (reference implementation)."""
    L = spins.shape[0]
    energy = 0

    for i in range(L):
        for j in range(L):
            energy -= spins[i, j] * (spins[(i + 1) % L, j] + spins[i, (j + 1) % L])

    return energy


def _metropolis_sweep_3d_python(spins, beta, acceptance_probs):
    """Pure Python Metropolis sweep for 3D Ising (reference implementation)."""
    L = spins.shape[0]
    n_accepted = 0

    for _ in range(L * L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        k = np.random.randint(0, L)

        nn_sum = (spins[(i + 1) % L, j, k] + spins[(i - 1) % L, j, k] +
                  spins[i, (j + 1) % L, k] + spins[i, (j - 1) % L, k] +
                  spins[i, j, (k + 1) % L] + spins[i, j, (k - 1) % L])

        dE = 2 * spins[i, j, k] * nn_sum

        if dE <= 0:
            spins[i, j, k] *= -1
            n_accepted += 1
        elif np.random.random() < np.exp(-beta * dE):
            spins[i, j, k] *= -1
            n_accepted += 1

    return n_accepted


# =============================================================================
# Benchmarking utilities
# =============================================================================

def run_benchmark(L=32, n_sweeps=100, temperature=2.269, dimension=2):
    """Run benchmark comparing Numba vs pure Python implementations.

    Parameters
    ----------
    L : int
        System size
    n_sweeps : int
        Number of Monte Carlo sweeps
    temperature : float
        Temperature
    dimension : int
        1, 2, or 3

    Returns
    -------
    dict
        Benchmark results with timing information
    """
    import time

    beta = 1.0 / temperature

    results = {
        'L': L,
        'n_sweeps': n_sweeps,
        'temperature': temperature,
        'dimension': dimension,
        'numba_available': NUMBA_AVAILABLE,
    }

    if dimension == 1:
        spins_numba = np.random.choice([-1, 1], size=L).astype(np.int8)
        spins_python = spins_numba.copy()
        acceptance_probs = precompute_acceptance_probs_1d(beta)
        numba_func = _metropolis_sweep_1d
        python_func = lambda s, b, p: _metropolis_sweep_2d_python(s.reshape(-1), b, p)
        n_sites = L

    elif dimension == 2:
        spins_numba = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
        spins_python = spins_numba.copy()
        acceptance_probs = precompute_acceptance_probs_2d(beta)
        numba_func = _metropolis_sweep_2d
        python_func = _metropolis_sweep_2d_python
        n_sites = L * L

    else:  # 3D
        spins_numba = np.random.choice([-1, 1], size=(L, L, L)).astype(np.int8)
        spins_python = spins_numba.copy()
        acceptance_probs = precompute_acceptance_probs_3d(beta)
        numba_func = _metropolis_sweep_3d
        python_func = _metropolis_sweep_3d_python
        n_sites = L * L * L

    # Warmup for JIT compilation
    if NUMBA_AVAILABLE:
        _ = numba_func(spins_numba.copy(), beta, acceptance_probs)

    # Benchmark Numba version
    if NUMBA_AVAILABLE:
        start = time.perf_counter()
        for _ in range(n_sweeps):
            numba_func(spins_numba, beta, acceptance_probs)
        numba_time = time.perf_counter() - start
        results['numba_time'] = numba_time
        results['numba_sweeps_per_sec'] = n_sweeps / numba_time
        results['numba_flips_per_sec'] = n_sweeps * n_sites / numba_time

    # Benchmark Python version (fewer sweeps since it's slow)
    python_sweeps = min(n_sweeps, 10)
    start = time.perf_counter()
    for _ in range(python_sweeps):
        python_func(spins_python, beta, acceptance_probs)
    python_time = time.perf_counter() - start

    # Extrapolate to same number of sweeps
    python_time_extrapolated = python_time * (n_sweeps / python_sweeps)
    results['python_time'] = python_time_extrapolated
    results['python_sweeps_per_sec'] = n_sweeps / python_time_extrapolated
    results['python_flips_per_sec'] = n_sweeps * n_sites / python_time_extrapolated

    # Speedup
    if NUMBA_AVAILABLE:
        results['speedup'] = python_time_extrapolated / numba_time
    else:
        results['speedup'] = 1.0

    return results


def print_benchmark_results(results):
    """Print formatted benchmark results."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {results['dimension']}D Ising, L={results['L']}, "
          f"T={results['temperature']:.3f}")
    print(f"{'='*60}")
    print(f"Sweeps: {results['n_sweeps']}")
    print(f"Numba available: {results['numba_available']}")
    print()

    if results['numba_available']:
        print(f"Numba JIT:")
        print(f"  Time: {results['numba_time']:.3f} s")
        print(f"  Sweeps/sec: {results['numba_sweeps_per_sec']:.1f}")
        print(f"  Spin flips/sec: {results['numba_flips_per_sec']:.2e}")
        print()

    print(f"Pure Python:")
    print(f"  Time (extrapolated): {results['python_time']:.3f} s")
    print(f"  Sweeps/sec: {results['python_sweeps_per_sec']:.1f}")
    print(f"  Spin flips/sec: {results['python_flips_per_sec']:.2e}")
    print()

    print(f"Speedup: {results['speedup']:.1f}x")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Run benchmarks when executed directly
    print("Running Numba kernel benchmarks...")
    print(f"Numba available: {NUMBA_AVAILABLE}")

    # 1D benchmark
    results_1d = run_benchmark(L=1000, n_sweeps=100, dimension=1)
    print_benchmark_results(results_1d)

    # 2D benchmark
    results_2d = run_benchmark(L=64, n_sweeps=100, dimension=2)
    print_benchmark_results(results_2d)

    # 3D benchmark
    results_3d = run_benchmark(L=16, n_sweeps=50, dimension=3)
    print_benchmark_results(results_3d)

    # Summary
    print("\nSummary:")
    print(f"  1D (L=1000): {results_1d['speedup']:.1f}x speedup")
    print(f"  2D (L=64):   {results_2d['speedup']:.1f}x speedup")
    print(f"  3D (L=16):   {results_3d['speedup']:.1f}x speedup")
