"""
Estimate correlation length from spin-spin correlation function.

Methods:
1. Second moment definition: xi = (1/2sin(pi/L)) * sqrt(chi/F - 1)
   where F is the structure factor at smallest wavevector
2. Exponential fit to C(r)
"""

import csv
from pathlib import Path

import numpy as np


def compute_structure_factor(spins, k):
    """Compute structure factor S(k) for wavevector k.

    Parameters
    ----------
    spins : np.ndarray
        2D spin configuration
    k : tuple
        Wavevector (kx, ky)

    Returns
    -------
    float
        Structure factor S(k)
    """
    Lx, Ly = spins.shape
    kx, ky = k

    # Compute Fourier transform
    x, y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij')
    phase = np.exp(-1j * (kx * x + ky * y))
    m_k = np.sum(spins * phase)

    return np.abs(m_k) ** 2 / (Lx * Ly)


def second_moment_xi(spins):
    """Compute correlation length using second moment definition.

    xi = (1/(2*sin(pi/L))) * sqrt(chi/F(k_min) - 1)

    Parameters
    ----------
    spins : np.ndarray
        2D spin configuration

    Returns
    -------
    float
        Correlation length estimate
    """
    L = spins.shape[0]

    # Susceptibility (zero wavevector)
    chi = np.mean(spins) ** 2 * L ** 2

    # Structure factor at smallest wavevector
    k_min = (2 * np.pi / L, 0)
    F = compute_structure_factor(spins, k_min)

    if F < 1e-10:
        return 0.0

    ratio = chi / F
    if ratio <= 1:
        return 0.0

    xi = (1 / (2 * np.sin(np.pi / L))) * np.sqrt(ratio - 1)

    return min(xi, L / 2)  # Cap at L/2


def main():
    """Compute correlation length for a single simulation."""
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    method = snakemake.params.get('method', 'second_moment')

    # Load data
    data = dict(np.load(input_file, allow_pickle=True))

    # Extract metadata
    model = str(data.get('model', ''))
    size = int(data.get('size', 0))
    temperature = float(data.get('temperature', 0))

    results = {
        'model': model,
        'size': size,
        'temperature': temperature,
        'method': method,
    }

    if 'spins' in data and '2d' in model.lower():
        spins = np.asarray(data['spins'])

        if method == 'second_moment':
            xi = second_moment_xi(spins)
            results['xi'] = xi
            results['xi_over_L'] = xi / size if size > 0 else 0.0
        else:
            results['xi'] = 0.0
            results['xi_over_L'] = 0.0
            results['error'] = f"Unknown method: {method}"
    else:
        results['xi'] = 0.0
        results['xi_over_L'] = 0.0
        results['error'] = "No spin configuration or not 2D model"

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write results
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]
            self.params = {'method': 'second_moment'}

    snakemake = MockSnakemake()
    main()
