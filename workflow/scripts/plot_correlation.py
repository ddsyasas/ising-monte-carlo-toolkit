"""
Plot spin-spin correlation function C(r).

The correlation function is defined as:
C(r) = <s_i s_j> - <s_i><s_j>

where r = |i - j| is the distance between spins.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_correlation_function(spins):
    """Compute radial correlation function from spin configuration.

    Parameters
    ----------
    spins : np.ndarray
        2D spin configuration

    Returns
    -------
    r : np.ndarray
        Distances
    C_r : np.ndarray
        Correlation function values
    """
    L = spins.shape[0]

    # Use FFT for efficient correlation computation
    # C(r) = IFFT(|FFT(s)|^2) / N - m^2

    m = np.mean(spins)
    s_fft = np.fft.fft2(spins)
    power_spectrum = np.abs(s_fft) ** 2

    # Inverse FFT gives correlation
    corr_2d = np.real(np.fft.ifft2(power_spectrum)) / (L * L) - m ** 2

    # Radial average
    # Create distance array
    x = np.arange(L)
    x = np.minimum(x, L - x)  # Use periodic distance
    y = np.arange(L)
    y = np.minimum(y, L - y)

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)

    # Bin by distance
    r_max = L // 2
    r_bins = np.arange(r_max + 1)
    C_r = np.zeros(r_max)
    counts = np.zeros(r_max)

    for i in range(L):
        for j in range(L):
            r = int(round(R[i, j]))
            if r < r_max:
                C_r[r] += corr_2d[i, j]
                counts[r] += 1

    # Normalize
    mask = counts > 0
    C_r[mask] /= counts[mask]

    return np.arange(r_max), C_r


def setup_style(style='publication'):
    """Configure matplotlib style."""
    if style == 'publication':
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 11,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.figsize': (8, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })
    else:
        plt.style.use('default')


def main():
    """Generate correlation function plot."""
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    style = snakemake.params.get('style', 'publication')

    setup_style(style)

    # Load data
    data = dict(np.load(input_file, allow_pickle=True))

    model = str(data.get('model', 'unknown'))
    size = int(data.get('size', 0))
    temperature = float(data.get('temperature', 0))

    if 'spins' not in data:
        raise ValueError("No spin configuration in data file")

    spins = np.asarray(data['spins'])

    # Compute correlation function
    r, C_r = compute_correlation_function(spins)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Linear scale
    ax1 = axes[0]
    ax1.plot(r, C_r, 'o-', markersize=4, linewidth=1)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Distance $r$')
    ax1.set_ylabel('$C(r)$')
    ax1.set_title(f'Correlation Function (L={size}, T={temperature:.4f})')
    ax1.grid(True, alpha=0.3)

    # Right: Log scale for exponential decay analysis
    ax2 = axes[1]
    # Only plot positive values for log scale
    mask = C_r > 0
    if np.any(mask):
        ax2.semilogy(r[mask], C_r[mask], 'o-', markersize=4, linewidth=1)

        # Fit exponential decay: C(r) ~ exp(-r/xi)
        if np.sum(mask) > 3:
            r_fit = r[mask]
            C_fit = C_r[mask]
            log_C = np.log(C_fit)

            # Linear fit to log(C)
            try:
                coeffs = np.polyfit(r_fit[:len(r_fit)//2], log_C[:len(r_fit)//2], 1)
                xi = -1 / coeffs[0] if coeffs[0] < 0 else np.inf

                # Plot fit
                r_line = np.linspace(0, r.max(), 100)
                C_line = np.exp(np.polyval(coeffs, r_line))
                ax2.semilogy(r_line, C_line, '--', color='red', alpha=0.7,
                             label=f'$\\xi \\approx {xi:.2f}$')
                ax2.legend()
            except Exception:
                pass

    ax2.set_xlabel('Distance $r$')
    ax2.set_ylabel('$C(r)$')
    ax2.set_title('Log Scale (Exponential Decay)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close()

    print(f"Correlation function plot saved to {output_file}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]
            self.params = {'style': 'publication'}

    snakemake = MockSnakemake()
    main()
