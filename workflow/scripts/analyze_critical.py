"""
Analyze simulation results to estimate critical temperature and exponents.

This script performs finite-size scaling analysis on the collected
simulation data to estimate the critical point.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_susceptibility_peaks(df):
    """Find susceptibility peaks for each system size.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: size, temperature, susceptibility

    Returns
    -------
    dict
        Dictionary mapping size to (T_peak, chi_peak)
    """
    peaks = {}
    for size in df['size'].unique():
        size_data = df[df['size'] == size].sort_values('temperature')
        idx_max = size_data['susceptibility'].idxmax()
        peaks[int(size)] = {
            'T_peak': float(size_data.loc[idx_max, 'temperature']),
            'chi_peak': float(size_data.loc[idx_max, 'susceptibility'])
        }
    return peaks


def find_binder_crossing(df, sizes=None):
    """Find Binder cumulant crossing points.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: size, temperature, binder
    sizes : list, optional
        Sizes to use for crossing analysis

    Returns
    -------
    dict
        Crossing information including estimated Tc
    """
    if sizes is None:
        sizes = sorted(df['size'].unique())

    if len(sizes) < 2:
        return {'error': 'Need at least 2 sizes for crossing analysis'}

    # For each pair of sizes, find crossing
    crossings = []

    for i, L1 in enumerate(sizes[:-1]):
        L2 = sizes[i + 1]

        df1 = df[df['size'] == L1].sort_values('temperature')
        df2 = df[df['size'] == L2].sort_values('temperature')

        # Interpolate to common temperature grid
        temps = np.linspace(
            max(df1['temperature'].min(), df2['temperature'].min()),
            min(df1['temperature'].max(), df2['temperature'].max()),
            100
        )

        binder1 = np.interp(temps, df1['temperature'], df1['binder'])
        binder2 = np.interp(temps, df2['temperature'], df2['binder'])

        # Find crossing (sign change in difference)
        diff = binder1 - binder2
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        for idx in sign_changes:
            # Linear interpolation to find exact crossing
            T_cross = temps[idx] - diff[idx] * (temps[idx+1] - temps[idx]) / (diff[idx+1] - diff[idx])
            U_cross = np.interp(T_cross, temps, binder1)
            crossings.append({
                'L1': int(L1),
                'L2': int(L2),
                'T_cross': float(T_cross),
                'U_cross': float(U_cross)
            })

    if not crossings:
        return {'error': 'No crossing found', 'crossings': []}

    # Estimate Tc from crossings
    T_estimates = [c['T_cross'] for c in crossings]
    Tc_estimate = np.mean(T_estimates)
    Tc_error = np.std(T_estimates) if len(T_estimates) > 1 else 0

    return {
        'crossings': crossings,
        'Tc_estimate': float(Tc_estimate),
        'Tc_error': float(Tc_error),
        'U_star': float(np.mean([c['U_cross'] for c in crossings]))
    }


def estimate_exponents(df, Tc):
    """Estimate critical exponents from scaling relations.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation data
    Tc : float
        Estimated critical temperature

    Returns
    -------
    dict
        Estimated exponents with errors
    """
    sizes = sorted(df['size'].unique())

    # Get data at temperatures closest to Tc
    exponents = {}

    # Susceptibility scaling: chi_max ~ L^(gamma/nu)
    chi_peaks = []
    for L in sizes:
        size_data = df[df['size'] == L]
        chi_max = size_data['susceptibility'].max()
        chi_peaks.append((L, chi_max))

    if len(chi_peaks) >= 2:
        log_L = np.log([x[0] for x in chi_peaks])
        log_chi = np.log([x[1] for x in chi_peaks])
        gamma_nu, _ = np.polyfit(log_L, log_chi, 1)
        exponents['gamma_nu'] = float(gamma_nu)

    # Magnetization scaling at Tc: M(Tc) ~ L^(-beta/nu)
    mag_at_Tc = []
    for L in sizes:
        size_data = df[df['size'] == L]
        # Find closest temperature to Tc
        idx = (size_data['temperature'] - Tc).abs().idxmin()
        mag_at_Tc.append((L, size_data.loc[idx, 'magnetization_mean']))

    if len(mag_at_Tc) >= 2:
        log_L = np.log([x[0] for x in mag_at_Tc])
        log_M = np.log([max(x[1], 1e-10) for x in mag_at_Tc])
        neg_beta_nu, _ = np.polyfit(log_L, log_M, 1)
        exponents['beta_nu'] = float(-neg_beta_nu)

    return exponents


def main():
    """Main entry point."""
    # Load data
    input_file = snakemake.input[0]
    output_file = snakemake.output[0]
    model = snakemake.params.get('model', 'ising2d')

    df = pd.read_csv(input_file)

    # Known exact values for comparison
    exact_values = {
        'ising2d': {
            'Tc': 2.269185,
            'beta': 0.125,
            'gamma': 1.75,
            'nu': 1.0,
            'alpha': 0.0
        },
        'ising3d': {
            'Tc': 4.511,
            'beta': 0.326,
            'gamma': 1.237,
            'nu': 0.630
        }
    }

    # Analysis results
    results = {
        'model': model,
        'sizes': sorted([int(x) for x in df['size'].unique()]),
        'temperature_range': [float(df['temperature'].min()),
                              float(df['temperature'].max())],
        'n_temperatures': int(df.groupby('size').size().iloc[0])
    }

    # Susceptibility peak analysis
    results['susceptibility_peaks'] = find_susceptibility_peaks(df)

    # Binder crossing
    binder_results = find_binder_crossing(df)
    results['binder_analysis'] = binder_results

    # Get Tc estimate
    if 'Tc_estimate' in binder_results:
        Tc = binder_results['Tc_estimate']
    else:
        # Fallback to susceptibility peak average
        peaks = results['susceptibility_peaks']
        Tc = np.mean([p['T_peak'] for p in peaks.values()])
    results['Tc_estimate'] = float(Tc)

    # Estimate exponents
    results['exponents'] = estimate_exponents(df, Tc)

    # Compare with exact values if available
    if model in exact_values:
        exact = exact_values[model]
        results['exact_values'] = exact
        results['Tc_deviation'] = float(abs(Tc - exact['Tc']))
        results['Tc_relative_error'] = float(abs(Tc - exact['Tc']) / exact['Tc'])

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Critical analysis saved to {output_file}")
    print(f"Estimated Tc = {Tc:.4f}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = [sys.argv[1]]
            self.output = [sys.argv[2]]
            self.params = {'model': 'ising2d'}

    snakemake = MockSnakemake()
    main()
