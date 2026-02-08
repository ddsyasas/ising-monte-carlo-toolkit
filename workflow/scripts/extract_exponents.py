"""
Extract critical exponents from finite-size scaling analysis.

Critical exponents are extracted from:
1. Susceptibility peak scaling: chi_max ~ L^(gamma/nu)
2. Magnetization at Tc: M(Tc) ~ L^(-beta/nu)
3. Binder cumulant derivative: dU/dT|_Tc ~ L^(1/nu)
4. Heat capacity peak: C_max ~ L^(alpha/nu) or log(L)

For 2D Ising, exact values are:
- beta = 1/8 = 0.125
- gamma = 7/4 = 1.75
- nu = 1
- alpha = 0 (logarithmic)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize


def fit_power_law(sizes, values, errors=None):
    """Fit power law: y = a * L^b

    Parameters
    ----------
    sizes : array
        System sizes
    values : array
        Observable values
    errors : array, optional
        Errors on values

    Returns
    -------
    dict
        Fit results with exponent and amplitude
    """
    log_L = np.log(sizes)
    log_y = np.log(values)

    # Weighted fit if errors provided
    if errors is not None and np.all(errors > 0):
        # Propagate errors to log scale
        log_errors = errors / values
        weights = 1 / log_errors**2
        coeffs, cov = np.polyfit(log_L, log_y, 1, w=weights, cov=True)
    else:
        coeffs, cov = np.polyfit(log_L, log_y, 1, cov=True)

    exponent = coeffs[0]
    log_amplitude = coeffs[1]

    # Error estimates
    exponent_err = np.sqrt(cov[0, 0])
    amplitude_err = np.exp(log_amplitude) * np.sqrt(cov[1, 1])

    return {
        'exponent': float(exponent),
        'exponent_err': float(exponent_err),
        'amplitude': float(np.exp(log_amplitude)),
        'amplitude_err': float(amplitude_err),
        'r_squared': float(1 - np.var(log_y - np.polyval(coeffs, log_L)) / np.var(log_y))
    }


def extract_gamma_nu(data_by_size, sizes):
    """Extract gamma/nu from susceptibility peak scaling.

    chi_max ~ L^(gamma/nu)
    """
    chi_max_values = []
    chi_max_errors = []

    for L in sizes:
        if L not in data_by_size:
            continue

        df = data_by_size[L]
        if 'susceptibility' not in df.columns:
            continue

        chi_max = df['susceptibility'].max()
        chi_max_values.append(chi_max)

        # Get error if available
        if 'susceptibility_err' in df.columns:
            idx = df['susceptibility'].idxmax()
            chi_max_errors.append(df.loc[idx, 'susceptibility_err'])
        else:
            chi_max_errors.append(0.0)

    if len(chi_max_values) < 2:
        return None

    valid_sizes = [L for L in sizes if L in data_by_size]
    fit = fit_power_law(
        np.array(valid_sizes[:len(chi_max_values)]),
        np.array(chi_max_values),
        np.array(chi_max_errors) if any(chi_max_errors) else None
    )

    return {
        'gamma_nu': fit['exponent'],
        'gamma_nu_err': fit['exponent_err'],
        'chi_amplitude': fit['amplitude'],
        'r_squared': fit['r_squared'],
        'sizes_used': valid_sizes[:len(chi_max_values)],
        'chi_max_values': chi_max_values
    }


def extract_beta_nu(data_by_size, sizes, Tc):
    """Extract beta/nu from magnetization at Tc.

    M(Tc) ~ L^(-beta/nu)
    """
    mag_at_Tc = []
    mag_errors = []
    valid_sizes = []

    for L in sizes:
        if L not in data_by_size:
            continue

        df = data_by_size[L].sort_values('temperature')

        if 'magnetization_mean' not in df.columns and 'abs_magnetization_mean' not in df.columns:
            continue

        # Find temperature closest to Tc
        idx = (df['temperature'] - Tc).abs().idxmin()

        # Use absolute magnetization if available
        if 'abs_magnetization_mean' in df.columns:
            mag = df.loc[idx, 'abs_magnetization_mean']
        else:
            mag = abs(df.loc[idx, 'magnetization_mean'])

        if mag > 1e-10:  # Only include non-zero values
            mag_at_Tc.append(mag)
            valid_sizes.append(L)

            if 'magnetization_err' in df.columns:
                mag_errors.append(df.loc[idx, 'magnetization_err'])
            elif 'abs_magnetization_err' in df.columns:
                mag_errors.append(df.loc[idx, 'abs_magnetization_err'])
            else:
                mag_errors.append(0.0)

    if len(mag_at_Tc) < 2:
        return None

    fit = fit_power_law(
        np.array(valid_sizes),
        np.array(mag_at_Tc),
        np.array(mag_errors) if any(mag_errors) else None
    )

    # beta/nu is negative of the exponent since M ~ L^(-beta/nu)
    return {
        'beta_nu': -fit['exponent'],
        'beta_nu_err': fit['exponent_err'],
        'm_amplitude': fit['amplitude'],
        'r_squared': fit['r_squared'],
        'Tc_used': float(Tc),
        'sizes_used': valid_sizes,
        'mag_at_Tc': mag_at_Tc
    }


def extract_one_over_nu(data_by_size, sizes, Tc):
    """Extract 1/nu from Binder cumulant derivative scaling.

    dU/dT|_Tc ~ L^(1/nu)
    """
    dU_dT_values = []
    valid_sizes = []

    for L in sizes:
        if L not in data_by_size:
            continue

        df = data_by_size[L].sort_values('temperature')

        if 'binder' not in df.columns:
            continue

        temps = df['temperature'].values
        binder = df['binder'].values

        # Numerical derivative
        dU_dT = np.gradient(binder, temps)

        # Find value at Tc
        idx = (temps - Tc).argmin()

        # Use absolute value of derivative (should be negative below Tc)
        dU_dT_values.append(abs(dU_dT[idx]))
        valid_sizes.append(L)

    if len(dU_dT_values) < 2:
        return None

    fit = fit_power_law(
        np.array(valid_sizes),
        np.array(dU_dT_values)
    )

    return {
        'one_over_nu': fit['exponent'],
        'one_over_nu_err': fit['exponent_err'],
        'nu': 1.0 / fit['exponent'] if fit['exponent'] != 0 else None,
        'r_squared': fit['r_squared'],
        'sizes_used': valid_sizes
    }


def main():
    """Extract critical exponents from sweep data."""
    sweep_files = snakemake.input.sweeps
    Tc_file = snakemake.input.Tc
    output_file = snakemake.output[0]
    model = snakemake.params.get('model', 'ising2d')
    sizes = snakemake.params.get('sizes', [])

    # Load Tc
    with open(Tc_file) as f:
        Tc_data = json.load(f)

    Tc = Tc_data.get('Tc')
    if Tc is None:
        raise ValueError("No Tc found in critical temperature file")

    # Load sweep data
    data_by_size = {}
    for filepath in sweep_files:
        try:
            df = pd.read_csv(filepath)
            size = int(df['size'].iloc[0])
            data_by_size[size] = df
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    available_sizes = sorted(data_by_size.keys())

    results = {
        'model': model,
        'Tc': float(Tc),
        'sizes': available_sizes
    }

    # Extract gamma/nu from susceptibility
    gamma_nu_result = extract_gamma_nu(data_by_size, available_sizes)
    if gamma_nu_result:
        results['gamma_nu'] = gamma_nu_result

    # Extract beta/nu from magnetization
    beta_nu_result = extract_beta_nu(data_by_size, available_sizes, Tc)
    if beta_nu_result:
        results['beta_nu'] = beta_nu_result

    # Extract 1/nu from Binder derivative
    nu_result = extract_one_over_nu(data_by_size, available_sizes, Tc)
    if nu_result:
        results['one_over_nu'] = nu_result

    # Compute individual exponents if we have nu
    if nu_result and nu_result.get('nu'):
        nu = nu_result['nu']
        results['nu'] = float(nu)

        if gamma_nu_result:
            results['gamma'] = float(gamma_nu_result['gamma_nu'] * nu)
            results['gamma_err'] = float(gamma_nu_result['gamma_nu_err'] * nu)

        if beta_nu_result:
            results['beta'] = float(beta_nu_result['beta_nu'] * nu)
            results['beta_err'] = float(beta_nu_result['beta_nu_err'] * nu)

    # Compare with exact values for 2D Ising
    if '2d' in model.lower():
        exact = {
            'beta': 0.125,
            'gamma': 1.75,
            'nu': 1.0,
            'alpha': 0.0
        }
        results['exact_values'] = exact

        # Compute deviations
        if 'beta' in results:
            results['beta_deviation'] = abs(results['beta'] - exact['beta'])
        if 'gamma' in results:
            results['gamma_deviation'] = abs(results['gamma'] - exact['gamma'])
        if 'nu' in results:
            results['nu_deviation'] = abs(results['nu'] - exact['nu'])

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Critical exponents saved to {output_file}")

    # Print summary
    if 'beta' in results:
        print(f"  β = {results['beta']:.4f} (exact: 0.125)")
    if 'gamma' in results:
        print(f"  γ = {results['gamma']:.4f} (exact: 1.75)")
    if 'nu' in results:
        print(f"  ν = {results['nu']:.4f} (exact: 1.0)")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        class Input:
            def __init__(self, args):
                self.sweeps = args[:-1]
                self.Tc = args[-1]

        def __init__(self):
            self.input = self.Input(sys.argv[2:])
            self.output = [sys.argv[1]]
            self.params = {'model': 'ising2d', 'sizes': []}

    snakemake = MockSnakemake()
    main()
