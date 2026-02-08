"""
Estimate critical temperature from multiple system sizes.

Methods available:
1. Binder crossing: Find crossing point of U(T) curves for different L
2. Susceptibility peak: Extrapolate chi_max peak positions
3. Magnetization inflection: Find steepest drop in M(T)

The Binder crossing method is generally most reliable.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, optimize


def log(message):
    """Log message to stderr."""
    print(message, file=sys.stderr)


def find_binder_crossings(data_by_size, sizes):
    """Find Binder cumulant crossing points between size pairs.

    Parameters
    ----------
    data_by_size : dict
        Dictionary mapping size to DataFrame with 'temperature' and 'binder'
    sizes : list
        List of system sizes

    Returns
    -------
    list
        List of crossing information dicts
    """
    crossings = []
    sorted_sizes = sorted(sizes)

    for i in range(len(sorted_sizes) - 1):
        L1 = sorted_sizes[i]
        L2 = sorted_sizes[i + 1]

        if L1 not in data_by_size or L2 not in data_by_size:
            log(f"  Skipping L={L1}-L={L2}: missing data")
            continue

        df1 = data_by_size[L1].sort_values('temperature')
        df2 = data_by_size[L2].sort_values('temperature')

        if 'binder' not in df1.columns or 'binder' not in df2.columns:
            log(f"  Skipping L={L1}-L={L2}: no binder data")
            continue

        # Find common temperature range
        t_min = max(df1['temperature'].min(), df2['temperature'].min())
        t_max = min(df1['temperature'].max(), df2['temperature'].max())

        if t_min >= t_max:
            log(f"  Skipping L={L1}-L={L2}: no overlapping temperature range")
            continue

        # Interpolate to common grid
        temps = np.linspace(t_min, t_max, 200)

        try:
            interp1 = interpolate.interp1d(
                df1['temperature'], df1['binder'],
                kind='cubic', fill_value='extrapolate'
            )
            interp2 = interpolate.interp1d(
                df2['temperature'], df2['binder'],
                kind='cubic', fill_value='extrapolate'
            )

            binder1 = interp1(temps)
            binder2 = interp2(temps)

            # Find crossing (sign change in difference)
            diff = binder1 - binder2
            sign_changes = np.where(np.diff(np.sign(diff)))[0]

            for idx in sign_changes:
                # Refine crossing using root finding
                try:
                    t_cross = optimize.brentq(
                        lambda t: float(interp1(t)) - float(interp2(t)),
                        temps[idx], temps[idx + 1]
                    )
                    u_cross = float(interp1(t_cross))

                    crossings.append({
                        'L1': int(L1),
                        'L2': int(L2),
                        'T_cross': float(t_cross),
                        'U_cross': u_cross
                    })
                    log(f"  Found crossing L={L1}-L={L2}: T={t_cross:.4f}, U={u_cross:.4f}")

                except ValueError:
                    # Linear interpolation fallback
                    t_cross = temps[idx] - diff[idx] * (temps[idx+1] - temps[idx]) / (diff[idx+1] - diff[idx])
                    crossings.append({
                        'L1': int(L1),
                        'L2': int(L2),
                        'T_cross': float(t_cross),
                        'U_cross': float(interp1(t_cross))
                    })
                    log(f"  Found crossing L={L1}-L={L2} (linear): T={t_cross:.4f}")

        except Exception as e:
            log(f"  ERROR processing L={L1}-L={L2}: {e}")

    return crossings


def find_susceptibility_peaks(data_by_size, sizes):
    """Find susceptibility peak positions for each size.

    Parameters
    ----------
    data_by_size : dict
        Dictionary mapping size to DataFrame
    sizes : list
        List of system sizes

    Returns
    -------
    dict
        Peak information for each size
    """
    peaks = {}

    for L in sizes:
        if L not in data_by_size:
            continue

        df = data_by_size[L].sort_values('temperature')

        if 'susceptibility' not in df.columns:
            log(f"  No susceptibility data for L={L}")
            continue

        # Find peak
        idx_peak = df['susceptibility'].idxmax()
        T_peak = df.loc[idx_peak, 'temperature']
        chi_peak = df.loc[idx_peak, 'susceptibility']

        # Estimate error from temperature spacing
        temps = df['temperature'].values
        T_err = (temps[1] - temps[0]) / 2 if len(temps) > 1 else 0.0

        peaks[int(L)] = {
            'T_peak': float(T_peak),
            'T_peak_err': float(T_err),
            'chi_peak': float(chi_peak)
        }
        log(f"  Susceptibility peak L={L}: T={T_peak:.4f}, chi={chi_peak:.4f}")

    return peaks


def extrapolate_Tc_from_peaks(peaks, nu=1.0):
    """Extrapolate Tc from susceptibility peak scaling.

    The peak position scales as: T_peak(L) = Tc + a * L^(-1/nu)

    Parameters
    ----------
    peaks : dict
        Peak information by size
    nu : float
        Correlation length exponent

    Returns
    -------
    dict
        Tc estimate and fit parameters
    """
    if len(peaks) < 2:
        return None

    sizes = np.array(sorted(peaks.keys()))
    T_peaks = np.array([peaks[L]['T_peak'] for L in sizes])

    # Fit: T_peak = Tc + a * L^(-1/nu)
    x = sizes ** (-1 / nu)
    y = T_peaks

    try:
        coeffs = np.polyfit(x, y, 1)
        Tc = coeffs[1]  # Intercept is Tc
        a = coeffs[0]

        # Estimate error from residuals
        y_fit = np.polyval(coeffs, x)
        residuals = y - y_fit
        Tc_err = np.std(residuals) if len(residuals) > 1 else 0.0

        return {
            'Tc': float(Tc),
            'Tc_err': float(Tc_err),
            'amplitude': float(a),
            'nu_assumed': float(nu)
        }
    except Exception as e:
        log(f"  ERROR in peak extrapolation: {e}")
        return None


def main():
    """Find critical temperature from sweep data."""
    input_files = snakemake.input
    output_file = snakemake.output[0]
    method = snakemake.params.get('method', 'binder_crossing')
    sizes = snakemake.params.get('sizes', [])

    log(f"Finding critical temperature using {method} method...")
    log(f"Loading {len(input_files)} sweep files...")

    # Load data for each size
    data_by_size = {}
    for filepath in input_files:
        try:
            df = pd.read_csv(filepath)
            if len(df) == 0:
                log(f"  WARNING: Empty file {Path(filepath).name}")
                continue

            size = int(df['size'].iloc[0])
            data_by_size[size] = df
            log(f"  Loaded L={size}: {len(df)} temperature points")
        except Exception as e:
            log(f"  ERROR loading {Path(filepath).name}: {e}")

    if not data_by_size:
        raise ValueError("No valid data files loaded")

    available_sizes = sorted(data_by_size.keys())
    log(f"Available sizes: {available_sizes}")

    results = {
        'method': method,
        'sizes': available_sizes,
        'n_sizes': len(available_sizes)
    }

    # Binder crossing analysis
    log("Performing Binder crossing analysis...")
    crossings = find_binder_crossings(data_by_size, available_sizes)
    results['binder_crossings'] = crossings

    if crossings:
        T_values = [c['T_cross'] for c in crossings]
        results['Tc_binder'] = float(np.mean(T_values))
        results['Tc_binder_err'] = float(np.std(T_values)) if len(T_values) > 1 else 0.0
        results['U_star'] = float(np.mean([c['U_cross'] for c in crossings]))
        log(f"Binder crossing Tc = {results['Tc_binder']:.4f} +/- {results['Tc_binder_err']:.4f}")

    # Susceptibility peak analysis
    log("Performing susceptibility peak analysis...")
    peaks = find_susceptibility_peaks(data_by_size, available_sizes)
    results['susceptibility_peaks'] = peaks

    if len(peaks) >= 2:
        extrapolation = extrapolate_Tc_from_peaks(peaks)
        if extrapolation:
            results['Tc_chi_extrapolated'] = extrapolation['Tc']
            results['Tc_chi_err'] = extrapolation['Tc_err']
            log(f"Susceptibility extrapolation Tc = {extrapolation['Tc']:.4f}")

    # Determine best Tc estimate
    if method == 'binder_crossing' and crossings:
        results['Tc'] = results['Tc_binder']
        results['Tc_err'] = results.get('Tc_binder_err', 0.0)
        results['Tc_method'] = 'binder_crossing'
    elif 'Tc_chi_extrapolated' in results:
        results['Tc'] = results['Tc_chi_extrapolated']
        results['Tc_err'] = results.get('Tc_chi_err', 0.0)
        results['Tc_method'] = 'susceptibility_extrapolation'
    elif peaks:
        # Fallback: average of susceptibility peaks
        T_peaks = [p['T_peak'] for p in peaks.values()]
        results['Tc'] = float(np.mean(T_peaks))
        results['Tc_err'] = float(np.std(T_peaks)) if len(T_peaks) > 1 else 0.0
        results['Tc_method'] = 'susceptibility_peak_average'
    else:
        results['Tc'] = None
        results['Tc_err'] = None
        results['Tc_method'] = 'failed'
        log("WARNING: Could not determine Tc")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        log(f"Results saved to {output_file}")
    except Exception as e:
        log(f"ERROR writing output: {e}")
        raise

    if results['Tc'] is not None:
        log(f"RESULT: Tc = {results['Tc']:.4f} +/- {results['Tc_err']:.4f} ({results['Tc_method']})")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python find_Tc.py output.json input1.csv input2.csv ...", file=sys.stderr)
        sys.exit(1)

    class MockSnakemake:
        def __init__(self):
            self.input = sys.argv[2:]
            self.output = [sys.argv[1]]
            self.params = {'method': 'binder_crossing', 'sizes': []}

    snakemake = MockSnakemake()
    main()
