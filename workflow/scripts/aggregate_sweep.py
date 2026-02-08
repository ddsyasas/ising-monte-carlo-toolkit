"""
Aggregate observable data across temperatures for a single system size.

This script combines individual temperature point results into a single
sweep file for analysis.
"""

import sys
from pathlib import Path

import pandas as pd


def log(message):
    """Log message to stderr."""
    print(message, file=sys.stderr)


def main():
    """Aggregate observable files into a sweep file."""
    input_files = snakemake.input
    output_file = snakemake.output[0]

    log(f"Aggregating {len(input_files)} temperature point files...")

    # Read all input files
    dfs = []
    failed_files = []

    for filepath in input_files:
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                dfs.append(df)
                log(f"  Loaded {Path(filepath).name}: T={df['temperature'].iloc[0]:.4f}")
            else:
                log(f"  WARNING: Empty file {Path(filepath).name}")
        except Exception as e:
            log(f"  ERROR reading {Path(filepath).name}: {e}")
            failed_files.append((filepath, str(e)))

    if not dfs:
        raise ValueError(f"No valid input files. {len(failed_files)} files failed to load.")

    # Concatenate all dataframes
    combined = pd.concat(dfs, ignore_index=True)

    # Sort by temperature
    combined = combined.sort_values('temperature').reset_index(drop=True)

    # Extract model and size from first row
    model = combined['model'].iloc[0] if 'model' in combined.columns else 'unknown'
    size = combined['size'].iloc[0] if 'size' in combined.columns else 0

    log(f"  Model: {model}, Size: {size}")
    log(f"  Temperature range: [{combined['temperature'].min():.4f}, {combined['temperature'].max():.4f}]")
    log(f"  Total points: {len(combined)}")

    # Add sweep metadata columns
    combined['n_temps'] = len(combined)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write combined results
    try:
        combined.to_csv(output_file, index=False)
        log(f"Saved aggregated sweep to {output_file}")
    except Exception as e:
        log(f"ERROR writing output: {e}")
        raise

    if failed_files:
        log(f"WARNING: {len(failed_files)} files failed to load:")
        for fp, err in failed_files[:5]:
            log(f"  - {Path(fp).name}: {err}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python aggregate_sweep.py output.csv input1.csv input2.csv ...", file=sys.stderr)
        sys.exit(1)

    class MockSnakemake:
        def __init__(self):
            self.input = sys.argv[2:]
            self.output = [sys.argv[1]]

    snakemake = MockSnakemake()
    main()
