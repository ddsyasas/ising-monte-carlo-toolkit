"""
Aggregate observable data across temperatures for a single system size.

This script combines individual temperature point results into a single
sweep file for analysis.
"""

import csv
from pathlib import Path

import pandas as pd


def main():
    """Aggregate observable files into a sweep file."""
    input_files = snakemake.input
    output_file = snakemake.output[0]

    # Read all input files
    dfs = []
    for filepath in input_files:
        try:
            df = pd.read_csv(filepath)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {filepath}: {e}")

    if not dfs:
        raise ValueError("No valid input files")

    # Concatenate all dataframes
    combined = pd.concat(dfs, ignore_index=True)

    # Sort by temperature
    combined = combined.sort_values('temperature').reset_index(drop=True)

    # Extract model and size from first row
    model = combined['model'].iloc[0]
    size = combined['size'].iloc[0]

    # Add sweep metadata
    combined['temp_min'] = combined['temperature'].min()
    combined['temp_max'] = combined['temperature'].max()
    combined['n_temps'] = len(combined)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write combined results
    combined.to_csv(output_file, index=False)

    print(f"Aggregated {len(combined)} temperature points to {output_file}")
    print(f"  Model: {model}, Size: {size}")
    print(f"  T range: [{combined['temperature'].min():.4f}, {combined['temperature'].max():.4f}]")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = sys.argv[2:]
            self.output = [sys.argv[1]]

    snakemake = MockSnakemake()
    main()
