"""
Aggregate Binder cumulant data from all simulations.
"""

from pathlib import Path

import pandas as pd


def main():
    """Aggregate Binder cumulant files."""
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

    # Sort by size, then temperature
    combined = combined.sort_values(['size', 'temperature']).reset_index(drop=True)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write combined results
    combined.to_csv(output_file, index=False)

    print(f"Aggregated {len(combined)} Binder cumulant values to {output_file}")
    print(f"  Sizes: {sorted(combined['size'].unique())}")


if __name__ == '__main__':
    import sys

    class MockSnakemake:
        def __init__(self):
            self.input = sys.argv[2:]
            self.output = [sys.argv[1]]

    snakemake = MockSnakemake()
    main()
