"""Command-line interface for the Ising Monte Carlo toolkit."""

import argparse


def main():
    """Main entry point for the ising-sim CLI."""
    parser = argparse.ArgumentParser(
        description="Ising Monte Carlo Toolkit - Run Ising model simulations"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )

    args = parser.parse_args()

    print("Ising Monte Carlo Toolkit v0.1.0")
    print("Use --help for available options.")


if __name__ == "__main__":
    main()
