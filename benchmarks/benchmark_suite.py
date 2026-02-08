#!/usr/bin/env python
"""
Comprehensive benchmark suite for Ising Monte Carlo toolkit.

This suite measures:
1. Metropolis algorithm scaling with system size
2. Wolff cluster algorithm performance
3. Wolff vs Metropolis comparison at critical temperature
4. Numba acceleration speedup
5. Parallel execution scaling
6. Memory usage and compression efficiency

Run with: python benchmarks/benchmark_suite.py [--quick] [--output results.csv]
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from ising_toolkit.models import Ising1D, Ising2D, Ising3D
from ising_toolkit.samplers import MetropolisSampler, WolffSampler
from ising_toolkit.utils.numba_kernels import NUMBA_AVAILABLE
from ising_toolkit.utils.parallel import parallel_map, run_temperature_sweep_parallel
from ising_toolkit.io.compression import pack_spins, get_compression_ratio


# =============================================================================
# Utility functions
# =============================================================================

def format_time(seconds: float) -> str:
    """Format time duration as human-readable string."""
    if seconds < 1e-6:
        return f"{seconds*1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.2f} us"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def format_rate(rate: float, unit: str = "ops") -> str:
    """Format rate as human-readable string."""
    if rate > 1e9:
        return f"{rate/1e9:.2f} G{unit}/s"
    elif rate > 1e6:
        return f"{rate/1e6:.2f} M{unit}/s"
    elif rate > 1e3:
        return f"{rate/1e3:.2f} K{unit}/s"
    else:
        return f"{rate:.2f} {unit}/s"


def warmup_numba():
    """Warm up Numba JIT compilation."""
    if NUMBA_AVAILABLE:
        model = Ising2D(size=8, temperature=2.269, use_numba=True)
        model.initialize('random')
        for _ in range(10):
            model.metropolis_sweep()
            model.wolff_step()


# =============================================================================
# Benchmark functions
# =============================================================================

def benchmark_metropolis_scaling(
    sizes: List[int] = None,
    n_steps: int = 1000,
    temperature: float = 2.269,
    use_numba: bool = True,
) -> List[Dict]:
    """Benchmark Metropolis performance vs system size.

    Parameters
    ----------
    sizes : list of int
        System sizes to benchmark.
    n_steps : int
        Number of MC sweeps per size.
    temperature : float
        Temperature for simulation.
    use_numba : bool
        Whether to use Numba acceleration.

    Returns
    -------
    list of dict
        Benchmark results.
    """
    if sizes is None:
        sizes = [16, 32, 64, 128, 256]

    print("\n" + "=" * 60)
    print("METROPOLIS SCALING BENCHMARK")
    print("=" * 60)
    print(f"Temperature: {temperature:.4f}, Numba: {use_numba and NUMBA_AVAILABLE}")
    print(f"Steps per size: {n_steps}")
    print("-" * 60)

    results = []

    for L in sizes:
        model = Ising2D(size=L, temperature=temperature, use_numba=use_numba)
        model.initialize('random')
        sampler = MetropolisSampler(model, use_fast_sweep=True)

        # Warmup
        for _ in range(5):
            sampler.step()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_steps):
            sampler.step()
        elapsed = time.perf_counter() - start

        n_spins = L * L
        time_per_sweep = elapsed / n_steps
        time_per_spin = elapsed / (n_steps * n_spins)
        spin_flips_per_sec = n_steps * n_spins / elapsed

        result = {
            'benchmark': 'metropolis_scaling',
            'size': L,
            'n_spins': n_spins,
            'n_steps': n_steps,
            'total_time': elapsed,
            'time_per_sweep': time_per_sweep,
            'time_per_spin': time_per_spin,
            'spin_flips_per_sec': spin_flips_per_sec,
            'use_numba': use_numba and NUMBA_AVAILABLE,
        }
        results.append(result)

        print(f"L={L:4d}: {format_time(time_per_sweep)}/sweep, "
              f"{format_rate(spin_flips_per_sec, 'flips')}")

    # Check scaling
    if len(results) >= 2:
        times = [r['time_per_sweep'] for r in results]
        n_spins = [r['n_spins'] for r in results]
        # Time should scale linearly with n_spins
        ratio = (times[-1] / times[0]) / (n_spins[-1] / n_spins[0])
        print(f"\nScaling efficiency: {1/ratio:.2f} (1.0 = perfect linear)")

    return results


def benchmark_wolff_scaling(
    sizes: List[int] = None,
    n_clusters: int = 1000,
    temperature: float = 2.269,
    use_numba: bool = True,
) -> List[Dict]:
    """Benchmark Wolff cluster algorithm vs system size.

    Parameters
    ----------
    sizes : list of int
        System sizes to benchmark.
    n_clusters : int
        Number of cluster updates per size.
    temperature : float
        Temperature (near Tc for large clusters).
    use_numba : bool
        Whether to use Numba acceleration.

    Returns
    -------
    list of dict
        Benchmark results.
    """
    if sizes is None:
        sizes = [16, 32, 64, 128]

    print("\n" + "=" * 60)
    print("WOLFF CLUSTER SCALING BENCHMARK")
    print("=" * 60)
    print(f"Temperature: {temperature:.4f} (Tc ≈ 2.269)")
    print(f"Clusters per size: {n_clusters}")
    print("-" * 60)

    results = []

    for L in sizes:
        model = Ising2D(size=L, temperature=temperature, use_numba=use_numba)
        model.initialize('random')

        # Warmup
        for _ in range(10):
            model.wolff_step()

        # Benchmark
        cluster_sizes = []
        start = time.perf_counter()
        for _ in range(n_clusters):
            cluster_sizes.append(model.wolff_step())
        elapsed = time.perf_counter() - start

        n_spins = L * L
        avg_cluster = np.mean(cluster_sizes)
        time_per_cluster = elapsed / n_clusters
        clusters_per_sec = n_clusters / elapsed

        result = {
            'benchmark': 'wolff_scaling',
            'size': L,
            'n_spins': n_spins,
            'n_clusters': n_clusters,
            'total_time': elapsed,
            'time_per_cluster': time_per_cluster,
            'avg_cluster_size': avg_cluster,
            'avg_cluster_fraction': avg_cluster / n_spins,
            'clusters_per_sec': clusters_per_sec,
            'use_numba': use_numba and NUMBA_AVAILABLE,
        }
        results.append(result)

        print(f"L={L:4d}: {format_time(time_per_cluster)}/cluster, "
              f"avg size: {avg_cluster:.0f} ({100*avg_cluster/n_spins:.1f}%)")

    return results


def benchmark_wolff_vs_metropolis(
    size: int = 64,
    n_sweeps: int = 1000,
    temperatures: List[float] = None,
) -> List[Dict]:
    """Compare Wolff and Metropolis at various temperatures.

    Parameters
    ----------
    size : int
        System size.
    n_sweeps : int
        Number of MC sweeps for Metropolis (Wolff does equivalent work).
    temperatures : list of float
        Temperatures to compare.

    Returns
    -------
    list of dict
        Comparison results.
    """
    if temperatures is None:
        temperatures = [1.5, 2.0, 2.269, 2.5, 3.0, 4.0]

    print("\n" + "=" * 60)
    print("WOLFF VS METROPOLIS COMPARISON")
    print("=" * 60)
    print(f"Size: {size}x{size}, Sweeps: {n_sweeps}")
    print("-" * 60)

    results = []
    n_spins = size * size

    for T in temperatures:
        # Metropolis
        model_metro = Ising2D(size=size, temperature=T, use_numba=True)
        model_metro.initialize('random')
        sampler_metro = MetropolisSampler(model_metro, use_fast_sweep=True)

        start = time.perf_counter()
        for _ in range(n_sweeps):
            sampler_metro.step()
        metro_time = time.perf_counter() - start

        # Wolff - do equivalent number of spin updates
        model_wolff = Ising2D(size=size, temperature=T, use_numba=True)
        model_wolff.initialize('random')

        total_flipped = 0
        wolff_clusters = 0
        start = time.perf_counter()
        while total_flipped < n_sweeps * n_spins:
            total_flipped += model_wolff.wolff_step()
            wolff_clusters += 1
        wolff_time = time.perf_counter() - start

        # Measure decorrelation (simple autocorrelation proxy)
        metro_acceptance = sampler_metro.acceptance_rate

        speedup = metro_time / wolff_time if wolff_time > 0 else float('inf')

        result = {
            'benchmark': 'wolff_vs_metropolis',
            'size': size,
            'temperature': T,
            'metro_time': metro_time,
            'wolff_time': wolff_time,
            'speedup': speedup,
            'metro_acceptance': metro_acceptance,
            'wolff_clusters': wolff_clusters,
            'avg_cluster_size': n_sweeps * n_spins / wolff_clusters if wolff_clusters > 0 else 0,
        }
        results.append(result)

        print(f"T={T:.3f}: Metro {format_time(metro_time)}, "
              f"Wolff {format_time(wolff_time)}, "
              f"Speedup: {speedup:.1f}x")

    return results


def benchmark_numba_speedup(
    sizes: List[int] = None,
    n_steps: int = 100,
) -> List[Dict]:
    """Measure Numba vs pure Python speedup.

    Parameters
    ----------
    sizes : list of int
        System sizes to benchmark.
    n_steps : int
        Number of MC sweeps.

    Returns
    -------
    list of dict
        Speedup results.
    """
    if sizes is None:
        sizes = [16, 32, 64]

    if not NUMBA_AVAILABLE:
        print("\n" + "=" * 60)
        print("NUMBA SPEEDUP BENCHMARK - SKIPPED (Numba not available)")
        print("=" * 60)
        return []

    print("\n" + "=" * 60)
    print("NUMBA SPEEDUP BENCHMARK")
    print("=" * 60)
    print(f"Steps per size: {n_steps}")
    print("-" * 60)

    results = []

    for L in sizes:
        # Python version
        model_py = Ising2D(size=L, temperature=2.269, use_numba=False)
        model_py.initialize('random')

        # Fewer steps for Python (it's slow)
        py_steps = min(n_steps, 10)

        start = time.perf_counter()
        for _ in range(py_steps):
            model_py.metropolis_sweep()
        py_time = (time.perf_counter() - start) * (n_steps / py_steps)

        # Numba version
        model_numba = Ising2D(size=L, temperature=2.269, use_numba=True)
        model_numba.initialize('random')

        # Warmup
        for _ in range(5):
            model_numba.metropolis_sweep()

        start = time.perf_counter()
        for _ in range(n_steps):
            model_numba.metropolis_sweep()
        numba_time = time.perf_counter() - start

        speedup = py_time / numba_time

        result = {
            'benchmark': 'numba_speedup',
            'size': L,
            'n_spins': L * L,
            'python_time': py_time,
            'numba_time': numba_time,
            'speedup': speedup,
        }
        results.append(result)

        print(f"L={L:4d}: Python {format_time(py_time)}, "
              f"Numba {format_time(numba_time)}, "
              f"Speedup: {speedup:.0f}x")

    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage speedup: {avg_speedup:.0f}x")

    return results


def benchmark_parallel_scaling(
    n_temperatures: int = 20,
    size: int = 32,
    n_steps: int = 500,
    max_workers: int = None,
) -> List[Dict]:
    """Measure parallel efficiency vs number of workers.

    Parameters
    ----------
    n_temperatures : int
        Number of temperature points.
    size : int
        System size.
    n_steps : int
        MC steps per temperature.
    max_workers : int
        Maximum workers to test.

    Returns
    -------
    list of dict
        Parallel scaling results.
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    print("\n" + "=" * 60)
    print("PARALLEL SCALING BENCHMARK")
    print("=" * 60)
    print(f"Temperature points: {n_temperatures}, Size: {size}x{size}")
    print(f"Steps per point: {n_steps}")
    print("-" * 60)

    temperatures = np.linspace(1.5, 3.5, n_temperatures).tolist()
    worker_counts = [1] + list(range(2, max_workers + 1, 2))
    if max_workers not in worker_counts:
        worker_counts.append(max_workers)

    results = []
    baseline_time = None

    for n_workers in worker_counts:
        start = time.perf_counter()
        sweep_results = run_temperature_sweep_parallel(
            model_class='Ising2D',
            size=size,
            temperatures=temperatures,
            n_steps=n_steps,
            equilibration=n_steps // 4,
            n_workers=n_workers,
            progress=False,
        )
        elapsed = time.perf_counter() - start

        if baseline_time is None:
            baseline_time = elapsed

        speedup = baseline_time / elapsed
        efficiency = speedup / n_workers

        result = {
            'benchmark': 'parallel_scaling',
            'n_workers': n_workers,
            'n_temperatures': n_temperatures,
            'total_time': elapsed,
            'speedup': speedup,
            'efficiency': efficiency,
            'time_per_point': elapsed / n_temperatures,
        }
        results.append(result)

        print(f"Workers={n_workers:2d}: {format_time(elapsed)}, "
              f"Speedup: {speedup:.2f}x, "
              f"Efficiency: {efficiency*100:.0f}%")

    return results


def benchmark_energy_calculation(
    sizes: List[int] = None,
    n_repeats: int = 1000,
) -> List[Dict]:
    """Benchmark energy calculation performance.

    Parameters
    ----------
    sizes : list of int
        System sizes.
    n_repeats : int
        Number of energy calculations.

    Returns
    -------
    list of dict
        Results.
    """
    if sizes is None:
        sizes = [32, 64, 128, 256]

    print("\n" + "=" * 60)
    print("ENERGY CALCULATION BENCHMARK")
    print("=" * 60)
    print(f"Repeats per size: {n_repeats}")
    print("-" * 60)

    results = []

    for L in sizes:
        model = Ising2D(size=L, temperature=2.269, use_numba=True)
        model.initialize('random')

        # Warmup
        for _ in range(10):
            _ = model.get_energy()

        start = time.perf_counter()
        for _ in range(n_repeats):
            _ = model.get_energy()
        elapsed = time.perf_counter() - start

        time_per_calc = elapsed / n_repeats
        calcs_per_sec = n_repeats / elapsed

        result = {
            'benchmark': 'energy_calculation',
            'size': L,
            'n_spins': L * L,
            'n_repeats': n_repeats,
            'time_per_calc': time_per_calc,
            'calcs_per_sec': calcs_per_sec,
        }
        results.append(result)

        print(f"L={L:4d}: {format_time(time_per_calc)}/calc, "
              f"{format_rate(calcs_per_sec, 'calcs')}")

    return results


def benchmark_compression(
    sizes: List[int] = None,
    n_configs: int = 100,
) -> List[Dict]:
    """Benchmark spin compression performance.

    Parameters
    ----------
    sizes : list of int
        System sizes.
    n_configs : int
        Number of configurations to compress.

    Returns
    -------
    list of dict
        Results.
    """
    if sizes is None:
        sizes = [32, 64, 128, 256]

    print("\n" + "=" * 60)
    print("COMPRESSION BENCHMARK")
    print("=" * 60)
    print(f"Configurations per size: {n_configs}")
    print("-" * 60)

    results = []

    for L in sizes:
        configs = [
            np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
            for _ in range(n_configs)
        ]

        # Benchmark packing
        start = time.perf_counter()
        packed = [pack_spins(c) for c in configs]
        pack_time = time.perf_counter() - start

        ratio = get_compression_ratio(configs[0])
        original_size = sum(c.nbytes for c in configs)
        packed_size = sum(p.nbytes for p in packed)

        result = {
            'benchmark': 'compression',
            'size': L,
            'n_spins': L * L,
            'n_configs': n_configs,
            'compression_ratio': ratio,
            'original_mb': original_size / (1024 * 1024),
            'packed_mb': packed_size / (1024 * 1024),
            'pack_time': pack_time,
            'pack_rate_mb_per_sec': original_size / pack_time / (1024 * 1024),
        }
        results.append(result)

        print(f"L={L:4d}: {ratio:.1f}x compression, "
              f"{result['pack_rate_mb_per_sec']:.0f} MB/s")

    return results


def benchmark_3d_scaling(
    sizes: List[int] = None,
    n_steps: int = 100,
) -> List[Dict]:
    """Benchmark 3D Ising model scaling.

    Parameters
    ----------
    sizes : list of int
        System sizes.
    n_steps : int
        MC sweeps.

    Returns
    -------
    list of dict
        Results.
    """
    if sizes is None:
        sizes = [8, 16, 32]

    print("\n" + "=" * 60)
    print("3D ISING MODEL SCALING")
    print("=" * 60)
    print(f"Steps per size: {n_steps}")
    print("-" * 60)

    results = []

    for L in sizes:
        model = Ising3D(size=L, temperature=4.5)
        model.initialize('random')
        sampler = MetropolisSampler(model, use_fast_sweep=True)

        start = time.perf_counter()
        for _ in range(n_steps):
            sampler.step()
        elapsed = time.perf_counter() - start

        n_spins = L ** 3
        time_per_sweep = elapsed / n_steps
        spin_flips_per_sec = n_steps * n_spins / elapsed

        result = {
            'benchmark': '3d_scaling',
            'size': L,
            'n_spins': n_spins,
            'n_steps': n_steps,
            'total_time': elapsed,
            'time_per_sweep': time_per_sweep,
            'spin_flips_per_sec': spin_flips_per_sec,
        }
        results.append(result)

        print(f"L={L:4d} ({n_spins:6d} spins): {format_time(time_per_sweep)}/sweep, "
              f"{format_rate(spin_flips_per_sec, 'flips')}")

    return results


# =============================================================================
# Main benchmark runner
# =============================================================================

def run_all_benchmarks(quick: bool = False) -> Dict[str, List[Dict]]:
    """Run all benchmarks and generate report.

    Parameters
    ----------
    quick : bool
        If True, run quick benchmarks with fewer iterations.

    Returns
    -------
    dict
        All benchmark results.
    """
    print("=" * 60)
    print("ISING MONTE CARLO TOOLKIT - BENCHMARK SUITE")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"Quick mode: {quick}")

    # Warmup Numba
    if NUMBA_AVAILABLE:
        print("\nWarming up Numba JIT compilation...")
        warmup_numba()

    # Adjust parameters for quick mode
    if quick:
        metro_sizes = [16, 32, 64]
        metro_steps = 100
        wolff_sizes = [16, 32]
        wolff_clusters = 100
        comparison_sweeps = 100
        numba_steps = 50
        parallel_temps = 8
        parallel_steps = 100
        energy_repeats = 100
        compression_configs = 10
        scaling_3d_sizes = [8, 16]
        scaling_3d_steps = 50
    else:
        metro_sizes = [16, 32, 64, 128, 256]
        metro_steps = 1000
        wolff_sizes = [16, 32, 64, 128]
        wolff_clusters = 1000
        comparison_sweeps = 1000
        numba_steps = 100
        parallel_temps = 20
        parallel_steps = 500
        energy_repeats = 1000
        compression_configs = 100
        scaling_3d_sizes = [8, 16, 32]
        scaling_3d_steps = 100

    all_results = {}

    # Run benchmarks
    all_results['metropolis_scaling'] = benchmark_metropolis_scaling(
        sizes=metro_sizes, n_steps=metro_steps
    )

    all_results['wolff_scaling'] = benchmark_wolff_scaling(
        sizes=wolff_sizes, n_clusters=wolff_clusters
    )

    all_results['wolff_vs_metropolis'] = benchmark_wolff_vs_metropolis(
        n_sweeps=comparison_sweeps
    )

    all_results['numba_speedup'] = benchmark_numba_speedup(
        sizes=[16, 32, 64], n_steps=numba_steps
    )

    all_results['parallel_scaling'] = benchmark_parallel_scaling(
        n_temperatures=parallel_temps, n_steps=parallel_steps
    )

    all_results['energy_calculation'] = benchmark_energy_calculation(
        n_repeats=energy_repeats
    )

    all_results['compression'] = benchmark_compression(
        n_configs=compression_configs
    )

    all_results['3d_scaling'] = benchmark_3d_scaling(
        sizes=scaling_3d_sizes, n_steps=scaling_3d_steps
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if all_results.get('numba_speedup'):
        avg_speedup = np.mean([r['speedup'] for r in all_results['numba_speedup']])
        print(f"Average Numba speedup: {avg_speedup:.0f}x")

    if all_results.get('parallel_scaling'):
        max_workers = max(r['n_workers'] for r in all_results['parallel_scaling'])
        max_speedup = max(r['speedup'] for r in all_results['parallel_scaling'])
        print(f"Max parallel speedup ({max_workers} workers): {max_speedup:.2f}x")

    if all_results.get('compression'):
        avg_ratio = np.mean([r['compression_ratio'] for r in all_results['compression']])
        print(f"Average compression ratio: {avg_ratio:.1f}x")

    if all_results.get('metropolis_scaling'):
        max_rate = max(r['spin_flips_per_sec'] for r in all_results['metropolis_scaling'])
        print(f"Peak Metropolis rate: {format_rate(max_rate, 'flips')}")

    return all_results


def save_results(results: Dict[str, List[Dict]], output_path: str):
    """Save benchmark results to CSV files.

    Parameters
    ----------
    results : dict
        Benchmark results.
    output_path : str
        Output directory or file path.
    """
    if not HAS_PANDAS:
        print("Warning: pandas not available, cannot save to CSV")
        return

    output_path = Path(output_path)

    if output_path.suffix == '.csv':
        # Single file with all results
        all_rows = []
        for benchmark_name, benchmark_results in results.items():
            for row in benchmark_results:
                row = row.copy()
                row['benchmark'] = benchmark_name
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        # Directory with separate files
        output_path.mkdir(parents=True, exist_ok=True)

        for benchmark_name, benchmark_results in results.items():
            if benchmark_results:
                df = pd.DataFrame(benchmark_results)
                filepath = output_path / f"{benchmark_name}.csv"
                df.to_csv(filepath, index=False)

        print(f"\nResults saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(
        description="Run Ising model benchmark suite"
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help="Run quick benchmarks with fewer iterations"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Output path for results (CSV file or directory)"
    )
    parser.add_argument(
        '--benchmark', '-b',
        type=str,
        default=None,
        choices=[
            'metropolis', 'wolff', 'comparison', 'numba',
            'parallel', 'energy', 'compression', '3d'
        ],
        help="Run only a specific benchmark"
    )

    args = parser.parse_args()

    if args.benchmark:
        # Run single benchmark
        warmup_numba()

        if args.benchmark == 'metropolis':
            results = {'metropolis_scaling': benchmark_metropolis_scaling()}
        elif args.benchmark == 'wolff':
            results = {'wolff_scaling': benchmark_wolff_scaling()}
        elif args.benchmark == 'comparison':
            results = {'wolff_vs_metropolis': benchmark_wolff_vs_metropolis()}
        elif args.benchmark == 'numba':
            results = {'numba_speedup': benchmark_numba_speedup()}
        elif args.benchmark == 'parallel':
            results = {'parallel_scaling': benchmark_parallel_scaling()}
        elif args.benchmark == 'energy':
            results = {'energy_calculation': benchmark_energy_calculation()}
        elif args.benchmark == 'compression':
            results = {'compression': benchmark_compression()}
        elif args.benchmark == '3d':
            results = {'3d_scaling': benchmark_3d_scaling()}
    else:
        # Run all benchmarks
        results = run_all_benchmarks(quick=args.quick)

    if args.output:
        save_results(results, args.output)

    print("\nBenchmark suite complete!")


if __name__ == '__main__':
    main()
