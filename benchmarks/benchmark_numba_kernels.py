#!/usr/bin/env python
"""
Benchmark Numba-optimized kernels vs pure Python implementations.

This script compares performance of:
1. Metropolis sweeps (1D, 2D, 3D)
2. Wolff cluster updates (1D, 2D, 3D)
3. Energy calculations (1D, 2D, 3D)
4. Parallel vs serial implementations

Usage:
    python benchmarks/benchmark_numba_kernels.py
    python benchmarks/benchmark_numba_kernels.py --sizes 16 32 64 128
    python benchmarks/benchmark_numba_kernels.py --output results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np

# Add src to path for development
sys.path.insert(0, 'src')

from ising_toolkit.utils.numba_kernels import (
    NUMBA_AVAILABLE,
    # Numba kernels
    _metropolis_sweep_1d,
    _metropolis_sweep_2d,
    _metropolis_sweep_3d,
    _calculate_energy_1d,
    _calculate_energy_2d,
    _calculate_energy_3d,
    _calculate_energy_2d_parallel,
    _calculate_energy_3d_parallel,
    _wolff_cluster_1d,
    _wolff_cluster_2d,
    _wolff_cluster_3d,
    # Pure Python
    _metropolis_sweep_2d_python,
    _metropolis_sweep_3d_python,
    _calculate_energy_2d_python,
    # Precomputation
    precompute_acceptance_probs_1d,
    precompute_acceptance_probs_2d,
    precompute_acceptance_probs_3d,
)


def warmup_jit():
    """Trigger JIT compilation for all kernels."""
    if not NUMBA_AVAILABLE:
        return

    print("Warming up JIT compilation...")

    # 1D
    spins_1d = np.random.choice([-1, 1], size=100).astype(np.int8)
    probs_1d = precompute_acceptance_probs_1d(0.5)
    _metropolis_sweep_1d(spins_1d.copy(), 0.5, probs_1d)
    _calculate_energy_1d(spins_1d)
    _wolff_cluster_1d(spins_1d.copy(), 0.5, 0)

    # 2D
    spins_2d = np.random.choice([-1, 1], size=(10, 10)).astype(np.int8)
    probs_2d = precompute_acceptance_probs_2d(0.5)
    _metropolis_sweep_2d(spins_2d.copy(), 0.5, probs_2d)
    _calculate_energy_2d(spins_2d)
    _calculate_energy_2d_parallel(spins_2d)
    _wolff_cluster_2d(spins_2d.copy(), 0.5, 0, 0)

    # 3D
    spins_3d = np.random.choice([-1, 1], size=(5, 5, 5)).astype(np.int8)
    probs_3d = precompute_acceptance_probs_3d(0.5)
    _metropolis_sweep_3d(spins_3d.copy(), 0.5, probs_3d)
    _calculate_energy_3d(spins_3d)
    _calculate_energy_3d_parallel(spins_3d)
    _wolff_cluster_3d(spins_3d.copy(), 0.5, 0, 0, 0)

    print("JIT warmup complete.\n")


def benchmark_metropolis(sizes, n_sweeps=100, temperature=2.269):
    """Benchmark Metropolis algorithm implementations."""
    beta = 1.0 / temperature
    results = []

    print(f"\n{'='*70}")
    print("METROPOLIS SWEEP BENCHMARK")
    print(f"Temperature: {temperature}, Sweeps: {n_sweeps}")
    print(f"{'='*70}")

    # 2D benchmarks
    print("\n2D Ising Model:")
    print(f"{'L':>6} {'N sites':>10} {'Numba (s)':>12} {'Python (s)':>12} {'Speedup':>10}")
    print("-" * 54)

    for L in sizes:
        spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
        probs = precompute_acceptance_probs_2d(beta)
        n_sites = L * L

        # Numba timing
        if NUMBA_AVAILABLE:
            start = time.perf_counter()
            for _ in range(n_sweeps):
                _metropolis_sweep_2d(spins.copy(), beta, probs)
            numba_time = time.perf_counter() - start
        else:
            numba_time = float('inf')

        # Python timing (fewer sweeps for large systems)
        python_sweeps = min(n_sweeps, max(1, 1000 // n_sites))
        spins_py = spins.copy()
        start = time.perf_counter()
        for _ in range(python_sweeps):
            _metropolis_sweep_2d_python(spins_py, beta, probs)
        python_time = (time.perf_counter() - start) * (n_sweeps / python_sweeps)

        speedup = python_time / numba_time if numba_time > 0 else 0

        print(f"{L:>6} {n_sites:>10} {numba_time:>12.4f} {python_time:>12.4f} {speedup:>10.1f}x")

        results.append({
            'dimension': 2,
            'L': L,
            'n_sites': n_sites,
            'n_sweeps': n_sweeps,
            'numba_time': numba_time,
            'python_time': python_time,
            'speedup': speedup,
            'numba_flips_per_sec': n_sweeps * n_sites / numba_time if numba_time > 0 else 0,
        })

    # 3D benchmarks
    sizes_3d = [s for s in sizes if s <= 32]  # Limit 3D sizes
    if sizes_3d:
        print("\n3D Ising Model:")
        print(f"{'L':>6} {'N sites':>10} {'Numba (s)':>12} {'Python (s)':>12} {'Speedup':>10}")
        print("-" * 54)

        for L in sizes_3d:
            spins = np.random.choice([-1, 1], size=(L, L, L)).astype(np.int8)
            probs = precompute_acceptance_probs_3d(beta)
            n_sites = L ** 3
            sweeps_3d = max(1, n_sweeps // 10)

            if NUMBA_AVAILABLE:
                start = time.perf_counter()
                for _ in range(sweeps_3d):
                    _metropolis_sweep_3d(spins.copy(), beta, probs)
                numba_time = time.perf_counter() - start
            else:
                numba_time = float('inf')

            python_sweeps = min(sweeps_3d, max(1, 100 // (L ** 2)))
            spins_py = spins.copy()
            start = time.perf_counter()
            for _ in range(python_sweeps):
                _metropolis_sweep_3d_python(spins_py, beta, probs)
            python_time = (time.perf_counter() - start) * (sweeps_3d / python_sweeps)

            speedup = python_time / numba_time if numba_time > 0 else 0

            print(f"{L:>6} {n_sites:>10} {numba_time:>12.4f} {python_time:>12.4f} {speedup:>10.1f}x")

            results.append({
                'dimension': 3,
                'L': L,
                'n_sites': n_sites,
                'n_sweeps': sweeps_3d,
                'numba_time': numba_time,
                'python_time': python_time,
                'speedup': speedup,
            })

    return results


def benchmark_energy(sizes):
    """Benchmark energy calculation implementations."""
    results = []

    print(f"\n{'='*70}")
    print("ENERGY CALCULATION BENCHMARK")
    print(f"{'='*70}")

    # 2D benchmarks
    print("\n2D Ising Model:")
    print(f"{'L':>6} {'Numba (ms)':>12} {'Parallel (ms)':>14} {'Python (ms)':>12} {'Speedup':>10}")
    print("-" * 58)

    n_repeats = 1000

    for L in sizes:
        spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

        # Numba serial
        if NUMBA_AVAILABLE:
            start = time.perf_counter()
            for _ in range(n_repeats):
                _calculate_energy_2d(spins)
            numba_time = (time.perf_counter() - start) / n_repeats * 1000
        else:
            numba_time = float('inf')

        # Numba parallel
        if NUMBA_AVAILABLE:
            start = time.perf_counter()
            for _ in range(n_repeats):
                _calculate_energy_2d_parallel(spins)
            parallel_time = (time.perf_counter() - start) / n_repeats * 1000
        else:
            parallel_time = float('inf')

        # Python
        py_repeats = min(n_repeats, 100)
        start = time.perf_counter()
        for _ in range(py_repeats):
            _calculate_energy_2d_python(spins)
        python_time = (time.perf_counter() - start) / py_repeats * 1000

        speedup = python_time / numba_time if numba_time > 0 else 0

        print(f"{L:>6} {numba_time:>12.4f} {parallel_time:>14.4f} {python_time:>12.4f} {speedup:>10.1f}x")

        results.append({
            'dimension': 2,
            'L': L,
            'numba_time_ms': numba_time,
            'parallel_time_ms': parallel_time,
            'python_time_ms': python_time,
            'speedup': speedup,
        })

    return results


def benchmark_wolff(sizes, n_clusters=100, temperature=2.269):
    """Benchmark Wolff cluster algorithm implementations."""
    beta = 1.0 / temperature
    p_add = 1.0 - np.exp(-2.0 * beta)

    results = []

    print(f"\n{'='*70}")
    print("WOLFF CLUSTER BENCHMARK")
    print(f"Temperature: {temperature} (near Tc), Clusters: {n_clusters}")
    print(f"{'='*70}")

    print("\n2D Ising Model:")
    print(f"{'L':>6} {'Time (s)':>12} {'Avg Cluster':>12} {'Clusters/s':>12}")
    print("-" * 46)

    for L in sizes:
        spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
        total_cluster_size = 0

        if NUMBA_AVAILABLE:
            start = time.perf_counter()
            for _ in range(n_clusters):
                seed_i = np.random.randint(0, L)
                seed_j = np.random.randint(0, L)
                cluster_size = _wolff_cluster_2d(spins, p_add, seed_i, seed_j)
                total_cluster_size += cluster_size
            elapsed = time.perf_counter() - start
        else:
            elapsed = float('inf')
            total_cluster_size = 0

        avg_cluster = total_cluster_size / n_clusters if n_clusters > 0 else 0
        clusters_per_sec = n_clusters / elapsed if elapsed > 0 else 0

        print(f"{L:>6} {elapsed:>12.4f} {avg_cluster:>12.1f} {clusters_per_sec:>12.1f}")

        results.append({
            'dimension': 2,
            'L': L,
            'n_clusters': n_clusters,
            'time': elapsed,
            'avg_cluster_size': avg_cluster,
            'clusters_per_sec': clusters_per_sec,
        })

    return results


def benchmark_scaling(max_size=256, n_sweeps=10):
    """Benchmark scaling with system size."""
    sizes = [8, 16, 32, 64, 128]
    if max_size >= 256:
        sizes.append(256)

    beta = 1.0 / 2.269  # Critical temperature
    results = []

    print(f"\n{'='*70}")
    print("SCALING ANALYSIS")
    print(f"{'='*70}")

    print("\nMetropolis sweep time scaling (2D):")
    print(f"{'L':>6} {'N':>10} {'Time (s)':>12} {'Time/N (ns)':>12}")
    print("-" * 44)

    for L in sizes:
        spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
        probs = precompute_acceptance_probs_2d(beta)
        n_sites = L * L

        if NUMBA_AVAILABLE:
            start = time.perf_counter()
            for _ in range(n_sweeps):
                _metropolis_sweep_2d(spins, beta, probs)
            elapsed = time.perf_counter() - start
        else:
            elapsed = float('inf')

        time_per_site = elapsed / (n_sweeps * n_sites) * 1e9  # nanoseconds

        print(f"{L:>6} {n_sites:>10} {elapsed:>12.4f} {time_per_site:>12.2f}")

        results.append({
            'L': L,
            'N': n_sites,
            'total_time': elapsed,
            'time_per_site_ns': time_per_site,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Numba kernels')
    parser.add_argument('--sizes', type=int, nargs='+', default=[8, 16, 32, 64],
                        help='System sizes to benchmark')
    parser.add_argument('--sweeps', type=int, default=100,
                        help='Number of MC sweeps')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--quick', action='store_true',
                        help='Quick benchmark with fewer iterations')

    args = parser.parse_args()

    print("=" * 70)
    print("NUMBA KERNEL BENCHMARKS")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"Sizes: {args.sizes}")
    print(f"Sweeps: {args.sweeps}")

    if not NUMBA_AVAILABLE:
        print("\nWARNING: Numba not installed. Only Python benchmarks will run.")
        print("Install with: pip install numba")

    # Warmup
    warmup_jit()

    # Run benchmarks
    all_results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'numba_available': NUMBA_AVAILABLE,
            'sizes': args.sizes,
            'sweeps': args.sweeps,
        }
    }

    n_sweeps = args.sweeps // 10 if args.quick else args.sweeps
    n_clusters = 10 if args.quick else 100

    all_results['metropolis'] = benchmark_metropolis(args.sizes, n_sweeps)
    all_results['energy'] = benchmark_energy(args.sizes)
    all_results['wolff'] = benchmark_wolff(args.sizes, n_clusters)
    all_results['scaling'] = benchmark_scaling()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if NUMBA_AVAILABLE:
        metro_results = [r for r in all_results['metropolis'] if r['dimension'] == 2]
        if metro_results:
            avg_speedup = np.mean([r['speedup'] for r in metro_results])
            max_flips = max(r.get('numba_flips_per_sec', 0) for r in metro_results)
            print(f"Average Metropolis speedup (2D): {avg_speedup:.1f}x")
            print(f"Peak spin flips/sec (2D): {max_flips:.2e}")

        wolff_results = [r for r in all_results['wolff'] if r['dimension'] == 2]
        if wolff_results:
            max_clusters = max(r['clusters_per_sec'] for r in wolff_results)
            print(f"Peak Wolff clusters/sec (2D): {max_clusters:.1f}")
    else:
        print("Install Numba for significant speedups: pip install numba")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print()


if __name__ == '__main__':
    main()
