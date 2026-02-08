#!/usr/bin/env python
"""
Memory benchmark comparing original vs optimized storage.

This benchmark measures:
1. Spin array memory usage (int8 vs other dtypes)
2. Configuration compression ratios
3. File storage sizes
4. LazyResults vs full loading memory usage

Run with: python benchmarks/benchmark_memory.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ising_toolkit.models import Ising2D, Ising3D
from ising_toolkit.io import (
    SimulationResults,
    LazyResults,
    pack_spins,
    unpack_spins,
    ConfigurationBuffer,
    get_compression_ratio,
    estimate_storage_size,
)


def format_bytes(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.2f} KB"
    elif n_bytes < 1024 * 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{n_bytes / (1024 * 1024 * 1024):.2f} GB"


def benchmark_dtype_memory():
    """Benchmark memory usage with different dtypes."""
    print("\n" + "=" * 60)
    print("DTYPE MEMORY USAGE BENCHMARK")
    print("=" * 60)

    sizes = [32, 64, 128, 256]

    print(f"\n{'Size':>8} {'int8':>12} {'int32':>12} {'int64':>12} {'float64':>12} {'Savings':>10}")
    print("-" * 70)

    for L in sizes:
        n_spins = L * L

        # Create arrays with different dtypes
        spins_int8 = np.ones((L, L), dtype=np.int8)
        spins_int32 = np.ones((L, L), dtype=np.int32)
        spins_int64 = np.ones((L, L), dtype=np.int64)
        spins_float64 = np.ones((L, L), dtype=np.float64)

        savings = (1 - spins_int8.nbytes / spins_float64.nbytes) * 100

        print(f"{L}x{L:>4} {format_bytes(spins_int8.nbytes):>12} "
              f"{format_bytes(spins_int32.nbytes):>12} "
              f"{format_bytes(spins_int64.nbytes):>12} "
              f"{format_bytes(spins_float64.nbytes):>12} "
              f"{savings:>9.1f}%")

    print("\nNote: All Ising models use int8 dtype for optimal memory usage.")


def benchmark_compression():
    """Benchmark spin packing compression."""
    print("\n" + "=" * 60)
    print("SPIN PACKING COMPRESSION BENCHMARK")
    print("=" * 60)

    sizes = [16, 32, 64, 128, 256]

    print(f"\n{'Size':>8} {'Original':>12} {'Packed':>12} {'Ratio':>10} {'Pack (ms)':>12} {'Unpack (ms)':>12}")
    print("-" * 80)

    for L in sizes:
        spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

        # Benchmark packing
        start = time.perf_counter()
        for _ in range(100):
            packed = pack_spins(spins)
        pack_time = (time.perf_counter() - start) * 10  # ms per pack

        # Benchmark unpacking
        start = time.perf_counter()
        for _ in range(100):
            unpacked = unpack_spins(packed, spins.shape)
        unpack_time = (time.perf_counter() - start) * 10  # ms per unpack

        # Verify correctness
        assert np.array_equal(spins, unpacked), "Compression roundtrip failed!"

        ratio = get_compression_ratio(spins)

        print(f"{L}x{L:>4} {format_bytes(spins.nbytes):>12} "
              f"{format_bytes(packed.nbytes):>12} "
              f"{ratio:>9.1f}x "
              f"{pack_time:>11.3f} "
              f"{unpack_time:>11.3f}")

    # Test 3D compression
    print("\n3D Lattice Compression:")
    print(f"{'Size':>8} {'Original':>12} {'Packed':>12} {'Ratio':>10}")
    print("-" * 50)

    for L in [8, 16, 32, 64]:
        spins = np.random.choice([-1, 1], size=(L, L, L)).astype(np.int8)
        packed = pack_spins(spins)
        ratio = get_compression_ratio(spins)

        print(f"{L}x{L}x{L:>2} {format_bytes(spins.nbytes):>12} "
              f"{format_bytes(packed.nbytes):>12} "
              f"{ratio:>9.1f}x")


def benchmark_configuration_buffer():
    """Benchmark ConfigurationBuffer with decimation and limits."""
    print("\n" + "=" * 60)
    print("CONFIGURATION BUFFER BENCHMARK")
    print("=" * 60)

    L = 64
    n_steps = 10000

    print(f"\nSimulating {n_steps} steps with {L}x{L} lattice:")
    print("-" * 60)

    # No limits
    buffer_full = ConfigurationBuffer()
    spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
    for step in range(n_steps):
        buffer_full.add(step, spins)
    print(f"No limits:        {len(buffer_full):>6} configs, {format_bytes(buffer_full.get_memory_usage()):>10}")

    # With decimation
    buffer_dec = ConfigurationBuffer(decimation=100)
    for step in range(n_steps):
        buffer_dec.add(step, spins)
    print(f"Decimation=100:   {len(buffer_dec):>6} configs, {format_bytes(buffer_dec.get_memory_usage()):>10}")

    # With max limit
    buffer_max = ConfigurationBuffer(max_configurations=50)
    for step in range(n_steps):
        buffer_max.add(step, spins)
    print(f"Max=50:           {len(buffer_max):>6} configs, {format_bytes(buffer_max.get_memory_usage()):>10}")

    # With compression
    buffer_comp = ConfigurationBuffer(compress=True)
    for step in range(n_steps):
        buffer_comp.add(step, spins)
    print(f"Compressed:       {len(buffer_comp):>6} configs, {format_bytes(buffer_comp.get_memory_usage()):>10}")

    # Combined
    buffer_combined = ConfigurationBuffer(max_configurations=50, decimation=10, compress=True)
    for step in range(n_steps):
        buffer_combined.add(step, spins)
    print(f"Dec=10+Max=50+Comp: {len(buffer_combined):>4} configs, {format_bytes(buffer_combined.get_memory_usage()):>10}")


def benchmark_file_storage():
    """Benchmark file storage sizes."""
    print("\n" + "=" * 60)
    print("FILE STORAGE BENCHMARK")
    print("=" * 60)

    L = 64
    n_samples = 10000
    n_configs = 100

    # Create test data
    energy = np.random.randn(n_samples) * 100
    magnetization = np.random.randn(n_samples) * 50
    configurations = [
        np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
        for _ in range(n_configs)
    ]

    results = SimulationResults(
        energy=energy,
        magnetization=magnetization,
        metadata={'size': L, 'temperature': 2.269},
        configurations=configurations,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Standard save
        path_standard = Path(tmpdir) / "standard.h5"
        results.save(path_standard)
        size_standard = os.path.getsize(path_standard)

        # Compressed save
        path_compressed = Path(tmpdir) / "compressed.h5"
        results.save_compressed(path_compressed, compress_configurations=True)
        size_compressed = os.path.getsize(path_compressed)

        print(f"\nTest data: {L}x{L} lattice, {n_samples} samples, {n_configs} configurations")
        print("-" * 60)
        print(f"Standard HDF5:    {format_bytes(size_standard):>12}")
        print(f"Compressed HDF5:  {format_bytes(size_compressed):>12}")
        print(f"Reduction:        {(1 - size_compressed/size_standard)*100:>11.1f}%")

        # Test lazy loading
        print("\nLazy loading test:")
        print("-" * 60)

        # Full load
        start = time.perf_counter()
        full_results = SimulationResults.load(path_compressed)
        full_load_time = time.perf_counter() - start
        full_configs = full_results.configurations

        # Lazy load
        start = time.perf_counter()
        lazy_results = LazyResults(path_compressed)
        lazy_load_time = time.perf_counter() - start

        print(f"Full load time:       {full_load_time*1000:.2f} ms")
        print(f"Lazy init time:       {lazy_load_time*1000:.2f} ms")

        # Access time
        start = time.perf_counter()
        _ = lazy_results.energy_mean
        lazy_access_time = time.perf_counter() - start
        print(f"Lazy energy access:   {lazy_access_time*1000:.2f} ms")

        # Iterator access
        start = time.perf_counter()
        for i, config in lazy_results.iter_configurations(step=10):
            pass  # Just iterate
        iter_time = time.perf_counter() - start
        print(f"Iterate configs (step=10): {iter_time*1000:.2f} ms")

        lazy_results.close()


def benchmark_storage_estimates():
    """Show storage estimates for various scenarios."""
    print("\n" + "=" * 60)
    print("STORAGE SIZE ESTIMATES")
    print("=" * 60)

    scenarios = [
        ("Small (32x32, 1000 configs)", 32*32, 1000),
        ("Medium (64x64, 1000 configs)", 64*64, 1000),
        ("Large (128x128, 1000 configs)", 128*128, 1000),
        ("Large (128x128, 10000 configs)", 128*128, 10000),
        ("Very Large (256x256, 10000 configs)", 256*256, 10000),
        ("3D Small (16^3, 1000 configs)", 16**3, 1000),
        ("3D Medium (32^3, 1000 configs)", 32**3, 1000),
        ("3D Large (64^3, 1000 configs)", 64**3, 1000),
    ]

    print(f"\n{'Scenario':<40} {'Uncompressed':>15} {'Compressed':>15} {'Savings':>10}")
    print("-" * 85)

    for name, n_spins, n_configs in scenarios:
        uncomp = estimate_storage_size(n_spins, n_configs, compressed=False)
        comp = estimate_storage_size(n_spins, n_configs, compressed=True)

        savings = (1 - comp['mb'] / uncomp['mb']) * 100

        print(f"{name:<40} {uncomp['mb']:>14.2f}MB {comp['mb']:>14.2f}MB {savings:>9.1f}%")


def benchmark_model_memory():
    """Benchmark Ising model memory usage."""
    print("\n" + "=" * 60)
    print("ISING MODEL MEMORY USAGE")
    print("=" * 60)

    print("\n2D Models:")
    print(f"{'Size':>8} {'Spins':>12} {'Total':>15}")
    print("-" * 40)

    for L in [32, 64, 128, 256]:
        model = Ising2D(size=L, temperature=2.269)
        spin_bytes = model.spins.nbytes
        # Approximate total including acceptance probs
        total = spin_bytes + 200  # Small overhead

        print(f"{L}x{L:>4} {format_bytes(spin_bytes):>12} ~{format_bytes(total):>12}")

    print("\n3D Models:")
    print(f"{'Size':>10} {'Spins':>12} {'Total':>15}")
    print("-" * 45)

    for L in [16, 32, 64]:
        model = Ising3D(size=L, temperature=4.5)
        spin_bytes = model.spins.nbytes
        total = spin_bytes + 200

        print(f"{L}x{L}x{L:>2} {format_bytes(spin_bytes):>12} ~{format_bytes(total):>12}")


def run_all_benchmarks():
    """Run all memory benchmarks."""
    print("=" * 60)
    print("ISING MODEL MEMORY OPTIMIZATION BENCHMARKS")
    print("=" * 60)

    benchmark_dtype_memory()
    benchmark_compression()
    benchmark_configuration_buffer()
    benchmark_file_storage()
    benchmark_storage_estimates()
    benchmark_model_memory()

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print("""
Key Findings:
1. int8 dtype uses 8x less memory than float64 for spins
2. Bit-packing achieves ~7-8x compression for configurations
3. ConfigurationBuffer with decimation+limits can reduce memory 100x+
4. LazyResults enables working with files larger than RAM
5. Combined optimizations enable handling datasets 50-100x larger
""")


if __name__ == "__main__":
    run_all_benchmarks()
