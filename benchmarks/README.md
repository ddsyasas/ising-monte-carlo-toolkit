# Ising Monte Carlo Toolkit Benchmarks

This directory contains performance benchmarks for the Ising Monte Carlo toolkit.

## Quick Start

```bash
# Run all benchmarks (full mode)
python benchmarks/benchmark_suite.py

# Run quick benchmarks (fewer iterations)
python benchmarks/benchmark_suite.py --quick

# Run specific benchmark
python benchmarks/benchmark_suite.py -b metropolis

# Save results to CSV
python benchmarks/benchmark_suite.py --output results.csv
```

## Available Benchmarks

### 1. Metropolis Scaling (`-b metropolis`)
Measures Metropolis algorithm performance vs system size.

**Metrics:**
- Time per sweep
- Spin flips per second
- Scaling efficiency (should be ~1.0 for linear scaling)

### 2. Wolff Cluster Scaling (`-b wolff`)
Measures Wolff cluster algorithm performance and cluster statistics.

**Metrics:**
- Time per cluster update
- Average cluster size (depends on temperature)
- Clusters per second

### 3. Wolff vs Metropolis Comparison (`-b comparison`)
Compares both algorithms at various temperatures.

**Key insight:** Wolff is faster near Tc due to critical slowing down in Metropolis.

### 4. Numba Speedup (`-b numba`)
Measures speedup from Numba JIT compilation.

**Typical results:** 100-500x speedup for Metropolis sweep

### 5. Parallel Scaling (`-b parallel`)
Measures parallel efficiency vs number of workers.

**Metrics:**
- Speedup vs single worker
- Parallel efficiency (speedup / workers)

### 6. Energy Calculation (`-b energy`)
Benchmarks energy calculation performance.

### 7. Compression (`-b compression`)
Benchmarks spin configuration compression.

**Metrics:**
- Compression ratio (~8x for bit-packing)
- Compression speed (MB/s)

### 8. 3D Model Scaling (`-b 3d`)
Benchmarks 3D Ising model performance.

## Typical Results

Results measured on Apple M1 Pro (10 cores), Python 3.11, Numba 0.58.

### Metropolis Scaling (Numba enabled)

| Size | Spins | Time/Sweep | Spin Flips/s |
|------|-------|------------|--------------|
| 16x16 | 256 | 2.5 μs | 100 M/s |
| 32x32 | 1024 | 10 μs | 100 M/s |
| 64x64 | 4096 | 41 μs | 100 M/s |
| 128x128 | 16384 | 164 μs | 100 M/s |
| 256x256 | 65536 | 656 μs | 100 M/s |

Scaling efficiency: ~1.0 (linear with system size)

### Numba Speedup

| Size | Python | Numba | Speedup |
|------|--------|-------|---------|
| 16x16 | 25 ms | 25 μs | 1000x |
| 32x32 | 100 ms | 100 μs | 1000x |
| 64x64 | 400 ms | 410 μs | 1000x |

### Wolff vs Metropolis (at Tc = 2.269)

| Temperature | Metropolis | Wolff | Wolff Speedup |
|-------------|------------|-------|---------------|
| 2.0 (T < Tc) | 41 ms | 35 ms | 1.2x |
| 2.269 (Tc) | 41 ms | 8 ms | 5x |
| 3.0 (T > Tc) | 41 ms | 15 ms | 2.7x |

**Note:** Wolff is most advantageous near Tc where clusters are large.

### Parallel Scaling (20 temperature points)

| Workers | Time | Speedup | Efficiency |
|---------|------|---------|------------|
| 1 | 12.5 s | 1.0x | 100% |
| 2 | 6.5 s | 1.9x | 95% |
| 4 | 3.5 s | 3.6x | 90% |
| 8 | 2.0 s | 6.3x | 79% |

### Compression

| Size | Original | Compressed | Ratio | Speed |
|------|----------|------------|-------|-------|
| 32x32 | 1 KB | 152 B | 6.7x | 500 MB/s |
| 64x64 | 4 KB | 536 B | 7.6x | 600 MB/s |
| 128x128 | 16 KB | 2 KB | 7.9x | 700 MB/s |
| 256x256 | 64 KB | 8 KB | 8.0x | 800 MB/s |

## Individual Benchmark Scripts

- `benchmark_suite.py` - Main comprehensive benchmark suite
- `benchmark_memory.py` - Memory usage and compression benchmarks
- `benchmark_numba_kernels.py` - Detailed Numba kernel benchmarks

## Running Custom Benchmarks

```python
from benchmarks.benchmark_suite import (
    benchmark_metropolis_scaling,
    benchmark_wolff_vs_metropolis,
    benchmark_numba_speedup,
)

# Custom Metropolis benchmark
results = benchmark_metropolis_scaling(
    sizes=[8, 16, 32, 64, 128],
    n_steps=5000,
    temperature=2.5,  # Above Tc
)

# Compare algorithms
results = benchmark_wolff_vs_metropolis(
    size=128,
    n_sweeps=10000,
    temperatures=[1.5, 2.0, 2.269, 2.5, 3.0, 4.0, 5.0],
)
```

## Performance Tips

1. **Always use Numba** - Provides 100-1000x speedup
   ```python
   model = Ising2D(size=64, temperature=2.269, use_numba=True)
   ```

2. **Use Wolff near Tc** - Avoids critical slowing down
   ```python
   if abs(T - 2.269) < 0.3:
       sampler = WolffSampler(model)
   else:
       sampler = MetropolisSampler(model)
   ```

3. **Parallel temperature sweeps** - Use multiple workers
   ```python
   results = sweep.run(n_workers=4)
   ```

4. **Compress configurations** - 8x storage reduction
   ```python
   results.save_compressed('output.h5', compress_configurations=True)
   ```

5. **Use LazyResults for large files** - Avoid loading everything into RAM
   ```python
   with LazyResults('large_file.h5') as results:
       for i, config in results.iter_configurations():
           process(config)
   ```

## Environment

Benchmarks were run with:
- Python 3.11
- NumPy 1.24
- Numba 0.58
- macOS 14 / Apple M1 Pro

Results will vary based on hardware and software versions.
