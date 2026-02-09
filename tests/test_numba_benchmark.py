"""
Benchmark tests comparing Numba vs pure Python performance.

These tests verify that:
1. Numba and Python implementations produce consistent results
2. Numba provides significant speedup
3. Both implementations give correct physical results

Run with: pytest tests/test_numba_benchmark.py -v -s
"""

import time
import pytest
import numpy as np

from ising_toolkit.models import Ising2D
from ising_toolkit.samplers import MetropolisSampler
from ising_toolkit.utils.numba_kernels import NUMBA_AVAILABLE


class TestNumbaCorrectness:
    """Test that Numba and Python implementations give consistent results."""

    def test_energy_consistency(self):
        """Numba and Python energy calculations should match."""
        np.random.seed(42)

        for L in [8, 16, 32]:
            # Create random spin configuration
            spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

            # Model with Numba
            model_numba = Ising2D(size=L, temperature=2.269, use_numba=True)
            model_numba._spins = spins.copy()

            # Model without Numba
            model_python = Ising2D(size=L, temperature=2.269, use_numba=False)
            model_python._spins = spins.copy()

            energy_numba = model_numba.get_energy()
            energy_python = model_python.get_energy()

            assert np.isclose(energy_numba, energy_python, rtol=1e-10), \
                f"Energy mismatch for L={L}: numba={energy_numba}, python={energy_python}"

    def test_magnetization_consistency(self):
        """Numba and Python magnetization calculations should match."""
        np.random.seed(42)

        for L in [8, 16, 32]:
            spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

            model_numba = Ising2D(size=L, temperature=2.269, use_numba=True)
            model_numba._spins = spins.copy()

            model_python = Ising2D(size=L, temperature=2.269, use_numba=False)
            model_python._spins = spins.copy()

            mag_numba = model_numba.get_magnetization()
            mag_python = model_python.get_magnetization()

            assert mag_numba == mag_python, \
                f"Magnetization mismatch for L={L}: numba={mag_numba}, python={mag_python}"

    def test_sweep_statistical_consistency(self):
        """Metropolis sweeps should give similar acceptance rates."""
        np.random.seed(42)
        L = 16
        n_sweeps = 100

        # Run with Numba
        model_numba = Ising2D(size=L, temperature=2.269, use_numba=True)
        model_numba.initialize('random')
        model_numba.set_seed(42)

        numba_accepted = 0
        for _ in range(n_sweeps):
            numba_accepted += model_numba.metropolis_sweep()

        # Run with Python
        model_python = Ising2D(size=L, temperature=2.269, use_numba=False)
        model_python.initialize('random')
        model_python.set_seed(42)

        python_accepted = 0
        for _ in range(n_sweeps):
            python_accepted += model_python.metropolis_sweep()

        # Acceptance rates should be similar (not identical due to RNG differences)
        numba_rate = numba_accepted / (n_sweeps * L * L)
        python_rate = python_accepted / (n_sweeps * L * L)

        # They should be within 10% of each other
        assert abs(numba_rate - python_rate) < 0.1, \
            f"Acceptance rate mismatch: numba={numba_rate:.3f}, python={python_rate:.3f}"


class TestNumbaPerformance:
    """Benchmark tests comparing Numba vs Python performance."""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_energy_speedup(self):
        """Numba energy calculation should be faster than Python."""
        L = 64
        n_repeats = 1000

        model_numba = Ising2D(size=L, temperature=2.269, use_numba=True)
        model_numba.initialize('random')

        model_python = Ising2D(size=L, temperature=2.269, use_numba=False)
        model_python._spins = model_numba.spins.copy()

        # Warmup
        for _ in range(10):
            model_numba.get_energy()
            model_python.get_energy()

        # Benchmark Numba
        start = time.perf_counter()
        for _ in range(n_repeats):
            model_numba.get_energy()
        numba_time = time.perf_counter() - start

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(n_repeats):
            model_python.get_energy()
        python_time = time.perf_counter() - start

        speedup = python_time / numba_time

        print(f"\nEnergy calculation (L={L}, {n_repeats} repeats):")
        print(f"  Numba:  {numba_time*1000:.2f} ms")
        print(f"  Python: {python_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        # Numba should be at least 2x faster for energy
        assert speedup > 2, f"Expected speedup > 2x, got {speedup:.1f}x"

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_metropolis_sweep_speedup(self):
        """Numba Metropolis sweep should be much faster than Python."""
        L = 32
        n_sweeps = 100

        model_numba = Ising2D(size=L, temperature=2.269, use_numba=True)
        model_numba.initialize('random')

        model_python = Ising2D(size=L, temperature=2.269, use_numba=False)
        model_python._spins = model_numba.spins.copy()

        # Warmup
        for _ in range(5):
            model_numba.metropolis_sweep()

        # Benchmark Numba
        start = time.perf_counter()
        for _ in range(n_sweeps):
            model_numba.metropolis_sweep()
        numba_time = time.perf_counter() - start

        # Benchmark Python (fewer sweeps because it's slow)
        python_sweeps = min(n_sweeps, 10)
        start = time.perf_counter()
        for _ in range(python_sweeps):
            model_python.metropolis_sweep()
        python_time = (time.perf_counter() - start) * (n_sweeps / python_sweeps)

        speedup = python_time / numba_time

        print(f"\nMetropolis sweep (L={L}, {n_sweeps} sweeps):")
        print(f"  Numba:  {numba_time*1000:.2f} ms")
        print(f"  Python: {python_time*1000:.2f} ms (extrapolated)")
        print(f"  Speedup: {speedup:.1f}x")

        # Numba should be at least 50x faster for Metropolis
        assert speedup > 50, f"Expected speedup > 50x, got {speedup:.1f}x"

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_wolff_cluster_speedup(self):
        """Numba Wolff cluster should be faster than Python."""
        L = 32
        n_clusters = 100
        temperature = 2.269  # Near Tc for large clusters

        model_numba = Ising2D(size=L, temperature=temperature, use_numba=True)
        model_numba.initialize('random')

        model_python = Ising2D(size=L, temperature=temperature, use_numba=False)
        model_python._spins = model_numba.spins.copy()

        # Warmup
        for _ in range(5):
            model_numba.wolff_step()

        # Benchmark Numba
        start = time.perf_counter()
        numba_cluster_sizes = []
        for _ in range(n_clusters):
            numba_cluster_sizes.append(model_numba.wolff_step())
        numba_time = time.perf_counter() - start

        # Benchmark Python
        python_clusters = min(n_clusters, 20)
        start = time.perf_counter()
        python_cluster_sizes = []
        for _ in range(python_clusters):
            python_cluster_sizes.append(model_python.wolff_step())
        python_time = (time.perf_counter() - start) * (n_clusters / python_clusters)

        speedup = python_time / numba_time

        print(f"\nWolff cluster (L={L}, {n_clusters} clusters):")
        avg = np.mean(numba_cluster_sizes)
        print(f"  Numba:  {numba_time*1000:.2f} ms, avg cluster: {avg:.1f}")
        print(f"  Python: {python_time*1000:.2f} ms (extrapolated)")
        print(f"  Speedup: {speedup:.1f}x")

        # Wolff speedup depends on cluster size; Python uses efficient NumPy arrays
        # Speedup varies by platform and load; require at least 1x (no regression)
        assert speedup > 1, f"Expected speedup > 1x, got {speedup:.1f}x"

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_sampler_speedup(self):
        """MetropolisSampler with Numba should be faster than Python."""
        L = 32
        n_steps = 100

        # Numba sampler
        model_numba = Ising2D(size=L, temperature=2.269, use_numba=True)
        model_numba.initialize('random')
        sampler_numba = MetropolisSampler(model_numba, use_fast_sweep=True)

        # Python sampler
        model_python = Ising2D(size=L, temperature=2.269, use_numba=False)
        model_python._spins = model_numba.spins.copy()
        sampler_python = MetropolisSampler(model_python, use_fast_sweep=True)

        # Warmup
        for _ in range(5):
            sampler_numba.step()

        # Benchmark Numba
        start = time.perf_counter()
        for _ in range(n_steps):
            sampler_numba.step()
        numba_time = time.perf_counter() - start

        # Benchmark Python
        python_steps = min(n_steps, 10)
        start = time.perf_counter()
        for _ in range(python_steps):
            sampler_python.step()
        python_time = (time.perf_counter() - start) * (n_steps / python_steps)

        speedup = python_time / numba_time

        print(f"\nSampler step (L={L}, {n_steps} steps):")
        print(f"  Numba:  {numba_time*1000:.2f} ms")
        print(f"  Python: {python_time*1000:.2f} ms (extrapolated)")
        print(f"  Speedup: {speedup:.1f}x")

        assert speedup > 50, f"Expected speedup > 50x, got {speedup:.1f}x"


class TestScaling:
    """Test performance scaling with system size."""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_linear_scaling(self):
        """Sweep time should scale linearly with system size."""
        sizes = [16, 32, 64]
        n_sweeps = 50

        times = []
        n_sites = []

        for L in sizes:
            model = Ising2D(size=L, temperature=2.269, use_numba=True)
            model.initialize('random')

            # Warmup
            for _ in range(5):
                model.metropolis_sweep()

            # Benchmark
            start = time.perf_counter()
            for _ in range(n_sweeps):
                model.metropolis_sweep()
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            n_sites.append(L * L)

        # Check scaling
        # Time per site should be roughly constant
        time_per_site = [t / (n_sweeps * n) for t, n in zip(times, n_sites)]

        print("\nScaling analysis:")
        for L, t, tps in zip(sizes, times, time_per_site):
            print(f"  L={L}: {t*1000:.2f} ms, {tps*1e9:.2f} ns/site")

        # Time per site should vary by less than factor of 2
        ratio = max(time_per_site) / min(time_per_site)
        assert ratio < 2.0, f"Scaling not linear: time/site ratio = {ratio:.2f}"


class TestPhysicalCorrectness:
    """Test that simulations give correct physical results."""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_equilibration(self):
        """System should equilibrate to expected energy range."""
        L = 16
        T = 2.0  # Below Tc, should be ordered
        n_equilibration = 1000
        n_measurement = 500

        model = Ising2D(size=L, temperature=T, use_numba=True)
        model.initialize('random')

        # Equilibrate
        for _ in range(n_equilibration):
            model.metropolis_sweep()

        # Measure
        energies = []
        mags = []
        for _ in range(n_measurement):
            model.metropolis_sweep()
            energies.append(model.get_energy() / model.n_spins)
            mags.append(abs(model.get_magnetization()) / model.n_spins)

        mean_energy = np.mean(energies)
        mean_mag = np.mean(mags)

        print(f"\nEquilibration test (L={L}, T={T}):")
        print(f"  Mean energy/spin: {mean_energy:.4f}")
        print(f"  Mean |M|/spin: {mean_mag:.4f}")

        # Below Tc, should be ordered: E ~ -2, |M| ~ 1
        assert mean_energy < -1.5, f"Energy too high: {mean_energy:.4f}"
        assert mean_mag > 0.8, f"Magnetization too low: {mean_mag:.4f}"

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_high_temperature_disorder(self):
        """High temperature should give disordered state."""
        L = 16
        T = 10.0  # Well above Tc
        n_equilibration = 500
        n_measurement = 200

        model = Ising2D(size=L, temperature=T, use_numba=True)
        model.initialize('up')  # Start ordered

        # Equilibrate
        for _ in range(n_equilibration):
            model.metropolis_sweep()

        # Measure
        mags = []
        for _ in range(n_measurement):
            model.metropolis_sweep()
            mags.append(abs(model.get_magnetization()) / model.n_spins)

        mean_mag = np.mean(mags)

        print(f"\nHigh-T test (L={L}, T={T}):")
        print(f"  Mean |M|/spin: {mean_mag:.4f}")

        # At high T, magnetization should be small
        assert mean_mag < 0.3, f"Magnetization too high at high T: {mean_mag:.4f}"


def run_full_benchmark():
    """Run comprehensive benchmark and print summary."""
    print("=" * 60)
    print("NUMBA ACCELERATION BENCHMARK")
    print("=" * 60)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print()

    if not NUMBA_AVAILABLE:
        print("Install Numba for 100x+ speedups: pip install numba")
        return

    # Test energy calculation
    print("Testing energy calculation...")
    test = TestNumbaPerformance()
    test.test_energy_speedup()

    # Test Metropolis sweep
    print("\nTesting Metropolis sweep...")
    test.test_metropolis_sweep_speedup()

    # Test Wolff cluster
    print("\nTesting Wolff cluster...")
    test.test_wolff_cluster_speedup()

    # Test sampler
    print("\nTesting MetropolisSampler...")
    test.test_sampler_speedup()

    # Test scaling
    print("\nTesting scaling...")
    scaling = TestScaling()
    scaling.test_linear_scaling()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    run_full_benchmark()
