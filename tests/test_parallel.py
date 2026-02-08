"""
Tests for parallel execution utilities.

Tests cover:
1. parallel_map correctness with different worker counts
2. Error handling in parallel execution
3. TemperatureSweep parallelization
4. Finite-size scaling parallelization
5. Performance scaling with workers

Run with: pytest tests/test_parallel.py -v
"""

import os
import time
import pytest
import numpy as np

from ising_toolkit.utils.parallel import (
    parallel_map,
    parallel_map_with_errors,
    ParallelResult,
    run_temperature_point,
    run_temperature_sweep_parallel,
)


# Simple test functions for parallel_map
def square(x):
    """Simple function for testing."""
    return x * x


def slow_square(x):
    """Slow function for testing parallelism."""
    time.sleep(0.05)
    return x * x


def failing_function(x):
    """Function that fails for certain inputs."""
    if x == 5:
        raise ValueError(f"Cannot process {x}")
    return x * x


def always_failing_function(x):
    """Function that always fails."""
    raise RuntimeError("Always fails")


class TestParallelMap:
    """Test parallel_map with different configurations."""

    def test_sequential_execution(self):
        """parallel_map with n_workers=1 should work correctly."""
        items = list(range(10))
        results = parallel_map(square, items, n_workers=1, progress=False)
        expected = [x * x for x in items]
        assert results == expected

    def test_parallel_execution_2_workers(self):
        """parallel_map with 2 workers should preserve order."""
        items = list(range(20))
        results = parallel_map(square, items, n_workers=2, progress=False)
        expected = [x * x for x in items]
        assert results == expected

    def test_parallel_execution_4_workers(self):
        """parallel_map with 4 workers should preserve order."""
        items = list(range(20))
        results = parallel_map(square, items, n_workers=4, progress=False)
        expected = [x * x for x in items]
        assert results == expected

    def test_parallel_execution_many_workers(self):
        """parallel_map with more workers than items should work."""
        items = list(range(5))
        results = parallel_map(square, items, n_workers=10, progress=False)
        expected = [x * x for x in items]
        assert results == expected

    def test_parallel_auto_workers(self):
        """parallel_map with n_workers=-1 should use all CPUs."""
        items = list(range(10))
        results = parallel_map(square, items, n_workers=-1, progress=False)
        expected = [x * x for x in items]
        assert results == expected

    def test_empty_list(self):
        """parallel_map with empty list should return empty list."""
        results = parallel_map(square, [], n_workers=2, progress=False)
        assert results == []

    def test_single_item(self):
        """parallel_map with single item should work."""
        results = parallel_map(square, [5], n_workers=2, progress=False)
        assert results == [25]

    def test_result_order_preserved(self):
        """Results should be in the same order as inputs."""
        # Use a slow function so execution order might differ from input order
        items = list(range(10))
        np.random.shuffle(items)  # Randomize input order
        results = parallel_map(slow_square, items, n_workers=4, progress=False)
        expected = [x * x for x in items]
        assert results == expected

    @pytest.mark.skipif(True, reason="Speedup test skipped - process overhead varies by system")
    def test_speedup_with_workers(self):
        """Parallel execution should be faster than sequential for slow tasks."""
        items = list(range(20))

        # Sequential
        start = time.perf_counter()
        parallel_map(slow_square, items, n_workers=1, progress=False)
        sequential_time = time.perf_counter() - start

        # Parallel (4 workers)
        start = time.perf_counter()
        parallel_map(slow_square, items, n_workers=4, progress=False)
        parallel_time = time.perf_counter() - start

        # Should see significant speedup (at least 2x)
        speedup = sequential_time / parallel_time
        print(f"\nSpeedup with 4 workers: {speedup:.2f}x")
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"


class TestParallelMapWithErrors:
    """Test parallel_map_with_errors for error handling."""

    def test_no_errors(self):
        """When all succeed, results should match regular parallel_map."""
        items = list(range(10))
        result = parallel_map_with_errors(square, items, n_workers=2, progress=False)

        # Returns a ParallelResult object
        assert isinstance(result, ParallelResult)
        assert len(result.results) == 10
        assert len(result.errors) == 0
        assert result.n_successful == 10
        assert result.n_failed == 0

        for i, r in enumerate(result.results):
            assert r == i * i

    def test_partial_failure(self):
        """Should capture errors without failing the entire run."""
        items = list(range(10))  # 5 will fail
        result = parallel_map_with_errors(
            failing_function, items, n_workers=2, progress=False
        )

        assert isinstance(result, ParallelResult)
        assert len(result.results) == 10
        assert len(result.errors) == 1  # Only item 5 should fail

        # Check the error
        assert result.errors[0][0] == 5  # Index 5 failed
        assert "Cannot process 5" in str(result.errors[0][1])

        # Check successful results
        for i, r in enumerate(result.results):
            if i == 5:
                assert r is None
            else:
                assert r == i * i

    def test_all_failures(self):
        """Should handle case where all items fail."""
        items = list(range(5))
        result = parallel_map_with_errors(
            always_failing_function, items, n_workers=2, progress=False
        )

        assert isinstance(result, ParallelResult)
        assert len(result.results) == 5
        assert len(result.errors) == 5
        assert result.n_failed == 5

        for idx, err in result.errors:
            assert "Always fails" in str(err)


class TestTemperatureSweepParallel:
    """Test TemperatureSweep with parallel execution."""

    def test_sweep_sequential(self):
        """TemperatureSweep with n_workers=1 should work."""
        from ising_toolkit.models import Ising2D
        from ising_toolkit.analysis import TemperatureSweep

        temps = [2.0, 2.5, 3.0]
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=100,
            seed=42,
        )

        results = sweep.run(n_workers=1, progress=False)

        # Should have results for all temperatures
        assert len(sweep._results_list) == 3
        result_temps = [r['temperature'] for r in sweep._results_list]
        assert result_temps == sorted(temps)

    def test_sweep_parallel_2_workers(self):
        """TemperatureSweep with 2 workers should work."""
        from ising_toolkit.models import Ising2D
        from ising_toolkit.analysis import TemperatureSweep

        temps = [1.5, 2.0, 2.5, 3.0, 3.5]
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=100,
            seed=42,
        )

        results = sweep.run(n_workers=2, progress=False)

        # Should have results for all temperatures
        assert len(sweep._results_list) == 5
        result_temps = [r['temperature'] for r in sweep._results_list]
        assert result_temps == sorted(temps)

    def test_sweep_parallel_4_workers(self):
        """TemperatureSweep with 4 workers should work."""
        from ising_toolkit.models import Ising2D
        from ising_toolkit.analysis import TemperatureSweep

        temps = np.linspace(1.5, 3.5, 10)
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=100,
            seed=42,
        )

        results = sweep.run(n_workers=4, progress=False)

        # Should have results for all temperatures
        assert len(sweep._results_list) == 10

        # Results should be sorted by temperature
        result_temps = [r['temperature'] for r in sweep._results_list]
        assert result_temps == sorted(result_temps)

    def test_sweep_consistency(self):
        """Sequential and parallel sweeps should give consistent results."""
        from ising_toolkit.models import Ising2D
        from ising_toolkit.analysis import TemperatureSweep

        temps = [2.0, 2.5, 3.0]

        # Run sequentially
        sweep_seq = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=500,
            seed=42,
        )
        sweep_seq.run(n_workers=1, progress=False)

        # Run in parallel
        sweep_par = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=500,
            seed=42,
        )
        sweep_par.run(n_workers=2, progress=False)

        # Results should be similar (not identical due to RNG differences)
        for r_seq, r_par in zip(sweep_seq._results_list, sweep_par._results_list):
            assert r_seq['temperature'] == r_par['temperature']
            # Physical quantities should be in similar ranges
            assert abs(r_seq['energy_mean'] - r_par['energy_mean']) < 0.5
            assert abs(r_seq['abs_magnetization_mean'] - r_par['abs_magnetization_mean']) < 0.3


class TestFiniteSizeScalingParallel:
    """Test finite-size scaling with parallel execution."""

    def test_fss_sequential(self):
        """run_finite_size_scaling with n_workers=1 should work."""
        from ising_toolkit.models import Ising2D
        from ising_toolkit.analysis import run_finite_size_scaling

        sizes = [4, 8]
        temps = [2.0, 2.5, 3.0]

        results = run_finite_size_scaling(
            model_class=Ising2D,
            sizes=sizes,
            temperatures=temps,
            n_steps=50,
            n_workers=1,
            seed=42,
            progress=False,
        )

        assert len(results) == 2
        assert 4 in results
        assert 8 in results
        # Each size should have results for all temperatures
        assert len(results[4]) == 3
        assert len(results[8]) == 3

    def test_fss_parallel_temperatures(self):
        """run_finite_size_scaling parallelizing temperatures."""
        from ising_toolkit.models import Ising2D
        from ising_toolkit.analysis import run_finite_size_scaling

        sizes = [4, 8]
        temps = [2.0, 2.5, 3.0]

        results = run_finite_size_scaling(
            model_class=Ising2D,
            sizes=sizes,
            temperatures=temps,
            n_steps=50,
            n_workers=2,
            parallel_sizes=False,
            seed=42,
            progress=False,
        )

        assert len(results) == 2
        assert 4 in results
        assert 8 in results

    def test_fss_parallel_sizes(self):
        """run_finite_size_scaling parallelizing across sizes."""
        from ising_toolkit.models import Ising2D
        from ising_toolkit.analysis import run_finite_size_scaling

        sizes = [4, 8, 12]
        temps = [2.0, 2.5]

        results = run_finite_size_scaling(
            model_class=Ising2D,
            sizes=sizes,
            temperatures=temps,
            n_steps=50,
            n_workers=3,
            parallel_sizes=True,
            seed=42,
            progress=False,
        )

        assert len(results) == 3
        assert all(L in results for L in sizes)
        for L in sizes:
            assert len(results[L]) == 2


class TestRunTemperaturePoint:
    """Test the run_temperature_point worker function."""

    def test_basic_functionality(self):
        """run_temperature_point should return expected observables."""
        # Format: (model_class_name, model_kwargs, sampler_class_name, sampler_kwargs,
        #          temperature, n_steps, equilibration, measurement_interval, seed)
        args = (
            'Ising2D',           # model_class_name
            {'size': 8},         # model_kwargs
            'MetropolisSampler', # sampler_class_name
            {},                  # sampler_kwargs
            2.269,               # temperature
            100,                 # n_steps
            50,                  # equilibration
            1,                   # measurement_interval
            42,                  # seed
        )
        result = run_temperature_point(args)

        assert 'temperature' in result
        assert result['temperature'] == 2.269
        assert result['error'] is None
        assert 'energy_mean' in result
        assert 'magnetization_mean' in result
        assert 'abs_magnetization_mean' in result
        assert 'specific_heat' in result
        assert 'susceptibility' in result
        assert 'binder' in result

    def test_wolff_algorithm(self):
        """run_temperature_point should work with Wolff algorithm."""
        args = (
            'Ising2D',
            {'size': 8},
            'WolffSampler',
            {},
            2.269,
            100,
            50,
            1,
            42,
        )
        result = run_temperature_point(args)

        assert result['temperature'] == 2.269
        assert result['error'] is None
        assert 'energy_mean' in result


class TestParallelSweepUtility:
    """Test the run_temperature_sweep_parallel utility function."""

    def test_basic_functionality(self):
        """run_temperature_sweep_parallel should work."""
        results = run_temperature_sweep_parallel(
            model_class='Ising2D',
            size=8,
            temperatures=[2.0, 2.5, 3.0],
            n_steps=50,
            equilibration=25,
            algorithm='metropolis',
            seed=42,
            n_workers=2,
            progress=False,
        )

        assert len(results) == 3
        temps = [r['temperature'] for r in results]
        assert temps == [2.0, 2.5, 3.0]  # Should be sorted


class TestWorkerCountScaling:
    """Test performance scaling with different worker counts."""

    def test_scaling_analysis(self):
        """Analyze speedup with different worker counts.

        Note: For small simulations, multiprocessing overhead may dominate.
        This test primarily verifies that parallel execution works correctly
        with different worker counts, not that it's always faster.
        """
        from ising_toolkit.models import Ising2D
        from ising_toolkit.analysis import TemperatureSweep

        # Use more temperatures and steps to amortize overhead
        temps = np.linspace(2.0, 2.5, 8)
        n_cpus = os.cpu_count() or 4

        times = {}
        results = {}

        for n_workers in [1, 2, min(4, n_cpus)]:
            sweep = TemperatureSweep(
                model_class=Ising2D,
                size=16,
                temperatures=temps,
                n_steps=500,  # More steps
                seed=42,
            )

            start = time.perf_counter()
            sweep.run(n_workers=n_workers, progress=False)
            elapsed = time.perf_counter() - start

            times[n_workers] = elapsed
            results[n_workers] = sweep._results_list

        print("\nWorker scaling analysis:")
        for n, t in sorted(times.items()):
            speedup = times[1] / t if n > 1 else 1.0
            print(f"  {n} workers: {t:.2f}s (speedup: {speedup:.2f}x)")

        # Verify all worker counts produce valid results
        for n_workers, result_list in results.items():
            assert len(result_list) == len(temps), f"Worker {n_workers} missing results"
            for r in result_list:
                assert r['temperature'] is not None
                assert r['energy_mean'] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
