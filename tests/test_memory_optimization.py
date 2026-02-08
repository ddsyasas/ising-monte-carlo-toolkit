"""
Tests for memory optimization features.

Tests cover:
1. Spin packing/unpacking correctness
2. ConfigurationBuffer functionality
3. LazyResults lazy loading
4. File compression
5. Configuration limiting

Run with: pytest tests/test_memory_optimization.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ising_toolkit.io import (
    SimulationResults,
    LazyResults,
    pack_spins,
    unpack_spins,
    pack_configurations,
    unpack_configurations,
    ConfigurationBuffer,
    get_compression_ratio,
    estimate_storage_size,
)
from ising_toolkit.models import Ising2D


class TestSpinPacking:
    """Tests for spin packing/unpacking."""

    def test_pack_unpack_1d(self):
        """Pack and unpack 1D array should preserve data."""
        spins = np.array([1, -1, 1, 1, -1, -1, 1, -1], dtype=np.int8)
        packed = pack_spins(spins)
        unpacked = unpack_spins(packed)

        assert np.array_equal(spins, unpacked)

    def test_pack_unpack_2d(self):
        """Pack and unpack 2D array should preserve data."""
        for L in [8, 16, 32, 64]:
            spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
            packed = pack_spins(spins)
            unpacked = unpack_spins(packed)

            assert unpacked.shape == spins.shape
            assert np.array_equal(spins, unpacked), f"Failed for L={L}"

    def test_pack_unpack_3d(self):
        """Pack and unpack 3D array should preserve data."""
        for L in [4, 8, 16]:
            spins = np.random.choice([-1, 1], size=(L, L, L)).astype(np.int8)
            packed = pack_spins(spins)
            unpacked = unpack_spins(packed)

            assert unpacked.shape == spins.shape
            assert np.array_equal(spins, unpacked), f"Failed for L={L}"

    def test_pack_unpack_non_multiple_8(self):
        """Pack and unpack array with size not multiple of 8."""
        for size in [7, 13, 25, 100]:
            spins = np.random.choice([-1, 1], size=size).astype(np.int8)
            packed = pack_spins(spins)
            unpacked = unpack_spins(packed)

            assert np.array_equal(spins, unpacked), f"Failed for size={size}"

    def test_compression_ratio(self):
        """Compression ratio should be approximately 8x."""
        spins = np.random.choice([-1, 1], size=(64, 64)).astype(np.int8)
        ratio = get_compression_ratio(spins)

        # Should be close to 8x (minus header overhead)
        assert 6.0 < ratio < 8.5, f"Unexpected compression ratio: {ratio}"

    def test_pack_all_ones(self):
        """Pack array of all +1 spins."""
        spins = np.ones((16, 16), dtype=np.int8)
        packed = pack_spins(spins)
        unpacked = unpack_spins(packed)

        assert np.array_equal(spins, unpacked)

    def test_pack_all_negative_ones(self):
        """Pack array of all -1 spins."""
        spins = -np.ones((16, 16), dtype=np.int8)
        packed = pack_spins(spins)
        unpacked = unpack_spins(packed)

        assert np.array_equal(spins, unpacked)

    def test_pack_checkerboard(self):
        """Pack checkerboard pattern."""
        L = 16
        spins = np.ones((L, L), dtype=np.int8)
        spins[1::2, ::2] = -1
        spins[::2, 1::2] = -1

        packed = pack_spins(spins)
        unpacked = unpack_spins(packed)

        assert np.array_equal(spins, unpacked)


class TestPackConfigurations:
    """Tests for packing multiple configurations."""

    def test_pack_unpack_configurations(self):
        """Pack and unpack list of configurations."""
        configs = [
            np.random.choice([-1, 1], size=(8, 8)).astype(np.int8)
            for _ in range(10)
        ]

        packed = pack_configurations(configs)
        unpacked = unpack_configurations(packed)

        assert len(unpacked) == len(configs)
        for orig, unp in zip(configs, unpacked):
            assert np.array_equal(orig, unp)

    def test_pack_empty_list(self):
        """Pack empty list should return empty array."""
        packed = pack_configurations([])
        assert len(packed) == 0

        unpacked = unpack_configurations(packed)
        assert len(unpacked) == 0


class TestConfigurationBuffer:
    """Tests for ConfigurationBuffer."""

    def test_basic_add(self):
        """Basic add functionality."""
        buffer = ConfigurationBuffer()
        spins = np.ones((8, 8), dtype=np.int8)

        for i in range(10):
            buffer.add(i, spins)

        assert len(buffer) == 10

    def test_max_configurations(self):
        """Max configurations limit should work."""
        buffer = ConfigurationBuffer(max_configurations=5)
        spins = np.ones((8, 8), dtype=np.int8)

        for i in range(20):
            buffer.add(i, spins)

        assert len(buffer) == 5
        # Should have kept most recent
        steps = buffer.get_steps()
        assert steps == [15, 16, 17, 18, 19]

    def test_decimation(self):
        """Decimation should only keep every Nth configuration."""
        buffer = ConfigurationBuffer(decimation=10)
        spins = np.ones((8, 8), dtype=np.int8)

        for i in range(100):
            buffer.add(i, spins)

        assert len(buffer) == 10
        steps = buffer.get_steps()
        assert steps == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    def test_compression(self):
        """Compressed buffer should use less memory."""
        buffer_normal = ConfigurationBuffer(compress=False)
        buffer_compressed = ConfigurationBuffer(compress=True)

        spins = np.random.choice([-1, 1], size=(32, 32)).astype(np.int8)

        for i in range(100):
            buffer_normal.add(i, spins)
            buffer_compressed.add(i, spins)

        # Compressed should use less memory
        assert buffer_compressed.get_memory_usage() < buffer_normal.get_memory_usage()

        # But should give same results
        configs_normal = buffer_normal.get_configurations()
        configs_compressed = buffer_compressed.get_configurations()

        for c1, c2 in zip(configs_normal, configs_compressed):
            assert np.array_equal(c1, c2)

    def test_clear(self):
        """Clear should remove all configurations."""
        buffer = ConfigurationBuffer()
        spins = np.ones((8, 8), dtype=np.int8)

        for i in range(10):
            buffer.add(i, spins)

        buffer.clear()
        assert len(buffer) == 0

    def test_combined_limits(self):
        """Decimation and max together should work correctly."""
        buffer = ConfigurationBuffer(max_configurations=5, decimation=10)
        spins = np.ones((8, 8), dtype=np.int8)

        for i in range(1000):
            buffer.add(i, spins)

        assert len(buffer) == 5


class TestLazyResults:
    """Tests for LazyResults lazy loading."""

    @pytest.fixture
    def results_file(self, tmp_path):
        """Create a test results file."""
        L = 16
        n_samples = 1000
        n_configs = 20

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

        filepath = tmp_path / "test_results.h5"
        results.save_compressed(filepath, compress_configurations=True)

        return filepath, energy, magnetization, configurations

    def test_lazy_init(self, results_file):
        """LazyResults should load metadata without full data."""
        filepath, _, _, _ = results_file

        with LazyResults(filepath) as lazy:
            # Metadata should be available
            assert lazy.n_samples == 1000
            assert lazy.n_configurations == 20
            assert lazy.metadata['size'] == 16

            # Data should not be loaded yet
            assert lazy._energy is None
            assert lazy._magnetization is None

    def test_lazy_energy_access(self, results_file):
        """Energy should load on first access."""
        filepath, energy, _, _ = results_file

        with LazyResults(filepath) as lazy:
            # Access energy
            loaded_energy = lazy.energy

            # Should match original
            assert np.allclose(energy, loaded_energy)

            # Should be cached now
            assert lazy._energy is not None

    def test_lazy_magnetization_access(self, results_file):
        """Magnetization should load on first access."""
        filepath, _, magnetization, _ = results_file

        with LazyResults(filepath) as lazy:
            loaded_mag = lazy.magnetization
            assert np.allclose(magnetization, loaded_mag)

    def test_lazy_configuration_access(self, results_file):
        """Individual configurations should load on demand."""
        filepath, _, _, configurations = results_file

        with LazyResults(filepath) as lazy:
            for i in range(5):
                config = lazy.get_configuration(i)
                assert np.array_equal(config, configurations[i])

    def test_lazy_iter_configurations(self, results_file):
        """iter_configurations should yield configs one at a time."""
        filepath, _, _, configurations = results_file

        with LazyResults(filepath) as lazy:
            count = 0
            for i, config in lazy.iter_configurations():
                assert np.array_equal(config, configurations[i])
                count += 1

            assert count == len(configurations)

    def test_lazy_iter_configurations_step(self, results_file):
        """iter_configurations with step should skip correctly."""
        filepath, _, _, configurations = results_file

        with LazyResults(filepath) as lazy:
            indices = []
            for i, config in lazy.iter_configurations(step=5):
                indices.append(i)

            assert indices == [0, 5, 10, 15]

    def test_lazy_statistics(self, results_file):
        """Statistics should compute correctly."""
        filepath, energy, magnetization, _ = results_file

        with LazyResults(filepath) as lazy:
            assert np.isclose(lazy.energy_mean, np.mean(energy))
            assert np.isclose(lazy.magnetization_mean, np.mean(magnetization))
            assert np.isclose(lazy.abs_magnetization_mean, np.mean(np.abs(magnetization)))

    def test_lazy_to_simulation_results(self, results_file):
        """Convert lazy to full results."""
        filepath, energy, magnetization, configurations = results_file

        with LazyResults(filepath) as lazy:
            full = lazy.to_simulation_results()

            assert np.allclose(full.energy, energy)
            assert np.allclose(full.magnetization, magnetization)
            assert len(full.configurations) == len(configurations)

    def test_lazy_memory_estimate(self, results_file):
        """Memory estimate should be reasonable."""
        filepath, _, _, _ = results_file

        with LazyResults(filepath) as lazy:
            mem = lazy.get_memory_estimate()

            assert mem['timeseries_mb'] > 0
            assert mem['configurations_mb'] > 0
            assert mem['total_mb'] == mem['timeseries_mb'] + mem['configurations_mb']


class TestSimulationResultsCompression:
    """Tests for SimulationResults compression features."""

    def test_save_compressed(self, tmp_path):
        """save_compressed should create smaller files."""
        L = 32
        n_samples = 1000
        n_configs = 50

        energy = np.random.randn(n_samples) * 100
        magnetization = np.random.randn(n_samples) * 50
        configurations = [
            np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
            for _ in range(n_configs)
        ]

        results = SimulationResults(
            energy=energy,
            magnetization=magnetization,
            metadata={'size': L},
            configurations=configurations,
        )

        # Save both ways
        path_normal = tmp_path / "normal.h5"
        path_compressed = tmp_path / "compressed.h5"

        results.save(path_normal)
        results.save_compressed(path_compressed, compress_configurations=True)

        # Compressed should be smaller
        size_normal = path_normal.stat().st_size
        size_compressed = path_compressed.stat().st_size

        assert size_compressed < size_normal

    def test_add_configuration_decimation(self):
        """add_configuration with decimation should skip correctly."""
        results = SimulationResults(
            energy=np.array([1.0]),
            magnetization=np.array([1.0]),
        )

        spins = np.ones((8, 8), dtype=np.int8)

        for i in range(100):
            results.add_configuration(spins, decimation=10)

        assert len(results.configurations) == 10

    def test_add_configuration_max_limit(self):
        """add_configuration with max_configurations should limit size."""
        results = SimulationResults(
            energy=np.array([1.0]),
            magnetization=np.array([1.0]),
        )

        spins = np.ones((8, 8), dtype=np.int8)

        for i in range(100):
            results.add_configuration(spins, max_configurations=5)

        assert len(results.configurations) == 5


class TestStorageEstimates:
    """Tests for storage size estimation."""

    def test_estimate_storage_size_uncompressed(self):
        """Uncompressed storage estimate."""
        est = estimate_storage_size(64*64, 1000, compressed=False)

        assert est['bytes'] == 64 * 64 * 1000
        assert est['compression_ratio'] == 1.0

    def test_estimate_storage_size_compressed(self):
        """Compressed storage estimate."""
        est = estimate_storage_size(64*64, 1000, compressed=True)

        # Should be roughly 8x smaller plus header
        assert est['compression_ratio'] > 6.0


class TestModelMemory:
    """Test that models use optimal memory."""

    def test_ising2d_uses_int8(self):
        """Ising2D should use int8 for spins."""
        model = Ising2D(size=32, temperature=2.269)
        model.initialize('random')

        assert model.spins.dtype == np.int8

    def test_ising2d_memory_size(self):
        """Ising2D memory should be minimal."""
        model = Ising2D(size=64, temperature=2.269)
        model.initialize('random')

        # 64x64 int8 = 4096 bytes
        assert model.spins.nbytes == 64 * 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
