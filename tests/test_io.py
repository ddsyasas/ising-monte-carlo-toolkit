"""Tests for I/O utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ising_toolkit.io import SimulationResults
from ising_toolkit.utils import FileFormatError


class TestSimulationResultsBasic:
    """Basic tests for SimulationResults."""

    def test_initialization(self):
        """Test basic initialization."""
        energy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        magnetization = np.array([0.5, -0.5, 0.3, -0.3, 0.1])

        results = SimulationResults(
            energy=energy,
            magnetization=magnetization,
        )

        assert results.n_samples == 5
        np.testing.assert_array_equal(results.energy, energy)
        np.testing.assert_array_equal(results.magnetization, magnetization)

    def test_initialization_with_metadata(self):
        """Test initialization with metadata."""
        energy = np.array([1.0, 2.0, 3.0])
        magnetization = np.array([0.5, -0.5, 0.3])
        metadata = {
            "model_type": "ising2d",
            "size": 32,
            "temperature": 2.269,
            "algorithm": "metropolis",
            "n_steps": 10000,
        }

        results = SimulationResults(
            energy=energy,
            magnetization=magnetization,
            metadata=metadata,
        )

        assert results.metadata["model_type"] == "ising2d"
        assert results.metadata["size"] == 32
        assert results.metadata["temperature"] == 2.269

    def test_abs_magnetization_computed(self):
        """Test that absolute magnetization is computed."""
        energy = np.array([1.0, 2.0, 3.0])
        magnetization = np.array([0.5, -0.5, -0.3])

        results = SimulationResults(
            energy=energy,
            magnetization=magnetization,
        )

        expected_abs_mag = np.array([0.5, 0.5, 0.3])
        np.testing.assert_array_almost_equal(
            results.abs_magnetization, expected_abs_mag
        )

    def test_len(self):
        """Test __len__ returns number of samples."""
        results = SimulationResults(
            energy=np.zeros(100),
            magnetization=np.zeros(100),
        )

        assert len(results) == 100


class TestSimulationResultsStatistics:
    """Tests for statistical properties."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        np.random.seed(42)
        n = 1000
        energy = np.random.normal(-100, 10, n)
        magnetization = np.random.normal(0, 50, n)

        return SimulationResults(
            energy=energy,
            magnetization=magnetization,
        )

    def test_energy_mean(self, sample_results):
        """Test energy mean calculation."""
        expected = np.mean(sample_results.energy)
        assert sample_results.energy_mean == pytest.approx(expected)

    def test_energy_std(self, sample_results):
        """Test energy standard deviation."""
        expected = np.std(sample_results.energy, ddof=1)
        assert sample_results.energy_std == pytest.approx(expected)

    def test_energy_err(self, sample_results):
        """Test energy error is reasonable."""
        # Error should be approximately std / sqrt(n)
        expected_approx = sample_results.energy_std / np.sqrt(sample_results.n_samples)
        assert sample_results.energy_err > 0
        assert sample_results.energy_err < sample_results.energy_std
        # Within factor of 2 of simple estimate
        assert sample_results.energy_err == pytest.approx(expected_approx, rel=0.5)

    def test_magnetization_mean(self, sample_results):
        """Test magnetization mean calculation."""
        expected = np.mean(sample_results.magnetization)
        assert sample_results.magnetization_mean == pytest.approx(expected)

    def test_magnetization_std(self, sample_results):
        """Test magnetization standard deviation."""
        expected = np.std(sample_results.magnetization, ddof=1)
        assert sample_results.magnetization_std == pytest.approx(expected)

    def test_abs_magnetization_mean(self, sample_results):
        """Test absolute magnetization mean."""
        expected = np.mean(np.abs(sample_results.magnetization))
        assert sample_results.abs_magnetization_mean == pytest.approx(expected)

    def test_abs_magnetization_std(self, sample_results):
        """Test absolute magnetization standard deviation."""
        expected = np.std(np.abs(sample_results.magnetization), ddof=1)
        assert sample_results.abs_magnetization_std == pytest.approx(expected)

    def test_get_statistics(self, sample_results):
        """Test get_statistics returns all values."""
        stats = sample_results.get_statistics()

        assert "energy_mean" in stats
        assert "energy_std" in stats
        assert "energy_err" in stats
        assert "magnetization_mean" in stats
        assert "magnetization_std" in stats
        assert "abs_magnetization_mean" in stats
        assert "abs_magnetization_std" in stats
        assert "n_samples" in stats
        assert stats["n_samples"] == sample_results.n_samples


class TestSimulationResultsSaveLoad:
    """Tests for save/load functionality."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results with all features."""
        np.random.seed(42)
        n = 100

        energy = np.random.normal(-100, 10, n)
        magnetization = np.random.normal(0, 50, n)
        metadata = {
            "model_type": "ising2d",
            "size": 32,
            "temperature": 2.269,
            "algorithm": "metropolis",
            "n_steps": 10000,
            "equilibration": 1000,
            "measurement_interval": 10,
            "seed": 42,
            "elapsed_time": 1.234,
        }
        configurations = [
            np.random.choice([-1, 1], size=(32, 32)).astype(np.int8)
            for _ in range(5)
        ]

        return SimulationResults(
            energy=energy,
            magnetization=magnetization,
            metadata=metadata,
            configurations=configurations,
        )

    def test_save_load_roundtrip(self, sample_results):
        """Test that save/load preserves all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.h5"

            # Save
            sample_results.save(filepath)
            assert filepath.exists()

            # Load
            loaded = SimulationResults.load(filepath)

            # Check time series
            np.testing.assert_array_almost_equal(
                loaded.energy, sample_results.energy
            )
            np.testing.assert_array_almost_equal(
                loaded.magnetization, sample_results.magnetization
            )

            # Check derived data
            np.testing.assert_array_almost_equal(
                loaded.abs_magnetization, sample_results.abs_magnetization
            )

    def test_save_load_metadata(self, sample_results):
        """Test that metadata is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.h5"

            sample_results.save(filepath)
            loaded = SimulationResults.load(filepath)

            assert loaded.metadata["model_type"] == "ising2d"
            assert loaded.metadata["size"] == 32
            assert loaded.metadata["temperature"] == pytest.approx(2.269)
            assert loaded.metadata["algorithm"] == "metropolis"
            assert loaded.metadata["n_steps"] == 10000
            assert loaded.metadata["seed"] == 42

    def test_save_load_configurations(self, sample_results):
        """Test that configurations are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.h5"

            sample_results.save(filepath)
            loaded = SimulationResults.load(filepath)

            assert len(loaded.configurations) == 5
            for i in range(5):
                np.testing.assert_array_equal(
                    loaded.configurations[i],
                    sample_results.configurations[i]
                )

    def test_save_load_no_configurations(self):
        """Test save/load without configurations."""
        results = SimulationResults(
            energy=np.array([1.0, 2.0, 3.0]),
            magnetization=np.array([0.5, -0.5, 0.3]),
            metadata={"model_type": "ising1d"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.h5"

            results.save(filepath)
            loaded = SimulationResults.load(filepath)

            assert loaded.configurations == []

    def test_save_load_none_metadata_value(self):
        """Test that None values in metadata are preserved."""
        results = SimulationResults(
            energy=np.array([1.0, 2.0, 3.0]),
            magnetization=np.array([0.5, -0.5, 0.3]),
            metadata={"seed": None, "model_type": "ising2d"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.h5"

            results.save(filepath)
            loaded = SimulationResults.load(filepath)

            assert loaded.metadata["seed"] is None
            assert loaded.metadata["model_type"] == "ising2d"

    def test_save_load_tuple_metadata(self):
        """Test that tuple values in metadata are preserved."""
        results = SimulationResults(
            energy=np.array([1.0, 2.0, 3.0]),
            magnetization=np.array([0.5, -0.5, 0.3]),
            metadata={"size": (32, 32, 32)},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.h5"

            results.save(filepath)
            loaded = SimulationResults.load(filepath)

            assert loaded.metadata["size"] == (32, 32, 32)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileFormatError, match="File not found"):
            SimulationResults.load("/nonexistent/path/file.h5")

    def test_load_invalid_file(self):
        """Test that loading invalid file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid.h5"
            filepath.write_text("not an HDF5 file")

            with pytest.raises(FileFormatError, match="Failed to load"):
                SimulationResults.load(filepath)


class TestSimulationResultsDataFrame:
    """Tests for DataFrame conversion."""

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        energy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        magnetization = np.array([0.5, -0.5, 0.3, -0.3, 0.1])

        results = SimulationResults(
            energy=energy,
            magnetization=magnetization,
        )

        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["step", "energy", "magnetization", "abs_magnetization"]
        np.testing.assert_array_equal(df["step"].values, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(df["energy"].values, energy)
        np.testing.assert_array_equal(df["magnetization"].values, magnetization)


class TestSimulationResultsRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        results = SimulationResults(
            energy=np.zeros(100),
            magnetization=np.zeros(100),
            metadata={
                "model_type": "ising2d",
                "size": 32,
                "temperature": 2.269,
            },
        )

        repr_str = repr(results)

        assert "SimulationResults" in repr_str
        assert "ising2d" in repr_str
        assert "32" in repr_str
        assert "2.269" in repr_str
        assert "n_samples=100" in repr_str

    def test_repr_missing_metadata(self):
        """Test __repr__ with missing metadata."""
        results = SimulationResults(
            energy=np.zeros(10),
            magnetization=np.zeros(10),
        )

        repr_str = repr(results)

        assert "SimulationResults" in repr_str
        assert "unknown" in repr_str  # Default for missing model_type
