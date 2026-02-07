"""Tests for configuration file handling."""

import tempfile
from pathlib import Path

import pytest

from ising_toolkit.io.config import (
    SimulationConfig,
    AlgorithmConfig,
    RunConfig,
    OutputConfig,
    Config,
    load_config,
    merge_configs,
    create_sweep_configs,
    create_size_sweep_configs,
    get_default_config,
    DEFAULT_CONFIG,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_config():
    """Create basic configuration."""
    return Config(
        simulation=SimulationConfig(
            model='ising2d',
            size=32,
            temperature=2.269,
        )
    )


@pytest.fixture
def full_config():
    """Create configuration with all sections."""
    return Config(
        simulation=SimulationConfig(
            model='ising2d',
            size=[32, 32],
            temperature=[2.0, 2.2, 2.4],
            coupling=1.0,
            field=0.0,
            boundary='periodic',
            initial_state='random',
        ),
        algorithm=AlgorithmConfig(
            name='metropolis',
            seed=42,
        ),
        run=RunConfig(
            n_steps=50000,
            equilibration=5000,
            measurement_interval=10,
            save_configurations=True,
            configuration_interval=500,
        ),
        output=OutputConfig(
            directory='output',
            format='hdf5',
            filename_template='{model}_L{size}_T{temperature:.3f}',
        ),
    )


@pytest.fixture
def sample_yaml_content():
    """Sample YAML configuration content."""
    return """
simulation:
  model: ising2d
  size: 32
  temperature: 2.269
  coupling: 1.0
  boundary: periodic
  initial_state: random

algorithm:
  name: metropolis
  seed: 42

run:
  n_steps: 100000
  equilibration: 10000
  measurement_interval: 10

output:
  directory: results
  format: hdf5
"""


# ============================================================================
# Tests for SimulationConfig
# ============================================================================

class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        config = SimulationConfig(
            model='ising2d',
            size=32,
            temperature=2.269,
        )

        assert config.model == 'ising2d'
        assert config.size == 32
        assert config.temperature == 2.269

    def test_default_values(self):
        """Test default values."""
        config = SimulationConfig(
            model='ising2d',
            size=32,
            temperature=2.269,
        )

        assert config.coupling == 1.0
        assert config.field == 0.0
        assert config.boundary == 'periodic'
        assert config.initial_state == 'random'

    def test_list_size(self):
        """Test list size for anisotropic lattice."""
        config = SimulationConfig(
            model='ising2d',
            size=[32, 64],
            temperature=2.269,
        )

        assert config.size == [32, 64]

    def test_list_temperature(self):
        """Test list of temperatures for sweep."""
        config = SimulationConfig(
            model='ising2d',
            size=32,
            temperature=[2.0, 2.2, 2.4],
        )

        assert config.temperature == [2.0, 2.2, 2.4]

    def test_invalid_model_raises(self):
        """Test error for invalid model."""
        with pytest.raises(ValueError, match="Invalid model"):
            SimulationConfig(
                model='invalid',
                size=32,
                temperature=2.269,
            )

    def test_invalid_boundary_raises(self):
        """Test error for invalid boundary."""
        with pytest.raises(ValueError, match="Invalid boundary"):
            SimulationConfig(
                model='ising2d',
                size=32,
                temperature=2.269,
                boundary='invalid',
            )

    def test_invalid_initial_state_raises(self):
        """Test error for invalid initial state."""
        with pytest.raises(ValueError, match="Invalid initial_state"):
            SimulationConfig(
                model='ising2d',
                size=32,
                temperature=2.269,
                initial_state='invalid',
            )

    def test_negative_size_raises(self):
        """Test error for negative size."""
        with pytest.raises(ValueError, match="positive"):
            SimulationConfig(
                model='ising2d',
                size=-32,
                temperature=2.269,
            )

    def test_negative_temperature_raises(self):
        """Test error for negative temperature."""
        with pytest.raises(ValueError, match="positive"):
            SimulationConfig(
                model='ising2d',
                size=32,
                temperature=-1.0,
            )

    def test_all_valid_models(self):
        """Test all valid model types."""
        for model in ['ising1d', 'ising2d', 'ising3d']:
            config = SimulationConfig(
                model=model,
                size=32,
                temperature=2.0,
            )
            assert config.model == model

    def test_all_valid_initial_states(self):
        """Test all valid initial states."""
        for state in ['random', 'up', 'down', 'checkerboard']:
            config = SimulationConfig(
                model='ising2d',
                size=32,
                temperature=2.0,
                initial_state=state,
            )
            assert config.initial_state == state


# ============================================================================
# Tests for AlgorithmConfig
# ============================================================================

class TestAlgorithmConfig:
    """Tests for AlgorithmConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = AlgorithmConfig()

        assert config.name == 'metropolis'
        assert config.seed is None
        assert config.cluster_flip_prob == 1.0

    def test_with_seed(self):
        """Test with seed."""
        config = AlgorithmConfig(seed=42)

        assert config.seed == 42

    def test_wolff_algorithm(self):
        """Test Wolff algorithm."""
        config = AlgorithmConfig(name='wolff')

        assert config.name == 'wolff'

    def test_invalid_algorithm_raises(self):
        """Test error for invalid algorithm."""
        with pytest.raises(ValueError, match="Invalid algorithm"):
            AlgorithmConfig(name='invalid')

    def test_invalid_cluster_flip_prob_raises(self):
        """Test error for invalid cluster flip probability."""
        with pytest.raises(ValueError, match="cluster_flip_prob"):
            AlgorithmConfig(cluster_flip_prob=1.5)


# ============================================================================
# Tests for RunConfig
# ============================================================================

class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = RunConfig()

        assert config.n_steps == 100000
        assert config.equilibration == 10000
        assert config.measurement_interval == 10
        assert config.save_configurations is False
        assert config.configuration_interval == 1000

    def test_custom_values(self):
        """Test custom values."""
        config = RunConfig(
            n_steps=50000,
            equilibration=5000,
            measurement_interval=5,
            save_configurations=True,
        )

        assert config.n_steps == 50000
        assert config.equilibration == 5000
        assert config.save_configurations is True

    def test_negative_n_steps_raises(self):
        """Test error for negative n_steps."""
        with pytest.raises(ValueError, match="n_steps must be positive"):
            RunConfig(n_steps=-1000)

    def test_equilibration_exceeds_n_steps_raises(self):
        """Test error when equilibration exceeds n_steps."""
        with pytest.raises(ValueError, match="equilibration"):
            RunConfig(n_steps=1000, equilibration=2000)

    def test_negative_measurement_interval_raises(self):
        """Test error for negative measurement interval."""
        with pytest.raises(ValueError, match="measurement_interval"):
            RunConfig(measurement_interval=-10)


# ============================================================================
# Tests for OutputConfig
# ============================================================================

class TestOutputConfig:
    """Tests for OutputConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = OutputConfig()

        assert config.directory == 'results'
        assert config.format == 'hdf5'
        assert config.save_metadata is True
        assert config.compress is True

    def test_custom_values(self):
        """Test custom values."""
        config = OutputConfig(
            directory='output',
            format='npz',
            save_metadata=False,
        )

        assert config.directory == 'output'
        assert config.format == 'npz'
        assert config.save_metadata is False

    def test_invalid_format_raises(self):
        """Test error for invalid format."""
        with pytest.raises(ValueError, match="Invalid format"):
            OutputConfig(format='invalid')

    def test_all_valid_formats(self):
        """Test all valid formats."""
        for fmt in ['hdf5', 'npz', 'csv', 'json']:
            config = OutputConfig(format=fmt)
            assert config.format == fmt


# ============================================================================
# Tests for Config
# ============================================================================

class TestConfig:
    """Tests for Config dataclass."""

    def test_basic_creation(self, basic_config):
        """Test basic creation."""
        assert basic_config.simulation.model == 'ising2d'
        assert basic_config.algorithm.name == 'metropolis'
        assert basic_config.run.n_steps == 100000
        assert basic_config.output.format == 'hdf5'

    def test_from_dict(self):
        """Test creation from dictionary."""
        config = Config.from_dict({
            'simulation': {
                'model': 'ising2d',
                'size': 32,
                'temperature': 2.269,
            },
            'algorithm': {
                'name': 'wolff',
                'seed': 42,
            },
        })

        assert config.simulation.model == 'ising2d'
        assert config.algorithm.name == 'wolff'
        assert config.algorithm.seed == 42

    def test_from_dict_missing_simulation_raises(self):
        """Test error when simulation section is missing."""
        with pytest.raises(ValueError, match="simulation"):
            Config.from_dict({'algorithm': {'name': 'metropolis'}})

    def test_to_dict(self, basic_config):
        """Test conversion to dictionary."""
        d = basic_config.to_dict()

        assert isinstance(d, dict)
        assert 'simulation' in d
        assert 'algorithm' in d
        assert 'run' in d
        assert 'output' in d
        assert d['simulation']['model'] == 'ising2d'

    def test_roundtrip_dict(self, full_config):
        """Test roundtrip through dictionary."""
        d = full_config.to_dict()
        restored = Config.from_dict(d)

        assert restored.simulation.model == full_config.simulation.model
        assert restored.simulation.size == full_config.simulation.size
        assert restored.algorithm.seed == full_config.algorithm.seed
        assert restored.run.n_steps == full_config.run.n_steps

    def test_validate(self, basic_config):
        """Test validate method."""
        # Should not raise
        basic_config.validate()

    def test_cross_validate_wolff_periodic(self):
        """Test Wolff requires periodic boundaries."""
        with pytest.raises(ValueError, match="periodic"):
            Config(
                simulation=SimulationConfig(
                    model='ising2d',
                    size=32,
                    temperature=2.269,
                    boundary='open',
                ),
                algorithm=AlgorithmConfig(name='wolff'),
            ).validate()

    def test_cross_validate_1d_size(self):
        """Test 1D model size dimensions."""
        with pytest.raises(ValueError, match="1D model"):
            Config(
                simulation=SimulationConfig(
                    model='ising1d',
                    size=[10, 10],
                    temperature=1.0,
                ),
            ).validate()

    def test_get_output_path(self, basic_config):
        """Test output path generation."""
        path = basic_config.get_output_path()

        assert 'ising2d' in str(path)
        assert '32' in str(path)
        assert '2.269' in str(path)
        assert path.suffix == '.h5'

    def test_get_output_path_list_size(self):
        """Test output path with list size."""
        config = Config(
            simulation=SimulationConfig(
                model='ising2d',
                size=[32, 64],
                temperature=2.0,
            )
        )

        path = config.get_output_path()
        assert '32x64' in str(path)

    def test_copy(self, full_config):
        """Test deep copy."""
        copied = full_config.copy()

        # Should be equal
        assert copied.simulation.model == full_config.simulation.model
        assert copied.algorithm.seed == full_config.algorithm.seed

        # Should be independent
        copied.algorithm.seed = 999
        assert full_config.algorithm.seed == 42


# ============================================================================
# Tests for YAML I/O
# ============================================================================

class TestYamlIO:
    """Tests for YAML file I/O."""

    def test_to_yaml(self, basic_config, temp_dir):
        """Test saving to YAML."""
        yaml = pytest.importorskip("yaml")

        filepath = temp_dir / "config.yaml"
        basic_config.to_yaml(str(filepath))

        assert filepath.exists()

        # Verify content
        with open(filepath) as f:
            data = yaml.safe_load(f)

        assert data['simulation']['model'] == 'ising2d'

    def test_from_yaml(self, sample_yaml_content, temp_dir):
        """Test loading from YAML."""
        pytest.importorskip("yaml")

        filepath = temp_dir / "config.yaml"
        with open(filepath, 'w') as f:
            f.write(sample_yaml_content)

        config = Config.from_yaml(str(filepath))

        assert config.simulation.model == 'ising2d'
        assert config.simulation.size == 32
        assert config.algorithm.seed == 42

    def test_roundtrip_yaml(self, full_config, temp_dir):
        """Test roundtrip through YAML file."""
        pytest.importorskip("yaml")

        filepath = temp_dir / "config.yaml"
        full_config.to_yaml(str(filepath))
        restored = Config.from_yaml(str(filepath))

        assert restored.simulation.model == full_config.simulation.model
        assert restored.simulation.size == full_config.simulation.size
        assert restored.algorithm.seed == full_config.algorithm.seed
        assert restored.run.n_steps == full_config.run.n_steps

    def test_from_yaml_file_not_found(self):
        """Test error for missing file."""
        pytest.importorskip("yaml")

        with pytest.raises(FileNotFoundError):
            Config.from_yaml('nonexistent.yaml')

    def test_from_yaml_empty_file(self, temp_dir):
        """Test error for empty file."""
        pytest.importorskip("yaml")

        filepath = temp_dir / "empty.yaml"
        filepath.touch()

        with pytest.raises(ValueError, match="Empty"):
            Config.from_yaml(str(filepath))

    def test_to_yaml_creates_directory(self, basic_config, temp_dir):
        """Test that to_yaml creates parent directories."""
        pytest.importorskip("yaml")

        filepath = temp_dir / "nested" / "dir" / "config.yaml"
        basic_config.to_yaml(str(filepath))

        assert filepath.exists()


# ============================================================================
# Tests for load_config
# ============================================================================

class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config(self, sample_yaml_content, temp_dir):
        """Test load_config function."""
        pytest.importorskip("yaml")

        filepath = temp_dir / "config.yaml"
        with open(filepath, 'w') as f:
            f.write(sample_yaml_content)

        config = load_config(str(filepath))

        assert config.simulation.model == 'ising2d'
        assert config.algorithm.seed == 42

    def test_load_config_validates(self, temp_dir):
        """Test that load_config validates configuration."""
        yaml = pytest.importorskip("yaml")

        # Create invalid config (Wolff with open boundaries)
        invalid_config = {
            'simulation': {
                'model': 'ising2d',
                'size': 32,
                'temperature': 2.269,
                'boundary': 'open',
            },
            'algorithm': {
                'name': 'wolff',
            },
        }

        filepath = temp_dir / "invalid.yaml"
        with open(filepath, 'w') as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError, match="periodic"):
            load_config(str(filepath))


# ============================================================================
# Tests for merge_configs
# ============================================================================

class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_merge_temperature(self, basic_config):
        """Test merging temperature override."""
        merged = merge_configs(basic_config, {
            'simulation': {'temperature': 3.0}
        })

        assert merged.simulation.temperature == 3.0
        # Original unchanged
        assert basic_config.simulation.temperature == 2.269

    def test_merge_multiple_fields(self, basic_config):
        """Test merging multiple fields."""
        merged = merge_configs(basic_config, {
            'simulation': {'temperature': 3.0, 'size': 64},
            'algorithm': {'name': 'wolff'},
        })

        assert merged.simulation.temperature == 3.0
        assert merged.simulation.size == 64
        assert merged.algorithm.name == 'wolff'

    def test_merge_nested_override(self, basic_config):
        """Test deep merge preserves unmentioned fields."""
        merged = merge_configs(basic_config, {
            'simulation': {'temperature': 3.0}
        })

        # Unmentioned fields preserved
        assert merged.simulation.model == 'ising2d'
        assert merged.simulation.size == 32
        assert merged.simulation.coupling == 1.0


# ============================================================================
# Tests for sweep config creators
# ============================================================================

class TestSweepConfigs:
    """Tests for sweep configuration creators."""

    def test_create_sweep_configs(self, basic_config):
        """Test temperature sweep configuration."""
        temps = [2.0, 2.2, 2.4, 2.6]
        configs = create_sweep_configs(basic_config, temps)

        assert len(configs) == 4
        assert configs[0].simulation.temperature == 2.0
        assert configs[1].simulation.temperature == 2.2
        assert configs[2].simulation.temperature == 2.4
        assert configs[3].simulation.temperature == 2.6

        # Other fields unchanged
        for config in configs:
            assert config.simulation.model == 'ising2d'
            assert config.simulation.size == 32

    def test_create_size_sweep_configs(self, basic_config):
        """Test size sweep configuration."""
        sizes = [8, 16, 32, 64]
        configs = create_size_sweep_configs(basic_config, sizes)

        assert len(configs) == 4
        assert configs[0].simulation.size == 8
        assert configs[1].simulation.size == 16
        assert configs[2].simulation.size == 32
        assert configs[3].simulation.size == 64

        # Other fields unchanged
        for config in configs:
            assert config.simulation.model == 'ising2d'
            assert config.simulation.temperature == 2.269


# ============================================================================
# Tests for get_default_config
# ============================================================================

class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_valid_config(self):
        """Test default config is valid."""
        config = get_default_config()

        assert config.simulation.model == 'ising2d'
        assert config.simulation.size == 32
        assert config.simulation.temperature == 2.269
        config.validate()  # Should not raise

    def test_returns_copy(self):
        """Test each call returns a new copy."""
        config1 = get_default_config()
        config2 = get_default_config()

        config1.simulation.size = 64
        assert config2.simulation.size == 32


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_full_workflow(self, temp_dir):
        """Test full configuration workflow."""
        pytest.importorskip("yaml")

        # Create config
        config = Config(
            simulation=SimulationConfig(
                model='ising2d',
                size=32,
                temperature=2.269,
            ),
            algorithm=AlgorithmConfig(
                name='metropolis',
                seed=42,
            ),
            run=RunConfig(
                n_steps=10000,
                equilibration=1000,
            ),
        )

        # Validate
        config.validate()

        # Save to YAML
        filepath = temp_dir / "config.yaml"
        config.to_yaml(str(filepath))

        # Load back
        loaded = load_config(str(filepath))

        # Verify
        assert loaded.simulation.model == config.simulation.model
        assert loaded.algorithm.seed == config.algorithm.seed
        assert loaded.run.n_steps == config.run.n_steps

        # Create sweep
        temps = [2.0, 2.2, 2.4]
        sweep_configs = create_sweep_configs(loaded, temps)

        assert len(sweep_configs) == 3
        for i, T in enumerate(temps):
            assert sweep_configs[i].simulation.temperature == T

    def test_config_matches_default_dict(self):
        """Test default config matches DEFAULT_CONFIG dict."""
        config = get_default_config()
        d = config.to_dict()

        assert d['simulation']['model'] == DEFAULT_CONFIG['simulation']['model']
        assert d['run']['n_steps'] == DEFAULT_CONFIG['run']['n_steps']
        assert d['output']['format'] == DEFAULT_CONFIG['output']['format']
