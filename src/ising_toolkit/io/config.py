"""Configuration file handling for Ising model simulations."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union
import copy


@dataclass
class SimulationConfig:
    """Configuration for a simulation run.

    Parameters
    ----------
    model : str
        Model type ('ising1d', 'ising2d', 'ising3d').
    size : int or list of int
        System size(s). For 2D: L or [Lx, Ly]. For 3D: L or [Lx, Ly, Lz].
    temperature : float or list of float
        Temperature(s) for simulation. Single value or list for sweep.
    coupling : float, optional
        Coupling constant J. Default is 1.0.
    field : float, optional
        External magnetic field h. Default is 0.0.
    boundary : str, optional
        Boundary conditions ('periodic' or 'open'). Default is 'periodic'.
    initial_state : str, optional
        Initial state ('random', 'up', 'down', 'checkerboard').
        Default is 'random'.

    Examples
    --------
    >>> config = SimulationConfig(
    ...     model='ising2d',
    ...     size=32,
    ...     temperature=2.269
    ... )
    """
    model: str
    size: Union[int, List[int]]
    temperature: Union[float, List[float]]
    coupling: float = 1.0
    field: float = 0.0
    boundary: str = 'periodic'
    initial_state: str = 'random'

    def __post_init__(self):
        """Validate after initialization."""
        self._validate()

    def _validate(self):
        """Validate simulation configuration."""
        valid_models = {'ising1d', 'ising2d', 'ising3d'}
        if self.model not in valid_models:
            raise ValueError(
                f"Invalid model '{self.model}'. Must be one of {valid_models}"
            )

        valid_boundaries = {'periodic', 'open'}
        if self.boundary not in valid_boundaries:
            raise ValueError(
                f"Invalid boundary '{self.boundary}'. "
                f"Must be one of {valid_boundaries}"
            )

        valid_initial = {'random', 'up', 'down', 'checkerboard'}
        if self.initial_state not in valid_initial:
            raise ValueError(
                f"Invalid initial_state '{self.initial_state}'. "
                f"Must be one of {valid_initial}"
            )

        # Validate size
        if isinstance(self.size, int):
            if self.size <= 0:
                raise ValueError(f"Size must be positive, got {self.size}")
        elif isinstance(self.size, list):
            if not all(isinstance(s, int) and s > 0 for s in self.size):
                raise ValueError("All sizes must be positive integers")
        else:
            raise TypeError(f"Size must be int or list, got {type(self.size)}")

        # Validate temperature
        if isinstance(self.temperature, (int, float)):
            if self.temperature <= 0:
                raise ValueError(
                    f"Temperature must be positive, got {self.temperature}"
                )
        elif isinstance(self.temperature, list):
            if not all(isinstance(t, (int, float)) and t > 0
                       for t in self.temperature):
                raise ValueError("All temperatures must be positive")
        else:
            raise TypeError(
                f"Temperature must be float or list, got {type(self.temperature)}"
            )


@dataclass
class AlgorithmConfig:
    """Algorithm configuration.

    Parameters
    ----------
    name : str, optional
        Algorithm name ('metropolis' or 'wolff'). Default is 'metropolis'.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    cluster_flip_prob : float, optional
        For Wolff algorithm, probability to flip cluster. Default is 1.0.

    Examples
    --------
    >>> config = AlgorithmConfig(name='wolff', seed=42)
    """
    name: str = 'metropolis'
    seed: Optional[int] = None
    cluster_flip_prob: float = 1.0

    def __post_init__(self):
        """Validate after initialization."""
        self._validate()

    def _validate(self):
        """Validate algorithm configuration."""
        valid_algorithms = {'metropolis', 'wolff'}
        if self.name not in valid_algorithms:
            raise ValueError(
                f"Invalid algorithm '{self.name}'. "
                f"Must be one of {valid_algorithms}"
            )

        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError(f"Seed must be int or None, got {type(self.seed)}")

        if not 0 <= self.cluster_flip_prob <= 1:
            raise ValueError(
                f"cluster_flip_prob must be in [0, 1], "
                f"got {self.cluster_flip_prob}"
            )


@dataclass
class RunConfig:
    """Run parameters.

    Parameters
    ----------
    n_steps : int, optional
        Total number of Monte Carlo steps. Default is 100000.
    equilibration : int, optional
        Number of equilibration steps (discarded). Default is 10000.
    measurement_interval : int, optional
        Measure observables every N steps. Default is 10.
    save_configurations : bool, optional
        Save spin configurations. Default is False.
    configuration_interval : int, optional
        Save configuration every N steps. Default is 1000.

    Examples
    --------
    >>> config = RunConfig(n_steps=50000, equilibration=5000)
    """
    n_steps: int = 100000
    equilibration: int = 10000
    measurement_interval: int = 10
    save_configurations: bool = False
    configuration_interval: int = 1000

    def __post_init__(self):
        """Validate after initialization."""
        self._validate()

    def _validate(self):
        """Validate run configuration."""
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")

        if self.equilibration < 0:
            raise ValueError(
                f"equilibration must be non-negative, got {self.equilibration}"
            )

        if self.equilibration >= self.n_steps:
            raise ValueError(
                f"equilibration ({self.equilibration}) must be less than "
                f"n_steps ({self.n_steps})"
            )

        if self.measurement_interval <= 0:
            raise ValueError(
                f"measurement_interval must be positive, "
                f"got {self.measurement_interval}"
            )

        if self.configuration_interval <= 0:
            raise ValueError(
                f"configuration_interval must be positive, "
                f"got {self.configuration_interval}"
            )


@dataclass
class OutputConfig:
    """Output settings.

    Parameters
    ----------
    directory : str, optional
        Output directory. Default is 'results'.
    format : str, optional
        Output format ('hdf5', 'npz', 'csv'). Default is 'hdf5'.
    filename_template : str, optional
        Template for output filenames. Default is
        '{model}_L{size}_T{temperature:.3f}'.
    save_metadata : bool, optional
        Save simulation metadata. Default is True.
    compress : bool, optional
        Compress output files. Default is True.

    Examples
    --------
    >>> config = OutputConfig(directory='output', format='npz')
    """
    directory: str = 'results'
    format: str = 'hdf5'
    filename_template: str = '{model}_L{size}_T{temperature:.3f}'
    save_metadata: bool = True
    compress: bool = True

    def __post_init__(self):
        """Validate after initialization."""
        self._validate()

    def _validate(self):
        """Validate output configuration."""
        valid_formats = {'hdf5', 'npz', 'csv', 'json'}
        if self.format not in valid_formats:
            raise ValueError(
                f"Invalid format '{self.format}'. Must be one of {valid_formats}"
            )


@dataclass
class Config:
    """Complete simulation configuration.

    Parameters
    ----------
    simulation : SimulationConfig
        Simulation parameters.
    algorithm : AlgorithmConfig, optional
        Algorithm settings. Default creates default AlgorithmConfig.
    run : RunConfig, optional
        Run parameters. Default creates default RunConfig.
    output : OutputConfig, optional
        Output settings. Default creates default OutputConfig.

    Examples
    --------
    >>> config = Config(
    ...     simulation=SimulationConfig(
    ...         model='ising2d',
    ...         size=32,
    ...         temperature=2.269
    ...     )
    ... )
    >>> config.to_yaml('simulation.yaml')
    """
    simulation: SimulationConfig
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    run: RunConfig = field(default_factory=RunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load configuration from YAML file.

        Parameters
        ----------
        filepath : str
            Path to YAML configuration file.

        Returns
        -------
        Config
            Loaded configuration.

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        ValueError
            If YAML is invalid or missing required fields.

        Examples
        --------
        >>> config = Config.from_yaml('simulation.yaml')
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML required for YAML support. "
                "Install with: pip install pyyaml"
            )

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty configuration file: {filepath}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        """Create configuration from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with configuration data.

        Returns
        -------
        Config
            Configuration object.

        Raises
        ------
        ValueError
            If required fields are missing.
        """
        if 'simulation' not in d:
            raise ValueError("Configuration must include 'simulation' section")

        simulation = SimulationConfig(**d['simulation'])

        algorithm = AlgorithmConfig(**d.get('algorithm', {}))
        run = RunConfig(**d.get('run', {}))
        output = OutputConfig(**d.get('output', {}))

        return cls(
            simulation=simulation,
            algorithm=algorithm,
            run=run,
            output=output
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as nested dictionary.
        """
        return {
            'simulation': asdict(self.simulation),
            'algorithm': asdict(self.algorithm),
            'run': asdict(self.run),
            'output': asdict(self.output),
        }

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file.

        Parameters
        ----------
        filepath : str
            Output file path.

        Examples
        --------
        >>> config.to_yaml('simulation.yaml')
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML required for YAML support. "
                "Install with: pip install pyyaml"
            )

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> None:
        """Validate all configuration values.

        Calls validation on all sub-configurations and performs
        cross-validation checks.

        Raises
        ------
        ValueError
            If any validation fails.
        """
        # Individual validations are called in __post_init__
        self.simulation._validate()
        self.algorithm._validate()
        self.run._validate()
        self.output._validate()

        # Cross-validation checks
        self._cross_validate()

    def _cross_validate(self):
        """Perform cross-validation between config sections."""
        # Check model-specific constraints
        model = self.simulation.model
        size = self.simulation.size

        if model == 'ising1d':
            if isinstance(size, list) and len(size) != 1:
                raise ValueError(
                    f"1D model requires 1 size dimension, got {len(size)}"
                )
        elif model == 'ising2d':
            if isinstance(size, list) and len(size) not in (1, 2):
                raise ValueError(
                    f"2D model requires 1-2 size dimensions, got {len(size)}"
                )
        elif model == 'ising3d':
            if isinstance(size, list) and len(size) not in (1, 3):
                raise ValueError(
                    f"3D model requires 1 or 3 size dimensions, got {len(size)}"
                )

        # Wolff algorithm requires periodic boundaries
        if (self.algorithm.name == 'wolff' and
                self.simulation.boundary != 'periodic'):
            raise ValueError(
                "Wolff algorithm requires periodic boundary conditions"
            )

    def get_output_path(self, temperature: Optional[float] = None) -> Path:
        """Generate output file path from template.

        Parameters
        ----------
        temperature : float, optional
            Temperature for filename. Uses first temperature if not provided.

        Returns
        -------
        Path
            Full output file path.
        """
        if temperature is None:
            temp = self.simulation.temperature
            if isinstance(temp, list):
                temperature = temp[0]
            else:
                temperature = temp

        size = self.simulation.size
        if isinstance(size, list):
            size_str = 'x'.join(map(str, size))
        else:
            size_str = str(size)

        filename = self.output.filename_template.format(
            model=self.simulation.model,
            size=size_str,
            temperature=temperature,
        )

        ext = {'hdf5': '.h5', 'npz': '.npz', 'csv': '.csv', 'json': '.json'}
        filename += ext.get(self.output.format, '')

        return Path(self.output.directory) / filename

    def copy(self) -> 'Config':
        """Create a deep copy of this configuration.

        Returns
        -------
        Config
            Deep copy of configuration.
        """
        return Config.from_dict(copy.deepcopy(self.to_dict()))


def load_config(filepath: str) -> Config:
    """Load and validate configuration file.

    Parameters
    ----------
    filepath : str
        Path to configuration file (YAML).

    Returns
    -------
    Config
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If configuration is invalid.

    Examples
    --------
    >>> config = load_config('simulation.yaml')
    """
    config = Config.from_yaml(filepath)
    config.validate()
    return config


def merge_configs(base: Config, overrides: dict) -> Config:
    """Merge override values into base configuration.

    Creates a new configuration with values from base, updated with
    any values specified in overrides.

    Parameters
    ----------
    base : Config
        Base configuration.
    overrides : dict
        Dictionary of values to override. Can be nested.

    Returns
    -------
    Config
        New configuration with merged values.

    Examples
    --------
    >>> base = load_config('base.yaml')
    >>> config = merge_configs(base, {'simulation': {'temperature': 2.5}})
    """
    base_dict = base.to_dict()
    merged = _deep_merge(base_dict, overrides)
    return Config.from_dict(merged)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Deep merge two dictionaries.

    Parameters
    ----------
    base : dict
        Base dictionary.
    overrides : dict
        Override values.

    Returns
    -------
    dict
        Merged dictionary.
    """
    result = copy.deepcopy(base)

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def create_sweep_configs(
    base: Config,
    temperatures: List[float],
) -> List[Config]:
    """Create configurations for temperature sweep.

    Parameters
    ----------
    base : Config
        Base configuration.
    temperatures : list of float
        Temperatures for sweep.

    Returns
    -------
    list of Config
        List of configurations, one per temperature.

    Examples
    --------
    >>> base = load_config('base.yaml')
    >>> configs = create_sweep_configs(base, [2.0, 2.2, 2.4, 2.6])
    """
    configs = []
    for T in temperatures:
        config = merge_configs(base, {'simulation': {'temperature': T}})
        configs.append(config)
    return configs


def create_size_sweep_configs(
    base: Config,
    sizes: List[int],
) -> List[Config]:
    """Create configurations for system size sweep.

    Parameters
    ----------
    base : Config
        Base configuration.
    sizes : list of int
        System sizes for sweep.

    Returns
    -------
    list of Config
        List of configurations, one per size.

    Examples
    --------
    >>> base = load_config('base.yaml')
    >>> configs = create_size_sweep_configs(base, [8, 16, 32, 64])
    """
    configs = []
    for L in sizes:
        config = merge_configs(base, {'simulation': {'size': L}})
        configs.append(config)
    return configs


# Default configuration template
DEFAULT_CONFIG = {
    'simulation': {
        'model': 'ising2d',
        'size': 32,
        'temperature': 2.269,
        'coupling': 1.0,
        'field': 0.0,
        'boundary': 'periodic',
        'initial_state': 'random',
    },
    'algorithm': {
        'name': 'metropolis',
        'seed': None,
        'cluster_flip_prob': 1.0,
    },
    'run': {
        'n_steps': 100000,
        'equilibration': 10000,
        'measurement_interval': 10,
        'save_configurations': False,
        'configuration_interval': 1000,
    },
    'output': {
        'directory': 'results',
        'format': 'hdf5',
        'filename_template': '{model}_L{size}_T{temperature:.3f}',
        'save_metadata': True,
        'compress': True,
    },
}


def get_default_config() -> Config:
    """Get default configuration.

    Returns
    -------
    Config
        Default configuration for 2D Ising model at Tc.

    Examples
    --------
    >>> config = get_default_config()
    >>> config.simulation.model
    'ising2d'
    """
    return Config.from_dict(DEFAULT_CONFIG)
