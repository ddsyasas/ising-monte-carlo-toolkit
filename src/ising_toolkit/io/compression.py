"""
Spin configuration compression utilities.

Provides bit-packing for Ising spin configurations to minimize storage.
Achieves 8x compression by packing 8 spins per byte.

Examples
--------
>>> import numpy as np
>>> from ising_toolkit.io.compression import pack_spins, unpack_spins
>>>
>>> # Create a spin configuration
>>> spins = np.random.choice([-1, 1], size=(32, 32)).astype(np.int8)
>>> print(f"Original size: {spins.nbytes} bytes")
>>>
>>> # Pack spins (8x compression)
>>> packed = pack_spins(spins)
>>> print(f"Packed size: {packed.nbytes} bytes")
>>>
>>> # Unpack back to original
>>> unpacked = unpack_spins(packed, spins.shape)
>>> assert np.array_equal(spins, unpacked)
"""

from typing import Tuple, Union
import numpy as np


def pack_spins(spins: np.ndarray) -> np.ndarray:
    """Pack +1/-1 spins into bits (8 spins per byte).

    Converts spin values: +1 -> 1, -1 -> 0, then packs 8 spins
    into each uint8 byte using bit operations.

    Parameters
    ----------
    spins : np.ndarray
        Spin array with values +1 or -1. Can be any shape.

    Returns
    -------
    np.ndarray
        Packed uint8 array. Shape is (ceil(n_spins / 8),).
        First 8 bytes store the original shape for unpacking.

    Notes
    -----
    The packed format stores:
    - First 8 bytes: original number of dimensions (uint64)
    - Next 8*ndim bytes: original shape (uint64 per dimension)
    - Remaining bytes: packed spin data

    Achieves approximately 8x compression compared to int8 storage.

    Examples
    --------
    >>> spins = np.array([1, -1, 1, 1, -1, -1, 1, -1], dtype=np.int8)
    >>> packed = pack_spins(spins)
    >>> print(f"Compression ratio: {spins.nbytes / packed.nbytes:.1f}x")
    """
    # Flatten spins
    flat = spins.ravel()
    n_spins = len(flat)

    # Convert -1 -> 0, +1 -> 1
    bits = ((flat + 1) // 2).astype(np.uint8)

    # Pad to multiple of 8
    padded_len = ((n_spins + 7) // 8) * 8
    if padded_len > n_spins:
        bits = np.pad(bits, (0, padded_len - n_spins), mode='constant')

    # Reshape into groups of 8
    bits = bits.reshape(-1, 8)

    # Pack 8 bits into each byte
    # Bit 0 is LSB, bit 7 is MSB
    packed = np.zeros(len(bits), dtype=np.uint8)
    for i in range(8):
        packed |= (bits[:, i] << i)

    # Store shape information
    shape = np.array(spins.shape, dtype=np.uint64)
    ndim = np.array([spins.ndim], dtype=np.uint64)

    # Combine header and data
    header = np.concatenate([ndim.view(np.uint8), shape.view(np.uint8)])
    result = np.concatenate([header, packed])

    return result


def unpack_spins(packed: np.ndarray, shape: Tuple[int, ...] = None) -> np.ndarray:
    """Unpack bit-packed spins back to +1/-1 array.

    Parameters
    ----------
    packed : np.ndarray
        Packed uint8 array from pack_spins().
    shape : tuple of int, optional
        Original shape of the spin array. If None, reads from header.
        If provided, still reads header to skip it but uses provided shape.

    Returns
    -------
    np.ndarray
        Unpacked spin array with values +1 and -1 (dtype int8).

    Examples
    --------
    >>> original = np.random.choice([-1, 1], size=(16, 16)).astype(np.int8)
    >>> packed = pack_spins(original)
    >>> unpacked = unpack_spins(packed)
    >>> assert np.array_equal(original, unpacked)
    """
    # Always read header to get data offset
    ndim = int(packed[:8].view(np.uint64)[0])
    header_size = 8 + ndim * 8

    if shape is None:
        # Read shape from header
        shape_bytes = packed[8:header_size]
        shape = tuple(shape_bytes.view(np.uint64).astype(int))

    # Data starts after header
    data = packed[header_size:]

    n_spins = int(np.prod(shape))

    # Unpack bits
    bits = np.zeros(len(data) * 8, dtype=np.uint8)
    for i in range(8):
        bits[i::8] = (data >> i) & 1

    # Trim to actual size and reshape
    bits = bits[:n_spins]

    # Convert 0 -> -1, 1 -> +1
    spins = (bits.astype(np.int8) * 2) - 1

    return spins.reshape(shape)


def get_compression_ratio(spins: np.ndarray) -> float:
    """Calculate the compression ratio for a spin array.

    Parameters
    ----------
    spins : np.ndarray
        Spin array.

    Returns
    -------
    float
        Compression ratio (original size / packed size).
    """
    packed = pack_spins(spins)
    return spins.nbytes / packed.nbytes


def pack_configurations(
    configurations: list,
    compression_level: int = 1,
) -> np.ndarray:
    """Pack a list of spin configurations into a single compressed array.

    Parameters
    ----------
    configurations : list of np.ndarray
        List of spin configuration arrays (all same shape).
    compression_level : int, optional
        Compression level:
        - 0: No compression (just concatenate)
        - 1: Bit-packing only (default)

    Returns
    -------
    np.ndarray
        Packed configuration data.

    Notes
    -----
    The packed format is:
    - First 8 bytes: number of configurations (uint64)
    - Next 8 bytes: bytes per packed configuration (uint64)
    - Next N bytes: shape header (same as pack_spins)
    - Remaining: packed configurations concatenated
    """
    if not configurations:
        return np.array([], dtype=np.uint8)

    n_configs = len(configurations)
    shape = configurations[0].shape

    if compression_level == 0:
        # Just stack and return
        stacked = np.stack(configurations)
        return stacked.astype(np.int8).tobytes()

    # Pack each configuration
    packed_list = []
    for config in configurations:
        packed = pack_spins(config)
        packed_list.append(packed)

    # All packed arrays should have same size
    bytes_per_config = len(packed_list[0])

    # Create header
    header = np.array([n_configs, bytes_per_config], dtype=np.uint64)

    # Concatenate all packed data
    all_packed = np.concatenate(packed_list)

    return np.concatenate([header.view(np.uint8), all_packed])


def unpack_configurations(packed: np.ndarray) -> list:
    """Unpack a packed configuration array back to list of spin arrays.

    Parameters
    ----------
    packed : np.ndarray
        Packed data from pack_configurations().

    Returns
    -------
    list of np.ndarray
        List of spin configuration arrays.
    """
    if len(packed) == 0:
        return []

    # Read header
    header = packed[:16].view(np.uint64)
    n_configs = int(header[0])
    bytes_per_config = int(header[1])

    data = packed[16:]

    # Unpack each configuration
    configurations = []
    for i in range(n_configs):
        start = i * bytes_per_config
        end = start + bytes_per_config
        packed_config = data[start:end]
        config = unpack_spins(packed_config)
        configurations.append(config)

    return configurations


class ConfigurationBuffer:
    """Memory-efficient buffer for storing spin configurations.

    Provides automatic decimation and maximum size limits to manage
    memory usage when collecting many configurations.

    Parameters
    ----------
    max_configurations : int, optional
        Maximum number of configurations to keep. Oldest are dropped.
        Default is None (unlimited).
    decimation : int, optional
        Keep only every Nth configuration. Default is 1 (keep all).
    compress : bool, optional
        Whether to compress stored configurations. Default is False.

    Examples
    --------
    >>> buffer = ConfigurationBuffer(max_configurations=100, decimation=10)
    >>> for step in range(10000):
    ...     model.step()
    ...     buffer.add(step, model.spins)
    >>> print(f"Stored {len(buffer)} configurations")

    >>> # Get configurations as list
    >>> configs = buffer.get_configurations()
    """

    def __init__(
        self,
        max_configurations: int = None,
        decimation: int = 1,
        compress: bool = False,
    ) -> None:
        self.max_configurations = max_configurations
        self.decimation = max(1, decimation)
        self.compress = compress

        self._configurations = []
        self._steps = []
        self._count = 0
        self._shape = None

    def add(self, step: int, spins: np.ndarray) -> bool:
        """Add a configuration to the buffer.

        Parameters
        ----------
        step : int
            Current simulation step.
        spins : np.ndarray
            Spin configuration to store.

        Returns
        -------
        bool
            True if configuration was stored, False if skipped.
        """
        self._count += 1

        # Apply decimation
        if (self._count - 1) % self.decimation != 0:
            return False

        # Store shape on first add
        if self._shape is None:
            self._shape = spins.shape

        # Store configuration
        if self.compress:
            self._configurations.append(pack_spins(spins))
        else:
            self._configurations.append(spins.copy())
        self._steps.append(step)

        # Apply max size limit
        if self.max_configurations is not None:
            while len(self._configurations) > self.max_configurations:
                self._configurations.pop(0)
                self._steps.pop(0)

        return True

    def get_configurations(self) -> list:
        """Get all stored configurations as a list.

        Returns
        -------
        list of np.ndarray
            Stored spin configurations.
        """
        if self.compress:
            return [unpack_spins(p, self._shape) for p in self._configurations]
        return self._configurations.copy()

    def get_steps(self) -> list:
        """Get step indices of stored configurations.

        Returns
        -------
        list of int
            Step indices.
        """
        return self._steps.copy()

    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes.

        Returns
        -------
        int
            Memory usage in bytes.
        """
        if not self._configurations:
            return 0

        if self.compress:
            return sum(c.nbytes for c in self._configurations)
        else:
            return sum(c.nbytes for c in self._configurations)

    def clear(self) -> None:
        """Clear all stored configurations."""
        self._configurations = []
        self._steps = []
        self._count = 0

    def __len__(self) -> int:
        return len(self._configurations)

    def __repr__(self) -> str:
        return (
            f"ConfigurationBuffer("
            f"stored={len(self)}, "
            f"max={self.max_configurations}, "
            f"decimation={self.decimation}, "
            f"compress={self.compress}, "
            f"memory={self.get_memory_usage() / 1024:.1f}KB)"
        )


def estimate_storage_size(
    n_spins: int,
    n_configurations: int,
    compressed: bool = True,
) -> dict:
    """Estimate storage size for configurations.

    Parameters
    ----------
    n_spins : int
        Number of spins per configuration.
    n_configurations : int
        Number of configurations to store.
    compressed : bool, optional
        Whether compression is used. Default is True.

    Returns
    -------
    dict
        Dictionary with 'bytes', 'kb', 'mb' keys.
    """
    if compressed:
        # Bit-packing: 8 spins per byte + header
        bytes_per_config = (n_spins + 7) // 8 + 24  # 24 byte header
    else:
        # int8 storage
        bytes_per_config = n_spins

    total_bytes = bytes_per_config * n_configurations

    return {
        'bytes': total_bytes,
        'kb': total_bytes / 1024,
        'mb': total_bytes / (1024 * 1024),
        'bytes_per_config': bytes_per_config,
        'compression_ratio': n_spins / bytes_per_config if compressed else 1.0,
    }
