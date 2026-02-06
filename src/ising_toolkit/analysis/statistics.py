"""Statistical analysis tools including bootstrap resampling."""

from typing import Callable, Tuple, Optional

import numpy as np


def bootstrap_error(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_samples: int = 1000,
    confidence: float = 0.68,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Estimate error using bootstrap resampling.

    Bootstrap resampling provides a non-parametric way to estimate
    statistical uncertainty. It works by repeatedly resampling the
    data with replacement and computing the statistic of interest
    on each resample.

    Parameters
    ----------
    data : np.ndarray
        1D array of measurements.
    statistic : callable, optional
        Function to compute on data. Must accept a 1D array and return
        a scalar. Default is np.mean.
    n_samples : int, optional
        Number of bootstrap samples to generate. Default is 1000.
        More samples give more accurate error estimates.
    confidence : float, optional
        Confidence level for error estimation. Default is 0.68,
        corresponding to 1-sigma (one standard deviation).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    estimate : float
        Point estimate (statistic applied to original data).
    error : float
        Bootstrap standard error, computed as the standard deviation
        of the bootstrap distribution.

    Notes
    -----
    The bootstrap method assumes that the empirical distribution of
    the data approximates the true underlying distribution. It works
    well for:
    - Estimating uncertainty in means, medians, variances
    - Non-Gaussian distributions
    - Complex statistics without analytical error formulas

    The confidence parameter is stored for potential future use with
    percentile-based confidence intervals, but the current implementation
    returns the standard error (standard deviation of bootstrap samples).

    For correlated data (e.g., Monte Carlo time series), consider using
    block bootstrap or computing the correlation time first.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> mean, error = bootstrap_error(data, np.mean)
    >>> print(f"Mean: {mean:.3f} ± {error:.3f}")

    >>> # Custom statistic: median
    >>> median, error = bootstrap_error(data, np.median, n_samples=2000)

    >>> # Variance with reproducible results
    >>> var, error = bootstrap_error(data, np.var, seed=42)
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n == 0:
        raise ValueError("Data array is empty")

    if n == 1:
        # Single data point: return it with zero error
        return float(statistic(data)), 0.0

    # Set random seed if provided
    rng = np.random.default_rng(seed)

    # Compute point estimate on original data
    estimate = float(statistic(data))

    # Generate bootstrap samples and compute statistics
    bootstrap_statistics = np.zeros(n_samples)

    for i in range(n_samples):
        # Resample with replacement
        indices = rng.integers(0, n, size=n)
        resample = data[indices]

        # Compute statistic on resample
        bootstrap_statistics[i] = statistic(resample)

    # Compute bootstrap standard error
    error = float(np.std(bootstrap_statistics, ddof=1))

    return estimate, error


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval using percentile method.

    Parameters
    ----------
    data : np.ndarray
        1D array of measurements.
    statistic : callable, optional
        Function to compute on data. Default is np.mean.
    n_samples : int, optional
        Number of bootstrap samples. Default is 1000.
    confidence : float, optional
        Confidence level. Default is 0.95 (95% confidence interval).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    estimate : float
        Point estimate (statistic applied to original data).
    lower : float
        Lower bound of confidence interval.
    upper : float
        Upper bound of confidence interval.

    Examples
    --------
    >>> data = np.random.normal(10, 2, 100)
    >>> mean, lower, upper = bootstrap_confidence_interval(data)
    >>> print(f"Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n == 0:
        raise ValueError("Data array is empty")

    if n == 1:
        val = float(statistic(data))
        return val, val, val

    rng = np.random.default_rng(seed)

    # Point estimate
    estimate = float(statistic(data))

    # Generate bootstrap distribution
    bootstrap_statistics = np.zeros(n_samples)

    for i in range(n_samples):
        indices = rng.integers(0, n, size=n)
        resample = data[indices]
        bootstrap_statistics[i] = statistic(resample)

    # Compute percentile confidence interval
    alpha = 1 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower = float(np.percentile(bootstrap_statistics, lower_percentile))
    upper = float(np.percentile(bootstrap_statistics, upper_percentile))

    return estimate, lower, upper


def bootstrap_mean_error(
    data: np.ndarray,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Convenience function for bootstrap error of the mean.

    This is a simplified wrapper around bootstrap_error for the
    common case of estimating the uncertainty in the sample mean.

    Parameters
    ----------
    data : np.ndarray
        1D array of measurements.
    n_samples : int, optional
        Number of bootstrap samples. Default is 1000.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    mean : float
        Sample mean.
    error : float
        Bootstrap standard error of the mean.

    Notes
    -----
    For uncorrelated data from a normal distribution, the bootstrap
    error should approximately equal the standard error of the mean:
        SE = std(data) / sqrt(n)

    Examples
    --------
    >>> data = np.random.normal(5, 2, 100)
    >>> mean, error = bootstrap_mean_error(data, seed=42)
    >>> print(f"Mean: {mean:.3f} ± {error:.3f}")
    """
    return bootstrap_error(data, statistic=np.mean, n_samples=n_samples, seed=seed)


def jackknife_error(
    data: np.ndarray,
    statistic: Callable = np.mean,
) -> Tuple[float, float]:
    """Estimate error using jackknife resampling.

    The jackknife method computes the statistic on n subsamples,
    each with one observation removed. It provides a deterministic
    alternative to bootstrap.

    Parameters
    ----------
    data : np.ndarray
        1D array of measurements.
    statistic : callable, optional
        Function to compute on data. Default is np.mean.

    Returns
    -------
    estimate : float
        Point estimate (statistic on full data).
    error : float
        Jackknife standard error.

    Notes
    -----
    The jackknife error is computed as:
        error = sqrt((n-1)/n * sum((theta_i - theta_bar)^2))

    where theta_i is the statistic computed with observation i removed,
    and theta_bar is the mean of the jackknife statistics.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> mean, error = jackknife_error(data)
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n == 0:
        raise ValueError("Data array is empty")

    if n == 1:
        return float(statistic(data)), 0.0

    # Point estimate
    estimate = float(statistic(data))

    # Compute leave-one-out statistics
    jackknife_stats = np.zeros(n)

    for i in range(n):
        # Create subsample with i-th observation removed
        subsample = np.concatenate([data[:i], data[i + 1 :]])
        jackknife_stats[i] = statistic(subsample)

    # Jackknife variance formula
    mean_jackknife = np.mean(jackknife_stats)
    variance = (n - 1) / n * np.sum((jackknife_stats - mean_jackknife) ** 2)
    error = float(np.sqrt(variance))

    return estimate, error


def blocking_error(
    data: np.ndarray,
    min_blocks: int = 10,
) -> Tuple[float, float]:
    """Estimate error using blocking analysis for correlated data.

    Blocking divides the time series into blocks and computes the
    error from block averages. As block size increases, correlations
    within blocks are captured, giving the true statistical error.

    Parameters
    ----------
    data : np.ndarray
        1D array of (possibly correlated) measurements.
    min_blocks : int, optional
        Minimum number of blocks to use. Default is 10.

    Returns
    -------
    mean : float
        Sample mean.
    error : float
        Estimated standard error accounting for correlations.

    Notes
    -----
    The optimal block size is when the error estimate plateaus.
    This implementation uses a heuristic: it tries multiple block
    sizes and returns the maximum error estimate.

    For uncorrelated data, this should match std/sqrt(n).
    For correlated data, this will be larger.

    Examples
    --------
    >>> # Correlated data from Monte Carlo
    >>> data = np.cumsum(np.random.randn(1000))  # Random walk
    >>> mean, error = blocking_error(data)
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n == 0:
        raise ValueError("Data array is empty")

    mean = float(np.mean(data))

    if n < min_blocks:
        # Too few points, use standard error
        return mean, float(np.std(data, ddof=1) / np.sqrt(n))

    # Try different block sizes
    max_block_size = n // min_blocks
    errors = []

    for block_size in range(1, max_block_size + 1):
        n_blocks = n // block_size
        if n_blocks < min_blocks:
            break

        # Compute block averages
        block_data = data[: n_blocks * block_size].reshape(n_blocks, block_size)
        block_means = np.mean(block_data, axis=1)

        # Standard error of block means
        error = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
        errors.append(error)

    # Return maximum error (plateau value)
    return mean, float(np.max(errors)) if errors else float(np.std(data, ddof=1) / np.sqrt(n))
