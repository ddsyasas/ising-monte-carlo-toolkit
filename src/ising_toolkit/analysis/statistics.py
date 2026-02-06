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


def autocorrelation_function(
    data: np.ndarray,
    max_lag: Optional[int] = None,
) -> np.ndarray:
    """Compute normalized autocorrelation function.

    The autocorrelation function measures how correlated a time series
    is with itself at different time lags:

        A(t) = ⟨(x(τ) - ⟨x⟩)(x(τ+t) - ⟨x⟩)⟩ / ⟨(x - ⟨x⟩)²⟩

    Parameters
    ----------
    data : np.ndarray
        1D array of time series data.
    max_lag : int, optional
        Maximum lag to compute. Default is len(data) // 4.

    Returns
    -------
    np.ndarray
        Array of A(t) for t = 0, 1, ..., max_lag.
        A(0) = 1 by definition.

    Notes
    -----
    The autocorrelation function decays from 1 at t=0 to ~0 for
    uncorrelated data. For exponentially correlated data (e.g., from
    Monte Carlo), it decays as A(t) ≈ exp(-t/τ).

    This implementation uses the direct method, which is O(n * max_lag).
    For very long time series, FFT-based methods are more efficient.

    Examples
    --------
    >>> # Generate AR(1) process with known autocorrelation
    >>> tau = 10.0
    >>> alpha = np.exp(-1/tau)
    >>> data = generate_ar1(1000, alpha)
    >>> acf = autocorrelation_function(data, max_lag=50)
    >>> # A(t) should decay as alpha^t = exp(-t/tau)
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n == 0:
        raise ValueError("Data array is empty")

    if n == 1:
        return np.array([1.0])

    if max_lag is None:
        max_lag = n // 4

    # Ensure max_lag is valid
    max_lag = min(max_lag, n - 1)

    # Subtract mean
    x = data - np.mean(data)

    # Compute variance (denominator)
    variance = np.var(x, ddof=0)  # Population variance

    if variance == 0:
        # Constant data: return 1 at lag 0, 0 elsewhere
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        return acf

    # Compute autocorrelation for each lag
    acf = np.zeros(max_lag + 1)

    for t in range(max_lag + 1):
        if t == 0:
            acf[t] = 1.0
        else:
            # Autocorrelation at lag t: ⟨x[:-t] * x[t:]⟩ / var
            acf[t] = np.mean(x[:-t] * x[t:]) / variance

    return acf


def autocorrelation_function_fft(
    data: np.ndarray,
    max_lag: Optional[int] = None,
) -> np.ndarray:
    """Compute normalized autocorrelation function using FFT.

    This is a faster O(n log n) implementation using the Wiener-Khinchin
    theorem: the autocorrelation is the inverse FFT of the power spectrum.

    Parameters
    ----------
    data : np.ndarray
        1D array of time series data.
    max_lag : int, optional
        Maximum lag to return. Default is len(data) // 4.

    Returns
    -------
    np.ndarray
        Array of A(t) for t = 0, 1, ..., max_lag.

    Notes
    -----
    Uses zero-padding to avoid circular correlation artifacts.
    More efficient than direct method for large datasets.

    Examples
    --------
    >>> data = np.random.randn(10000)
    >>> acf = autocorrelation_function_fft(data, max_lag=100)
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n == 0:
        raise ValueError("Data array is empty")

    if n == 1:
        return np.array([1.0])

    if max_lag is None:
        max_lag = n // 4

    max_lag = min(max_lag, n - 1)

    # Subtract mean
    x = data - np.mean(data)

    # Zero-pad to avoid circular correlation
    n_padded = 2 * n

    # FFT-based autocorrelation
    fft_x = np.fft.fft(x, n=n_padded)
    power_spectrum = fft_x * np.conj(fft_x)
    acf_full = np.fft.ifft(power_spectrum).real

    # Normalize by number of overlapping points and variance
    # acf_full[t] = sum_{i=0}^{n-1-t} x[i]*x[i+t], need to divide by (n-t)
    normalization = np.arange(n, n - max_lag - 1, -1)
    acf = acf_full[: max_lag + 1] / normalization

    # Normalize to get correlation (divide by variance)
    if acf[0] != 0:
        acf = acf / acf[0]

    return acf


def integrated_autocorrelation_time(
    data: np.ndarray,
    max_lag: Optional[int] = None,
    c: float = 6.0,
) -> float:
    """Compute integrated autocorrelation time.

    The integrated autocorrelation time is defined as:

        τ_int = 1/2 + Σ_{t=1}^{∞} A(t)

    where A(t) is the normalized autocorrelation function.

    Parameters
    ----------
    data : np.ndarray
        1D array of time series data.
    max_lag : int, optional
        Maximum lag to consider. Default is len(data) // 4.
    c : float, optional
        Window parameter for automatic truncation. The sum is truncated
        when t > c * τ_int. Default is 6.0 (recommended by Sokal).

    Returns
    -------
    float
        Integrated autocorrelation time τ_int.

    Notes
    -----
    The integrated autocorrelation time determines the statistical
    efficiency of the Monte Carlo sampling:

    - The effective number of independent samples is N_eff = N / (2τ_int)
    - The true variance of the mean is Var(mean) = σ² * 2τ_int / N

    This implementation uses automatic windowing (Sokal's method):
    sum until A(t) ≤ 0 or t > c * τ_int (running estimate).

    For an AR(1) process with parameter α = exp(-1/τ):
        τ_int = (1 + α) / (2(1 - α)) ≈ τ for large τ

    Examples
    --------
    >>> # Uncorrelated data: τ_int ≈ 0.5
    >>> data = np.random.randn(10000)
    >>> tau = integrated_autocorrelation_time(data)
    >>> print(f"τ_int = {tau:.2f}")  # Should be ~0.5

    >>> # Correlated data: τ_int > 0.5
    >>> tau = 20.0
    >>> alpha = np.exp(-1/tau)
    >>> correlated = generate_ar1(10000, alpha)
    >>> tau_est = integrated_autocorrelation_time(correlated)

    References
    ----------
    A. D. Sokal, "Monte Carlo Methods in Statistical Mechanics"
    Lecture notes, 1997.
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n == 0:
        raise ValueError("Data array is empty")

    if n == 1:
        return 0.5

    if max_lag is None:
        max_lag = n // 4

    # Compute autocorrelation function
    acf = autocorrelation_function(data, max_lag)

    # Integrate with automatic windowing
    tau_int = 0.5  # Start with 1/2

    for t in range(1, len(acf)):
        # Check automatic windowing condition
        if t > c * tau_int:
            break

        # Stop if autocorrelation becomes negative
        if acf[t] <= 0:
            break

        tau_int += acf[t]

    return float(tau_int)


def effective_sample_size(
    data: np.ndarray,
    max_lag: Optional[int] = None,
) -> float:
    """Compute effective number of independent samples.

    For correlated data, the effective sample size is smaller than
    the actual number of data points:

        N_eff = N / (2 * τ_int)

    where τ_int is the integrated autocorrelation time.

    Parameters
    ----------
    data : np.ndarray
        1D array of time series data.
    max_lag : int, optional
        Maximum lag for autocorrelation calculation.
        Default is len(data) // 4.

    Returns
    -------
    float
        Effective number of independent samples.

    Notes
    -----
    The effective sample size is useful for:

    - Estimating true statistical uncertainty: SE = std / sqrt(N_eff)
    - Assessing Monte Carlo efficiency
    - Determining appropriate thinning intervals

    For uncorrelated data, N_eff ≈ N (since τ_int ≈ 0.5).
    For highly correlated data, N_eff << N.

    Examples
    --------
    >>> # Uncorrelated data
    >>> data = np.random.randn(1000)
    >>> n_eff = effective_sample_size(data)
    >>> print(f"N_eff = {n_eff:.0f}")  # Should be ~1000

    >>> # Correlated Monte Carlo data
    >>> energy = run_monte_carlo(...)
    >>> n_eff = effective_sample_size(energy)
    >>> true_error = np.std(energy) / np.sqrt(n_eff)
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n == 0:
        raise ValueError("Data array is empty")

    if n == 1:
        return 1.0

    # Compute integrated autocorrelation time
    tau_int = integrated_autocorrelation_time(data, max_lag)

    # Effective sample size
    n_eff = n / (2 * tau_int)

    # N_eff cannot exceed N
    return min(float(n_eff), float(n))


def generate_ar1(
    n: int,
    alpha: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate AR(1) (autoregressive) process for testing.

    Generates samples from:
        x[t] = α * x[t-1] + ε[t]

    where ε[t] ~ N(0, 1-α²) to ensure unit variance.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    alpha : float
        Autoregressive parameter, must be in (-1, 1).
        Higher |α| means stronger correlation.
        α = exp(-1/τ) gives autocorrelation time τ.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        AR(1) time series with unit variance.

    Notes
    -----
    The autocorrelation function of AR(1) is A(t) = α^t.
    The integrated autocorrelation time is:
        τ_int = (1 + α) / (2(1 - α))

    Examples
    --------
    >>> # Generate AR(1) with correlation time ~10
    >>> tau = 10.0
    >>> alpha = np.exp(-1/tau)
    >>> data = generate_ar1(10000, alpha, seed=42)
    >>> tau_est = integrated_autocorrelation_time(data)
    """
    if not -1 < alpha < 1:
        raise ValueError("alpha must be in (-1, 1)")

    rng = np.random.default_rng(seed)

    # Innovation variance to ensure unit variance of x
    innovation_std = np.sqrt(1 - alpha**2)

    # Generate AR(1) process
    x = np.zeros(n)
    x[0] = rng.standard_normal()

    for t in range(1, n):
        x[t] = alpha * x[t - 1] + innovation_std * rng.standard_normal()

    return x
