"""Tests for statistical analysis functions."""

import numpy as np
import pytest

from ising_toolkit.analysis import (
    bootstrap_error,
    bootstrap_confidence_interval,
    bootstrap_mean_error,
    jackknife_error,
    blocking_error,
)


class TestBootstrapError:
    """Tests for bootstrap_error function."""

    def test_bootstrap_error_mean_normal(self):
        """Test bootstrap error for normal distribution matches theory."""
        np.random.seed(42)
        n = 1000
        sigma = 2.0
        data = np.random.normal(0, sigma, n)

        mean, error = bootstrap_error(data, np.mean, n_samples=2000, seed=42)

        # Theoretical standard error of mean: sigma / sqrt(n)
        expected_error = sigma / np.sqrt(n)

        # Bootstrap error should be close to theoretical value
        assert error == pytest.approx(expected_error, rel=0.15)

    def test_bootstrap_error_reproducible(self):
        """Test that seed produces reproducible results."""
        data = np.random.randn(100)

        mean1, error1 = bootstrap_error(data, np.mean, seed=42)
        mean2, error2 = bootstrap_error(data, np.mean, seed=42)

        assert mean1 == mean2
        assert error1 == error2

    def test_bootstrap_error_different_seeds(self):
        """Test that different seeds give different results."""
        data = np.random.randn(100)

        _, error1 = bootstrap_error(data, np.mean, seed=42)
        _, error2 = bootstrap_error(data, np.mean, seed=123)

        # Errors should be similar but not identical
        assert error1 != error2
        assert error1 == pytest.approx(error2, rel=0.5)

    def test_bootstrap_error_median(self):
        """Test bootstrap error for median statistic."""
        np.random.seed(42)
        data = np.random.normal(5, 1, 200)

        median, error = bootstrap_error(data, np.median, n_samples=1000, seed=42)

        # Median should be close to true median (5)
        assert median == pytest.approx(5, abs=0.2)

        # Error should be positive and reasonable
        assert error > 0
        assert error < 0.5

    def test_bootstrap_error_variance(self):
        """Test bootstrap error for variance statistic."""
        np.random.seed(42)
        sigma = 2.0
        data = np.random.normal(0, sigma, 500)

        var, error = bootstrap_error(data, np.var, n_samples=1000, seed=42)

        # Variance should be close to sigma^2 = 4
        assert var == pytest.approx(sigma**2, rel=0.2)

        # Error should be positive
        assert error > 0

    def test_bootstrap_error_custom_statistic(self):
        """Test bootstrap error with custom statistic."""
        np.random.seed(42)
        data = np.random.exponential(2.0, 200)

        # Custom statistic: interquartile range
        def iqr(x):
            return np.percentile(x, 75) - np.percentile(x, 25)

        value, error = bootstrap_error(data, iqr, n_samples=1000, seed=42)

        assert value > 0
        assert error > 0

    def test_bootstrap_error_constant_data(self):
        """Test bootstrap error for constant data is small."""
        data = np.ones(100) * 5.0

        mean, error = bootstrap_error(data, np.mean, seed=42)

        assert mean == 5.0
        assert error == 0.0

    def test_bootstrap_error_single_point(self):
        """Test bootstrap error for single data point."""
        data = np.array([3.14])

        mean, error = bootstrap_error(data, np.mean)

        assert mean == 3.14
        assert error == 0.0

    def test_bootstrap_error_empty_raises(self):
        """Test bootstrap error raises for empty data."""
        data = np.array([])

        with pytest.raises(ValueError, match="empty"):
            bootstrap_error(data, np.mean)

    def test_bootstrap_error_more_samples_reduces_variance(self):
        """Test that more bootstrap samples reduces error variance."""
        np.random.seed(42)
        data = np.random.randn(100)

        # Compute multiple error estimates with few samples
        errors_few = []
        for i in range(10):
            _, error = bootstrap_error(data, np.mean, n_samples=50, seed=i)
            errors_few.append(error)

        # Compute multiple error estimates with many samples
        errors_many = []
        for i in range(10):
            _, error = bootstrap_error(data, np.mean, n_samples=2000, seed=i)
            errors_many.append(error)

        # Variance of error estimates should be smaller with more samples
        assert np.std(errors_many) < np.std(errors_few)

    def test_bootstrap_error_scales_with_sample_size(self):
        """Test that error decreases with more data (as 1/sqrt(n))."""
        np.random.seed(42)
        sigma = 1.0

        # Small sample
        data_small = np.random.normal(0, sigma, 100)
        _, error_small = bootstrap_error(data_small, np.mean, n_samples=2000, seed=42)

        # Large sample
        data_large = np.random.normal(0, sigma, 1000)
        _, error_large = bootstrap_error(data_large, np.mean, n_samples=2000, seed=42)

        # Error should scale approximately as 1/sqrt(n)
        # error_large / error_small ≈ sqrt(100/1000) = sqrt(0.1) ≈ 0.316
        ratio = error_large / error_small
        expected_ratio = np.sqrt(100 / 1000)

        assert ratio == pytest.approx(expected_ratio, rel=0.3)


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap_confidence_interval function."""

    def test_confidence_interval_contains_true_value(self):
        """Test that CI often contains true value."""
        np.random.seed(42)
        true_mean = 5.0
        n_trials = 50
        contains_count = 0

        for i in range(n_trials):
            data = np.random.normal(true_mean, 1, 100)
            _, lower, upper = bootstrap_confidence_interval(
                data, confidence=0.95, n_samples=500, seed=i
            )
            if lower <= true_mean <= upper:
                contains_count += 1

        # 95% CI should contain true value ~95% of time
        # Allow some margin for finite samples
        coverage = contains_count / n_trials
        assert coverage > 0.80  # Conservative check

    def test_confidence_interval_ordering(self):
        """Test that lower <= estimate <= upper."""
        np.random.seed(42)
        data = np.random.randn(100)

        estimate, lower, upper = bootstrap_confidence_interval(data, seed=42)

        assert lower <= estimate <= upper

    def test_confidence_interval_width_increases(self):
        """Test that higher confidence gives wider interval."""
        data = np.random.randn(100)

        _, lower_68, upper_68 = bootstrap_confidence_interval(
            data, confidence=0.68, seed=42
        )
        _, lower_95, upper_95 = bootstrap_confidence_interval(
            data, confidence=0.95, seed=42
        )

        width_68 = upper_68 - lower_68
        width_95 = upper_95 - lower_95

        assert width_95 > width_68

    def test_confidence_interval_single_point(self):
        """Test CI for single data point."""
        data = np.array([2.5])

        estimate, lower, upper = bootstrap_confidence_interval(data)

        assert estimate == 2.5
        assert lower == 2.5
        assert upper == 2.5

    def test_confidence_interval_empty_raises(self):
        """Test that empty data raises error."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_confidence_interval(np.array([]))


class TestBootstrapMeanError:
    """Tests for bootstrap_mean_error convenience function."""

    def test_bootstrap_mean_error_matches_general(self):
        """Test convenience function matches general function."""
        data = np.random.randn(100)

        mean1, error1 = bootstrap_mean_error(data, n_samples=1000, seed=42)
        mean2, error2 = bootstrap_error(data, np.mean, n_samples=1000, seed=42)

        assert mean1 == mean2
        assert error1 == error2

    def test_bootstrap_mean_error_normal_distribution(self):
        """Test bootstrap mean error for normal distribution."""
        np.random.seed(42)
        n = 500
        sigma = 3.0
        data = np.random.normal(10, sigma, n)

        mean, error = bootstrap_mean_error(data, n_samples=2000, seed=42)

        # Mean should be close to 10
        assert mean == pytest.approx(10, abs=0.5)

        # Error should be close to sigma/sqrt(n)
        expected_error = sigma / np.sqrt(n)
        assert error == pytest.approx(expected_error, rel=0.2)

    def test_bootstrap_mean_error_uniform_distribution(self):
        """Test bootstrap mean error for uniform distribution."""
        np.random.seed(42)
        n = 400
        a, b = 0, 10
        data = np.random.uniform(a, b, n)

        mean, error = bootstrap_mean_error(data, n_samples=2000, seed=42)

        # Mean should be close to (a+b)/2 = 5
        assert mean == pytest.approx(5, abs=0.5)

        # Theoretical std for uniform: (b-a)/sqrt(12)
        sigma = (b - a) / np.sqrt(12)
        expected_error = sigma / np.sqrt(n)
        assert error == pytest.approx(expected_error, rel=0.2)


class TestJackknifeError:
    """Tests for jackknife_error function."""

    def test_jackknife_error_mean(self):
        """Test jackknife error for mean matches analytical formula."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        mean, error = jackknife_error(data, np.mean)

        # For the mean, jackknife error = std / sqrt(n)
        expected_error = np.std(data, ddof=1) / np.sqrt(len(data))
        assert error == pytest.approx(expected_error, rel=0.01)

    def test_jackknife_error_deterministic(self):
        """Test jackknife produces deterministic results."""
        data = np.array([1, 2, 3, 4, 5])

        mean1, error1 = jackknife_error(data, np.mean)
        mean2, error2 = jackknife_error(data, np.mean)

        assert mean1 == mean2
        assert error1 == error2

    def test_jackknife_error_variance(self):
        """Test jackknife error for variance."""
        np.random.seed(42)
        data = np.random.normal(0, 2, 50)

        var, error = jackknife_error(data, np.var)

        # Variance should be close to 4
        assert var == pytest.approx(4, rel=0.3)
        assert error > 0

    def test_jackknife_error_constant_data(self):
        """Test jackknife error for constant data."""
        data = np.ones(10) * 7.0

        mean, error = jackknife_error(data, np.mean)

        assert mean == 7.0
        assert error == 0.0

    def test_jackknife_error_single_point(self):
        """Test jackknife error for single point."""
        data = np.array([5.0])

        mean, error = jackknife_error(data, np.mean)

        assert mean == 5.0
        assert error == 0.0

    def test_jackknife_error_empty_raises(self):
        """Test jackknife raises for empty data."""
        with pytest.raises(ValueError, match="empty"):
            jackknife_error(np.array([]))

    def test_jackknife_vs_bootstrap_similar(self):
        """Test jackknife and bootstrap give similar errors."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        _, error_jk = jackknife_error(data, np.mean)
        _, error_bs = bootstrap_error(data, np.mean, n_samples=2000, seed=42)

        # Should be within 20% of each other
        assert error_jk == pytest.approx(error_bs, rel=0.2)


class TestBlockingError:
    """Tests for blocking_error function."""

    def test_blocking_error_uncorrelated(self):
        """Test blocking error matches standard error for uncorrelated data."""
        np.random.seed(42)
        n = 1000
        sigma = 2.0
        data = np.random.normal(0, sigma, n)

        mean, error = blocking_error(data)

        # For uncorrelated data, blocking error ≈ std/sqrt(n)
        expected_error = sigma / np.sqrt(n)

        # Allow some margin because blocking takes max over block sizes
        assert error == pytest.approx(expected_error, rel=0.5)

    def test_blocking_error_correlated_larger(self):
        """Test blocking error is larger for correlated data."""
        np.random.seed(42)
        n = 1000

        # Create correlated data (exponential autocorrelation)
        tau = 20.0
        alpha = np.exp(-1 / tau)
        correlated = np.zeros(n)
        correlated[0] = np.random.randn()
        for i in range(1, n):
            correlated[i] = alpha * correlated[i - 1] + np.sqrt(1 - alpha**2) * np.random.randn()

        # Uncorrelated data with same variance
        uncorrelated = np.random.randn(n) * np.std(correlated)

        _, error_corr = blocking_error(correlated)
        _, error_uncorr = blocking_error(uncorrelated)

        # Correlated data should have larger error
        assert error_corr > error_uncorr

    def test_blocking_error_mean_correct(self):
        """Test blocking error returns correct mean."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        mean, _ = blocking_error(data, min_blocks=2)

        assert mean == 5.5

    def test_blocking_error_small_sample(self):
        """Test blocking error handles small samples."""
        data = np.array([1, 2, 3, 4, 5])

        mean, error = blocking_error(data, min_blocks=10)

        # Falls back to standard error
        expected_error = np.std(data, ddof=1) / np.sqrt(len(data))
        assert error == pytest.approx(expected_error)

    def test_blocking_error_empty_raises(self):
        """Test blocking error raises for empty data."""
        with pytest.raises(ValueError, match="empty"):
            blocking_error(np.array([]))

    def test_blocking_error_positive(self):
        """Test blocking error is always positive."""
        np.random.seed(42)
        data = np.random.randn(500)

        _, error = blocking_error(data)

        assert error > 0


class TestStatisticsIntegration:
    """Integration tests for statistics functions."""

    def test_all_methods_consistent_for_normal(self):
        """Test all error methods give consistent results for normal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)

        _, error_bs = bootstrap_error(data, np.mean, n_samples=2000, seed=42)
        _, error_jk = jackknife_error(data, np.mean)
        _, error_bl = blocking_error(data)

        # All should be within factor of 2 of each other
        errors = [error_bs, error_jk, error_bl]
        assert max(errors) / min(errors) < 2.0

    def test_error_decreases_with_n(self):
        """Test all methods show error decreasing with sample size."""
        np.random.seed(42)

        data_100 = np.random.randn(100)
        data_1000 = np.random.randn(1000)

        # Bootstrap
        _, err_bs_100 = bootstrap_mean_error(data_100, seed=42)
        _, err_bs_1000 = bootstrap_mean_error(data_1000, seed=42)
        assert err_bs_1000 < err_bs_100

        # Jackknife
        _, err_jk_100 = jackknife_error(data_100)
        _, err_jk_1000 = jackknife_error(data_1000)
        assert err_jk_1000 < err_jk_100

        # Blocking
        _, err_bl_100 = blocking_error(data_100)
        _, err_bl_1000 = blocking_error(data_1000)
        assert err_bl_1000 < err_bl_100

    def test_known_distribution_exponential(self):
        """Test bootstrap error for exponential distribution."""
        np.random.seed(42)
        rate = 0.5
        n = 500
        data = np.random.exponential(1 / rate, n)

        mean, error = bootstrap_mean_error(data, n_samples=2000, seed=42)

        # Mean of exponential(rate) is 1/rate = 2
        assert mean == pytest.approx(2.0, rel=0.1)

        # Std of exponential is also 1/rate, so SE = (1/rate) / sqrt(n)
        expected_error = (1 / rate) / np.sqrt(n)
        assert error == pytest.approx(expected_error, rel=0.2)

    def test_skewed_distribution(self):
        """Test bootstrap handles skewed distributions."""
        np.random.seed(42)
        # Chi-squared with 2 degrees of freedom (exponential)
        data = np.random.chisquare(2, 200)

        mean, error = bootstrap_mean_error(data, n_samples=1000, seed=42)

        # Mean of chi-squared(k) is k = 2
        assert mean == pytest.approx(2.0, rel=0.2)
        assert error > 0

        # Median (more robust for skewed data)
        median, error_med = bootstrap_error(data, np.median, n_samples=1000, seed=42)
        assert median > 0
        assert error_med > 0
