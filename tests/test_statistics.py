"""Tests for statistical analysis functions."""

import numpy as np
import pytest

from ising_toolkit.analysis import (
    bootstrap_error,
    bootstrap_confidence_interval,
    bootstrap_mean_error,
    jackknife_error,
    blocking_error,
    autocorrelation_function,
    autocorrelation_function_fft,
    integrated_autocorrelation_time,
    effective_sample_size,
    generate_ar1,
    blocking_analysis,
    blocking_analysis_log,
    optimal_block_size,
    plot_blocking_analysis,
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


class TestGenerateAR1:
    """Tests for AR(1) process generator."""

    def test_generate_ar1_length(self):
        """Test AR(1) generates correct length."""
        data = generate_ar1(100, alpha=0.5, seed=42)
        assert len(data) == 100

    def test_generate_ar1_reproducible(self):
        """Test AR(1) is reproducible with seed."""
        data1 = generate_ar1(100, alpha=0.9, seed=42)
        data2 = generate_ar1(100, alpha=0.9, seed=42)
        np.testing.assert_array_equal(data1, data2)

    def test_generate_ar1_unit_variance(self):
        """Test AR(1) has approximately unit variance."""
        data = generate_ar1(10000, alpha=0.9, seed=42)
        assert np.var(data) == pytest.approx(1.0, rel=0.1)

    def test_generate_ar1_zero_mean(self):
        """Test AR(1) has approximately zero mean."""
        data = generate_ar1(10000, alpha=0.5, seed=42)
        assert np.mean(data) == pytest.approx(0.0, abs=0.1)

    def test_generate_ar1_invalid_alpha(self):
        """Test AR(1) raises for invalid alpha."""
        with pytest.raises(ValueError, match="alpha"):
            generate_ar1(100, alpha=1.0)

        with pytest.raises(ValueError, match="alpha"):
            generate_ar1(100, alpha=-1.0)

        with pytest.raises(ValueError, match="alpha"):
            generate_ar1(100, alpha=1.5)

    def test_generate_ar1_different_alphas(self):
        """Test higher alpha gives more correlated data."""
        data_low = generate_ar1(1000, alpha=0.1, seed=42)
        data_high = generate_ar1(1000, alpha=0.95, seed=42)

        # Compute lag-1 autocorrelation
        acf_low = np.corrcoef(data_low[:-1], data_low[1:])[0, 1]
        acf_high = np.corrcoef(data_high[:-1], data_high[1:])[0, 1]

        # Higher alpha should give higher lag-1 correlation
        assert acf_high > acf_low


class TestAutocorrelationFunction:
    """Tests for autocorrelation_function."""

    def test_acf_at_zero_is_one(self):
        """Test A(0) = 1 always."""
        data = np.random.randn(100)
        acf = autocorrelation_function(data, max_lag=10)
        assert acf[0] == 1.0

    def test_acf_length(self):
        """Test ACF has correct length."""
        data = np.random.randn(100)
        acf = autocorrelation_function(data, max_lag=20)
        assert len(acf) == 21  # 0 to 20 inclusive

    def test_acf_uncorrelated_near_zero(self):
        """Test ACF is near zero for uncorrelated data."""
        np.random.seed(42)
        data = np.random.randn(5000)
        acf = autocorrelation_function(data, max_lag=50)

        # For uncorrelated data, ACF(t>0) should be near 0
        # With 5000 samples, statistical noise is ~1/sqrt(5000) ≈ 0.014
        assert np.all(np.abs(acf[1:]) < 0.1)

    def test_acf_ar1_exponential_decay(self):
        """Test ACF of AR(1) decays exponentially."""
        tau = 10.0
        alpha = np.exp(-1 / tau)
        data = generate_ar1(50000, alpha, seed=42)

        acf = autocorrelation_function(data, max_lag=50)

        # For AR(1), A(t) = alpha^t
        # Only check early lags where signal-to-noise is good
        for t in range(1, 15):
            expected = alpha**t
            assert acf[t] == pytest.approx(expected, rel=0.15)

    def test_acf_ar1_known_values(self):
        """Test ACF matches theoretical values for AR(1)."""
        alpha = 0.8
        data = generate_ar1(20000, alpha, seed=42)
        acf = autocorrelation_function(data, max_lag=20)

        # Check specific lags
        assert acf[1] == pytest.approx(alpha, rel=0.1)  # α
        assert acf[2] == pytest.approx(alpha**2, rel=0.1)  # α²
        assert acf[5] == pytest.approx(alpha**5, rel=0.15)  # α⁵

    def test_acf_symmetric_data(self):
        """Test ACF is same for data and -data."""
        np.random.seed(42)
        data = np.random.randn(500)

        acf_pos = autocorrelation_function(data, max_lag=20)
        acf_neg = autocorrelation_function(-data, max_lag=20)

        np.testing.assert_array_almost_equal(acf_pos, acf_neg)

    def test_acf_constant_data(self):
        """Test ACF for constant data."""
        data = np.ones(100) * 5.0
        acf = autocorrelation_function(data, max_lag=10)

        # A(0) = 1, A(t>0) = 0 for constant data
        assert acf[0] == 1.0
        assert np.all(acf[1:] == 0.0)

    def test_acf_single_point(self):
        """Test ACF for single data point."""
        data = np.array([3.14])
        acf = autocorrelation_function(data)
        assert len(acf) == 1
        assert acf[0] == 1.0

    def test_acf_empty_raises(self):
        """Test ACF raises for empty data."""
        with pytest.raises(ValueError, match="empty"):
            autocorrelation_function(np.array([]))

    def test_acf_max_lag_respected(self):
        """Test max_lag parameter is respected."""
        data = np.random.randn(1000)
        acf = autocorrelation_function(data, max_lag=5)
        assert len(acf) == 6

    def test_acf_default_max_lag(self):
        """Test default max_lag is n//4."""
        data = np.random.randn(100)
        acf = autocorrelation_function(data)
        assert len(acf) == 26  # 0 to 25


class TestAutocorrelationFunctionFFT:
    """Tests for FFT-based autocorrelation function."""

    def test_acf_fft_matches_direct(self):
        """Test FFT method matches direct method."""
        np.random.seed(42)
        data = np.random.randn(500)

        acf_direct = autocorrelation_function(data, max_lag=50)
        acf_fft = autocorrelation_function_fft(data, max_lag=50)

        np.testing.assert_array_almost_equal(acf_direct, acf_fft, decimal=5)

    def test_acf_fft_ar1(self):
        """Test FFT ACF for AR(1) process."""
        alpha = 0.7
        data = generate_ar1(20000, alpha, seed=42)

        acf = autocorrelation_function_fft(data, max_lag=30)

        # Check exponential decay for early lags
        for t in range(1, 10):
            expected = alpha**t
            assert acf[t] == pytest.approx(expected, rel=0.15)

    def test_acf_fft_empty_raises(self):
        """Test FFT ACF raises for empty data."""
        with pytest.raises(ValueError, match="empty"):
            autocorrelation_function_fft(np.array([]))

    def test_acf_fft_single_point(self):
        """Test FFT ACF for single point."""
        data = np.array([1.0])
        acf = autocorrelation_function_fft(data)
        assert len(acf) == 1
        assert acf[0] == 1.0


class TestIntegratedAutocorrelationTime:
    """Tests for integrated autocorrelation time."""

    def test_tau_int_uncorrelated(self):
        """Test τ_int ≈ 0.5 for uncorrelated data."""
        np.random.seed(42)
        data = np.random.randn(10000)

        tau = integrated_autocorrelation_time(data)

        # For uncorrelated data, τ_int = 0.5
        assert tau == pytest.approx(0.5, rel=0.2)

    def test_tau_int_ar1_formula(self):
        """Test τ_int matches theoretical formula for AR(1)."""
        # For AR(1) with parameter α:
        # τ_int = (1 + α) / (2(1 - α))

        for tau_target in [5.0, 10.0, 20.0]:
            alpha = np.exp(-1 / tau_target)
            data = generate_ar1(50000, alpha, seed=42)

            tau_est = integrated_autocorrelation_time(data)

            # Theoretical τ_int for AR(1)
            tau_theory = (1 + alpha) / (2 * (1 - alpha))

            # Should match within 20%
            assert tau_est == pytest.approx(tau_theory, rel=0.25)

    def test_tau_int_increases_with_correlation(self):
        """Test τ_int increases with correlation strength."""
        taus = []
        for alpha in [0.5, 0.8, 0.95]:
            data = generate_ar1(10000, alpha, seed=42)
            tau = integrated_autocorrelation_time(data)
            taus.append(tau)

        # τ_int should increase with alpha
        assert taus[0] < taus[1] < taus[2]

    def test_tau_int_positive(self):
        """Test τ_int is always positive."""
        np.random.seed(42)
        data = np.random.randn(1000)
        tau = integrated_autocorrelation_time(data)
        assert tau > 0

    def test_tau_int_at_least_half(self):
        """Test τ_int >= 0.5."""
        np.random.seed(42)
        data = np.random.randn(1000)
        tau = integrated_autocorrelation_time(data)
        assert tau >= 0.5

    def test_tau_int_empty_raises(self):
        """Test τ_int raises for empty data."""
        with pytest.raises(ValueError, match="empty"):
            integrated_autocorrelation_time(np.array([]))

    def test_tau_int_single_point(self):
        """Test τ_int for single point."""
        tau = integrated_autocorrelation_time(np.array([1.0]))
        assert tau == 0.5

    def test_tau_int_constant_data(self):
        """Test τ_int handles constant data."""
        data = np.ones(100)
        tau = integrated_autocorrelation_time(data)
        # Constant data has zero variance, ACF is [1, 0, 0, ...]
        # So τ_int = 0.5
        assert tau == 0.5


class TestEffectiveSampleSize:
    """Tests for effective sample size."""

    def test_n_eff_uncorrelated(self):
        """Test N_eff ≈ N for uncorrelated data."""
        np.random.seed(42)
        n = 1000
        data = np.random.randn(n)

        n_eff = effective_sample_size(data)

        # For uncorrelated data, N_eff ≈ N
        assert n_eff == pytest.approx(n, rel=0.2)

    def test_n_eff_correlated_smaller(self):
        """Test N_eff < N for correlated data."""
        alpha = 0.9
        n = 5000
        data = generate_ar1(n, alpha, seed=42)

        n_eff = effective_sample_size(data)

        # N_eff should be significantly smaller than N
        assert n_eff < n / 2

    def test_n_eff_formula(self):
        """Test N_eff = N / (2τ) relationship."""
        alpha = 0.8
        n = 10000
        data = generate_ar1(n, alpha, seed=42)

        tau = integrated_autocorrelation_time(data)
        n_eff = effective_sample_size(data)

        expected = n / (2 * tau)
        assert n_eff == pytest.approx(expected, rel=0.01)

    def test_n_eff_positive(self):
        """Test N_eff is always positive."""
        data = np.random.randn(100)
        n_eff = effective_sample_size(data)
        assert n_eff > 0

    def test_n_eff_at_most_n(self):
        """Test N_eff <= N."""
        data = np.random.randn(100)
        n_eff = effective_sample_size(data)
        assert n_eff <= 100

    def test_n_eff_empty_raises(self):
        """Test N_eff raises for empty data."""
        with pytest.raises(ValueError, match="empty"):
            effective_sample_size(np.array([]))

    def test_n_eff_single_point(self):
        """Test N_eff for single point."""
        n_eff = effective_sample_size(np.array([1.0]))
        assert n_eff == 1.0

    def test_n_eff_error_estimation(self):
        """Test using N_eff for error estimation."""
        # Generate correlated data with known properties
        alpha = 0.9
        n = 5000
        data = generate_ar1(n, alpha, seed=42)

        n_eff = effective_sample_size(data)

        # Naive error (ignoring correlations)
        naive_error = np.std(data) / np.sqrt(n)

        # True error (accounting for correlations)
        true_error = np.std(data) / np.sqrt(n_eff)

        # True error should be larger
        assert true_error > naive_error


class TestAutocorrelationIntegration:
    """Integration tests for autocorrelation functions."""

    def test_blocking_vs_tau_int(self):
        """Test blocking error is consistent with τ_int."""
        alpha = 0.85
        n = 5000
        data = generate_ar1(n, alpha, seed=42)

        # Get τ_int and compute expected error
        tau = integrated_autocorrelation_time(data)
        expected_error = np.std(data) * np.sqrt(2 * tau / n)

        # Get blocking error
        _, blocking_err = blocking_error(data)

        # Should be similar (within factor of 2)
        assert blocking_err == pytest.approx(expected_error, rel=0.5)

    def test_acf_methods_consistent(self):
        """Test direct and FFT ACF give same τ_int."""
        alpha = 0.8
        data = generate_ar1(5000, alpha, seed=42)

        acf_direct = autocorrelation_function(data, max_lag=100)
        acf_fft = autocorrelation_function_fft(data, max_lag=100)

        # Integrate both
        tau_direct = 0.5 + np.sum(acf_direct[1:])
        tau_fft = 0.5 + np.sum(acf_fft[1:])

        assert tau_direct == pytest.approx(tau_fft, rel=0.01)

    def test_monte_carlo_like_data(self):
        """Test with Monte Carlo-like correlated data."""
        # Simulate energy-like observable from a Monte Carlo
        np.random.seed(42)
        n = 10000

        # Create data with slow drift (like near phase transition)
        tau_true = 50.0
        alpha = np.exp(-1 / tau_true)
        data = generate_ar1(n, alpha, seed=42)

        # Add a mean (like actual energy values)
        data = data * 10 - 500  # E ≈ -500 ± 10

        # Compute effective sample size
        n_eff = effective_sample_size(data)

        # Should be much smaller than n
        assert n_eff < n / 10

        # Error estimate should be reasonable
        error = np.std(data) / np.sqrt(n_eff)
        naive_error = np.std(data) / np.sqrt(n)

        # True error should be ~10x larger than naive
        assert error > 5 * naive_error


class TestBlockingAnalysis:
    """Tests for blocking_analysis function."""

    def test_blocking_analysis_output_shapes(self):
        """Test output arrays have correct shapes."""
        data = np.random.randn(1000)
        block_sizes, errors = blocking_analysis(data)

        assert len(block_sizes) == len(errors)
        assert len(block_sizes) > 0

    def test_blocking_analysis_block_sizes_ascending(self):
        """Test block sizes are in ascending order."""
        data = np.random.randn(500)
        block_sizes, _ = blocking_analysis(data)

        assert np.all(np.diff(block_sizes) > 0)

    def test_blocking_analysis_starts_at_one(self):
        """Test block sizes start at 1."""
        data = np.random.randn(200)
        block_sizes, _ = blocking_analysis(data)

        assert block_sizes[0] == 1

    def test_blocking_analysis_errors_positive(self):
        """Test all error estimates are positive."""
        data = np.random.randn(500)
        _, errors = blocking_analysis(data)

        assert np.all(errors > 0)

    def test_blocking_analysis_uncorrelated_flat(self):
        """Test error is roughly constant for uncorrelated data."""
        np.random.seed(42)
        data = np.random.randn(2000)

        block_sizes, errors = blocking_analysis(data, min_blocks=20)

        # For uncorrelated data, errors should be roughly constant
        # Check that variation is small
        relative_std = np.std(errors) / np.mean(errors)
        assert relative_std < 0.3

    def test_blocking_analysis_correlated_increases(self):
        """Test error increases with block size for correlated data."""
        alpha = 0.9
        data = generate_ar1(5000, alpha, seed=42)

        block_sizes, errors = blocking_analysis(data, min_blocks=10)

        # Error at large block size should be larger than at small block size
        assert errors[-1] > errors[0]

    def test_blocking_analysis_plateau_value(self):
        """Test plateau error matches theory."""
        alpha = 0.8
        n = 10000
        data = generate_ar1(n, alpha, seed=42)

        block_sizes, errors = blocking_analysis(data, min_blocks=10)

        # Theoretical τ_int for AR(1)
        tau_theory = (1 + alpha) / (2 * (1 - alpha))

        # Expected error from theory
        expected_error = np.std(data) * np.sqrt(2 * tau_theory / n)

        # Plateau error should be close
        plateau_error = errors[-1]
        assert plateau_error == pytest.approx(expected_error, rel=0.3)

    def test_blocking_analysis_max_block_size(self):
        """Test max_block_size parameter is respected."""
        data = np.random.randn(1000)
        block_sizes, _ = blocking_analysis(data, max_block_size=20)

        assert block_sizes[-1] <= 20

    def test_blocking_analysis_min_blocks(self):
        """Test min_blocks parameter is respected."""
        n = 100
        data = np.random.randn(n)
        min_blocks = 10

        block_sizes, _ = blocking_analysis(data, min_blocks=min_blocks)

        # Each block size should allow at least min_blocks blocks
        for bs in block_sizes:
            assert n // bs >= min_blocks

    def test_blocking_analysis_empty_raises(self):
        """Test empty data raises error."""
        with pytest.raises(ValueError, match="empty"):
            blocking_analysis(np.array([]))

    def test_blocking_analysis_too_few_points_raises(self):
        """Test too few data points raises error."""
        with pytest.raises(ValueError):
            blocking_analysis(np.array([1, 2, 3]), min_blocks=10)


class TestBlockingAnalysisLog:
    """Tests for blocking_analysis_log function."""

    def test_blocking_analysis_log_output(self):
        """Test log-spaced blocking analysis returns valid output."""
        data = np.random.randn(1000)
        block_sizes, errors = blocking_analysis_log(data, n_points=20)

        assert len(block_sizes) == len(errors)
        assert len(block_sizes) > 0
        assert np.all(errors > 0)

    def test_blocking_analysis_log_spacing(self):
        """Test block sizes are roughly logarithmically spaced."""
        data = np.random.randn(10000)
        block_sizes, _ = blocking_analysis_log(data, n_points=30)

        # Log of block sizes should be roughly linear
        log_sizes = np.log(block_sizes)
        diffs = np.diff(log_sizes)

        # Differences should be roughly equal (within factor of 3)
        if len(diffs) > 2:
            assert np.max(diffs) / np.min(diffs) < 5

    def test_blocking_analysis_log_matches_linear(self):
        """Test log and linear methods give similar plateau values."""
        alpha = 0.85
        data = generate_ar1(5000, alpha, seed=42)

        _, errors_lin = blocking_analysis(data)
        _, errors_log = blocking_analysis_log(data)

        # Plateau values should be similar
        assert errors_lin[-1] == pytest.approx(errors_log[-1], rel=0.2)

    def test_blocking_analysis_log_empty_raises(self):
        """Test empty data raises error."""
        with pytest.raises(ValueError, match="empty"):
            blocking_analysis_log(np.array([]))


class TestOptimalBlockSize:
    """Tests for optimal_block_size function."""

    def test_optimal_block_size_uncorrelated(self):
        """Test optimal block size is small for uncorrelated data."""
        np.random.seed(42)
        data = np.random.randn(1000)

        opt = optimal_block_size(data)

        # For uncorrelated data, any block size is fine
        # Should return a small value
        assert opt >= 1
        assert opt < 20

    def test_optimal_block_size_correlated(self):
        """Test optimal block size increases with correlation."""
        opt_low = optimal_block_size(generate_ar1(5000, 0.5, seed=42))
        opt_high = optimal_block_size(generate_ar1(5000, 0.95, seed=42))

        # Higher correlation should give larger optimal block size
        assert opt_high >= opt_low

    def test_optimal_block_size_positive(self):
        """Test optimal block size is always positive."""
        data = np.random.randn(200)
        opt = optimal_block_size(data)
        assert opt >= 1

    def test_optimal_block_size_integer(self):
        """Test optimal block size is an integer."""
        data = np.random.randn(500)
        opt = optimal_block_size(data)
        assert isinstance(opt, int)

    def test_optimal_block_size_empty_raises(self):
        """Test empty data raises error."""
        with pytest.raises(ValueError, match="empty"):
            optimal_block_size(np.array([]))

    def test_optimal_block_size_small_data(self):
        """Test small data returns 1."""
        data = np.array([1, 2, 3, 4, 5])
        opt = optimal_block_size(data, min_blocks=10)
        assert opt == 1

    def test_optimal_block_size_vs_tau(self):
        """Test optimal block size is related to autocorrelation time."""
        alpha = 0.9
        data = generate_ar1(10000, alpha, seed=42)

        opt = optimal_block_size(data)
        tau = integrated_autocorrelation_time(data)

        # Optimal block size should be roughly 2-3 times τ
        # Allow wide margin due to heuristic nature
        assert opt > tau / 2
        assert opt < tau * 10


class TestPlotBlockingAnalysis:
    """Tests for plot_blocking_analysis function."""

    def test_plot_blocking_analysis_returns_ax_and_info(self):
        """Test function returns axes and info dict."""
        pytest.importorskip("matplotlib")

        data = generate_ar1(1000, 0.8, seed=42)
        ax, info = plot_blocking_analysis(data)

        assert ax is not None
        assert isinstance(info, dict)

    def test_plot_blocking_analysis_info_keys(self):
        """Test info dict has expected keys."""
        pytest.importorskip("matplotlib")

        data = np.random.randn(500)
        _, info = plot_blocking_analysis(data)

        expected_keys = {
            'block_sizes',
            'errors',
            'optimal_block_size',
            'plateau_error',
            'naive_error',
            'tau_int',
        }
        assert set(info.keys()) == expected_keys

    def test_plot_blocking_analysis_info_values(self):
        """Test info dict values are reasonable."""
        pytest.importorskip("matplotlib")

        data = np.random.randn(500)
        _, info = plot_blocking_analysis(data)

        assert len(info['block_sizes']) == len(info['errors'])
        assert info['optimal_block_size'] >= 1
        assert info['plateau_error'] > 0
        assert info['naive_error'] > 0
        assert info['tau_int'] >= 0.5

    def test_plot_blocking_analysis_linear_scale(self):
        """Test linear scale option works."""
        pytest.importorskip("matplotlib")

        data = np.random.randn(500)
        ax, info = plot_blocking_analysis(data, log_scale=False)

        assert ax is not None
        assert len(info['block_sizes']) > 0

    def test_plot_blocking_analysis_custom_ax(self):
        """Test plotting on custom axes."""
        plt = pytest.importorskip("matplotlib.pyplot")

        data = np.random.randn(500)
        fig, custom_ax = plt.subplots()

        returned_ax, _ = plot_blocking_analysis(data, ax=custom_ax)

        assert returned_ax is custom_ax
        plt.close(fig)

    def test_plot_blocking_analysis_no_optional_features(self):
        """Test disabling optional features."""
        pytest.importorskip("matplotlib")

        data = np.random.randn(500)
        ax, info = plot_blocking_analysis(
            data,
            show_optimal=False,
            show_tau_estimate=False
        )

        assert ax is not None
        assert info is not None


class TestBlockingAnalysisIntegration:
    """Integration tests for blocking analysis."""

    def test_blocking_vs_autocorrelation(self):
        """Test blocking and autocorrelation give consistent errors."""
        alpha = 0.85
        n = 10000  # More data for better statistics
        data = generate_ar1(n, alpha, seed=42)

        # Error from blocking
        block_sizes, errors = blocking_analysis(data)
        blocking_err = errors[-1]

        # Error from autocorrelation time
        tau = integrated_autocorrelation_time(data)
        tau_err = np.std(data) * np.sqrt(2 * tau / n)

        # Should agree within factor of 2 (both methods have their limitations)
        assert blocking_err == pytest.approx(tau_err, rel=0.5)

    def test_blocking_vs_bootstrap_correlated(self):
        """Test blocking vs bootstrap for correlated data.

        Note: Standard bootstrap underestimates error for correlated data.
        """
        alpha = 0.9
        data = generate_ar1(2000, alpha, seed=42)

        # Blocking error (accounts for correlations)
        _, errors = blocking_analysis(data)
        blocking_err = errors[-1]

        # Bootstrap error (ignores correlations)
        _, bootstrap_err = bootstrap_error(data, np.mean, n_samples=1000, seed=42)

        # Blocking error should be larger for correlated data
        assert blocking_err > bootstrap_err

    def test_optimal_block_gives_good_error(self):
        """Test using optimal block size gives good error estimate."""
        alpha = 0.8
        n = 5000
        data = generate_ar1(n, alpha, seed=42)

        # Get optimal block size
        opt = optimal_block_size(data)

        # Compute error at optimal block size
        block_sizes, errors = blocking_analysis(data, max_block_size=opt + 10)
        opt_idx = np.argmin(np.abs(block_sizes - opt))
        opt_error = errors[opt_idx]

        # Compare to tau-based error
        tau = integrated_autocorrelation_time(data)
        tau_error = np.std(data) * np.sqrt(2 * tau / n)

        # Should be reasonably close
        assert opt_error == pytest.approx(tau_error, rel=0.4)

    def test_full_workflow(self):
        """Test complete workflow: generate, analyze, estimate error."""
        # Generate correlated data (simulating Monte Carlo)
        np.random.seed(42)
        n = 10000  # More data for better statistics
        tau_true = 15.0  # Moderate correlation
        alpha = np.exp(-1 / tau_true)

        data = generate_ar1(n, alpha, seed=42)
        data = data * 10 - 100  # Scale and shift

        # Method 1: Blocking analysis
        block_sizes, errors = blocking_analysis(data)
        blocking_err = errors[-1]

        # Method 2: Autocorrelation time
        tau = integrated_autocorrelation_time(data)
        n_eff = effective_sample_size(data)
        acf_err = np.std(data) / np.sqrt(n_eff)

        # Method 3: blocking_error convenience function
        _, convenience_err = blocking_error(data)

        # All methods should give results within factor of 2
        # (different methods have different biases and variances)
        assert blocking_err == pytest.approx(acf_err, rel=0.5)
        assert convenience_err == pytest.approx(acf_err, rel=0.5)

        # Check estimated tau is reasonable
        tau_theory = (1 + alpha) / (2 * (1 - alpha))
        assert tau == pytest.approx(tau_theory, rel=0.3)
