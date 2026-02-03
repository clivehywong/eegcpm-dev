"""Tests for extended connectivity measures (icoh, dwPLI, AEC, PDC, DTF)."""

from pathlib import Path

import numpy as np
import pytest

from eegcpm.modules.connectivity import ConnectivityModule


@pytest.fixture
def synthetic_roi_data():
    """Create synthetic ROI time series for testing."""
    np.random.seed(42)
    n_rois = 5
    n_times = 500
    sfreq = 250.0

    # Create oscillatory signals with some coupling
    t = np.arange(n_times) / sfreq
    freqs = [8, 10, 12, 15, 20]  # Different frequencies for each ROI

    data = np.zeros((n_rois, n_times))
    for i in range(n_rois):
        # Base oscillation
        data[i] = np.sin(2 * np.pi * freqs[i] * t)
        # Add some noise
        data[i] += 0.1 * np.random.randn(n_times)

        # Add coupling to first ROI (for testing directed connectivity)
        if i > 0:
            data[i] += 0.3 * data[0]

    roi_data = {
        "condition1": data,
        "condition1_times": t,
        "roi_names": [f"ROI_{i+1}" for i in range(n_rois)],
    }

    return roi_data, sfreq


def test_dwpli_computation(tmp_path, synthetic_roi_data):
    """Test debiased wPLI computation."""
    roi_data, sfreq = synthetic_roi_data

    config = {
        "methods": ["dwpli"],
        "frequency_bands": {"alpha": (8, 13)},
        "time_windows": [{"name": "full", "tmin": 0.0, "tmax": 2.0}],
    }

    module = ConnectivityModule(config, tmp_path)
    result = module.process(roi_data, sfreq=sfreq)

    assert result.success
    assert "connectivity" in result.outputs
    assert "condition1_full_dwpli_alpha" in result.outputs["connectivity"]

    # Check matrix properties
    matrix = result.outputs["connectivity"]["condition1_full_dwpli_alpha"]
    assert matrix.shape == (5, 5)
    assert np.allclose(matrix, matrix.T)  # Symmetric
    assert not np.any(np.isnan(matrix))   # No NaNs
    assert np.all(matrix >= -1) and np.all(matrix <= 1)  # Valid range


def test_icoh_computation(tmp_path, synthetic_roi_data):
    """Test imaginary coherence computation."""
    roi_data, sfreq = synthetic_roi_data

    config = {
        "methods": ["icoh"],
        "frequency_bands": {"alpha": (8, 13)},
        "time_windows": [{"name": "full", "tmin": 0.0, "tmax": 2.0}],
    }

    module = ConnectivityModule(config, tmp_path)
    result = module.process(roi_data, sfreq=sfreq)

    assert result.success
    assert "condition1_full_icoh_alpha" in result.outputs["connectivity"]

    # Check matrix
    matrix = result.outputs["connectivity"]["condition1_full_icoh_alpha"]
    assert matrix.shape == (5, 5)
    assert np.allclose(matrix, matrix.T)  # Symmetric
    assert np.all(matrix >= 0) and np.all(matrix <= 1)  # Valid range


def test_aec_computation(tmp_path, synthetic_roi_data):
    """Test amplitude envelope correlation."""
    roi_data, sfreq = synthetic_roi_data

    config = {
        "methods": ["aec"],
        "frequency_bands": {"alpha": (8, 13)},
        "time_windows": [{"name": "full", "tmin": 0.0, "tmax": 2.0}],
    }

    module = ConnectivityModule(config, tmp_path)
    result = module.process(roi_data, sfreq=sfreq)

    assert result.success
    assert "condition1_full_aec_alpha" in result.outputs["connectivity"]

    matrix = result.outputs["connectivity"]["condition1_full_aec_alpha"]
    assert matrix.shape == (5, 5)
    assert np.allclose(matrix, matrix.T)  # Symmetric
    assert np.allclose(np.diag(matrix), 1, atol=1e-10)  # Diagonal ≈ 1


def test_aec_orthogonalized(tmp_path, synthetic_roi_data):
    """Test orthogonalized AEC."""
    roi_data, sfreq = synthetic_roi_data

    config = {
        "methods": ["aec_orth"],
        "frequency_bands": {"alpha": (8, 13)},
        "time_windows": [{"name": "full", "tmin": 0.0, "tmax": 2.0}],
    }

    module = ConnectivityModule(config, tmp_path)
    result = module.process(roi_data, sfreq=sfreq)

    assert result.success
    assert "condition1_full_aec_orth_alpha" in result.outputs["connectivity"]

    matrix = result.outputs["connectivity"]["condition1_full_aec_orth_alpha"]
    assert matrix.shape == (5, 5)
    assert np.allclose(matrix, matrix.T)  # Symmetric


def test_pdc_computation(tmp_path, synthetic_roi_data):
    """Test Partial Directed Coherence."""
    roi_data, sfreq = synthetic_roi_data

    config = {
        "methods": ["pdc"],
        "frequency_bands": {"alpha": (8, 13)},
        "time_windows": [{"name": "full", "tmin": 0.0, "tmax": 2.0}],
        "mvar_order": 5,
    }

    module = ConnectivityModule(config, tmp_path)
    result = module.process(roi_data, sfreq=sfreq)

    assert result.success
    assert "condition1_full_pdc_alpha" in result.outputs["connectivity"]

    matrix = result.outputs["connectivity"]["condition1_full_pdc_alpha"]
    assert matrix.shape == (5, 5)
    # PDC is directional, so not necessarily symmetric
    assert not np.any(np.isnan(matrix))
    assert np.all(matrix >= 0)  # Non-negative


def test_dtf_computation(tmp_path, synthetic_roi_data):
    """Test Directed Transfer Function."""
    roi_data, sfreq = synthetic_roi_data

    config = {
        "methods": ["dtf"],
        "frequency_bands": {"alpha": (8, 13)},
        "time_windows": [{"name": "full", "tmin": 0.0, "tmax": 2.0}],
        "mvar_order": 5,
    }

    module = ConnectivityModule(config, tmp_path)
    result = module.process(roi_data, sfreq=sfreq)

    assert result.success
    assert "condition1_full_dtf_alpha" in result.outputs["connectivity"]

    matrix = result.outputs["connectivity"]["condition1_full_dtf_alpha"]
    assert matrix.shape == (5, 5)
    assert not np.any(np.isnan(matrix))
    assert np.all(matrix >= 0)


def test_multiple_methods(tmp_path, synthetic_roi_data):
    """Test computing multiple connectivity methods simultaneously."""
    roi_data, sfreq = synthetic_roi_data

    config = {
        "methods": ["plv", "dwpli", "icoh", "aec"],
        "frequency_bands": {"alpha": (8, 13), "beta": (13, 30)},
        "time_windows": [{"name": "full", "tmin": 0.0, "tmax": 2.0}],
    }

    module = ConnectivityModule(config, tmp_path)
    result = module.process(roi_data, sfreq=sfreq)

    assert result.success

    # Should have 4 methods × 2 bands = 8 matrices
    expected_keys = [
        "condition1_full_plv_alpha", "condition1_full_plv_beta",
        "condition1_full_dwpli_alpha", "condition1_full_dwpli_beta",
        "condition1_full_icoh_alpha", "condition1_full_icoh_beta",
        "condition1_full_aec_alpha", "condition1_full_aec_beta",
    ]

    for key in expected_keys:
        assert key in result.outputs["connectivity"]


def test_flexible_time_windows(tmp_path, synthetic_roi_data):
    """Test flexible time window configuration."""
    roi_data, sfreq = synthetic_roi_data

    # Adjust time windows to match actual data (0 to 2 seconds)
    config = {
        "methods": ["plv"],
        "frequency_bands": {"alpha": (8, 13)},
        "time_windows": [
            {"name": "early", "tmin": 0.0, "tmax": 0.5},
            {"name": "middle", "tmin": 0.5, "tmax": 1.0},
            {"name": "late", "tmin": 1.0, "tmax": 1.5},
        ],
    }

    module = ConnectivityModule(config, tmp_path)
    result = module.process(roi_data, sfreq=sfreq)

    assert result.success

    # Should have 3 time windows
    expected_keys = [
        "condition1_early_plv_alpha",
        "condition1_middle_plv_alpha",
        "condition1_late_plv_alpha",
    ]

    for key in expected_keys:
        assert key in result.outputs["connectivity"]


def test_supported_methods_list(tmp_path):
    """Test that SUPPORTED_METHODS includes all new methods."""
    config = {
        "methods": ["plv"],
        "frequency_bands": {"alpha": (8, 13)},
    }

    module = ConnectivityModule(config, tmp_path)

    expected_methods = [
        "correlation", "spearman", "partial_correlation",
        "plv", "pli", "wpli", "dwpli",
        "coherence", "icoh",
        "aec", "aec_orth",
        "pdc", "dtf",
    ]

    for method in expected_methods:
        assert method in module.SUPPORTED_METHODS


def test_mvar_fitting(tmp_path, synthetic_roi_data):
    """Test MVAR model fitting."""
    roi_data, sfreq = synthetic_roi_data
    data = roi_data["condition1"]

    config = {"methods": ["pdc"], "mvar_order": 5}
    module = ConnectivityModule(config, tmp_path)

    # Test MVAR fitting directly
    A, sigma = module._fit_mvar(data, order=5)

    assert A.shape == (5, 5, 5)  # (n_rois, n_rois, order)
    assert sigma.shape == (5, 5)  # (n_rois, n_rois)
    assert not np.any(np.isnan(A))
    assert not np.any(np.isnan(sigma))
