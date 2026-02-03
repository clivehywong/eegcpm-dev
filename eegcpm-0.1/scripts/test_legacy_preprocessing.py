"""Tests for preprocessing module."""

import pytest
import tempfile
from pathlib import Path

import mne
import numpy as np

from eegcpm.modules.preprocessing import PreprocessingModule
from eegcpm.pipeline.base import ModuleResult


def make_correlated_eeg(n_channels, n_samples, sfreq):
    """Generate spatially correlated EEG-like data for testing.

    Creates data with a common signal component that ensures channels are
    correlated, preventing bad channel detection from marking all channels bad.
    """
    t = np.arange(n_samples) / sfreq
    common_signal = 15e-6 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha

    data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        # Each channel gets common signal (with slight variation) + noise
        weight = 0.5 + 0.5 * (1 - i / n_channels)  # Decreasing weights
        data[i] = common_signal * weight + np.random.randn(n_samples) * 5e-6

    return data


class TestPreprocessingModule:
    """Test PreprocessingModule class."""

    @pytest.fixture
    def sample_raw(self):
        """Create a sample raw object for testing."""
        # Create realistic EEG-like data with 19 channels (10-20 system)
        ch_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6",
            "O1", "O2"
        ]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=256.0,
            ch_types="eeg",
        )
        # 60 seconds of data with realistic EEG-like amplitude
        np.random.seed(42)
        n_samples = 256 * 60
        n_ch = len(ch_names)

        # Generate spatially correlated data (like real EEG)
        # Use a common signal + channel-specific noise to create correlation
        t = np.arange(n_samples) / 256.0

        # Common signals (shared across channels with varying weights)
        common_alpha = 15e-6 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        common_drift = 10e-6 * np.sin(2 * np.pi * 0.1 * t)  # Low-freq drift

        data = np.zeros((n_ch, n_samples))
        for i in range(n_ch):
            # Each channel gets common signals + unique noise
            # This creates inter-channel correlation like real EEG
            data[i] = (
                common_alpha * (0.5 + 0.5 * np.random.rand())  # Weighted alpha
                + common_drift * (0.8 + 0.4 * np.random.rand())  # Weighted drift
                + np.random.randn(n_samples) * 10e-6  # Channel-specific noise
            )

        raw = mne.io.RawArray(data, info)
        # Set montage for spatial filtering
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)
        return raw

    @pytest.fixture
    def sample_raw_with_artifacts(self):
        """Create raw with EOG-like artifacts.

        Uses 15 EEG channels (>= 10 for ICA feasibility) plus EOG.
        """
        # 15 EEG channels + 1 EOG = 16 total
        ch_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2",
            "EOG"
        ]
        ch_types = ["eeg"] * 15 + ["eog"]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=256.0,
            ch_types=ch_types,
        )
        np.random.seed(42)
        n_samples = 256 * 60
        n_eeg = 15

        # Generate spatially correlated EEG-like data for ICA
        data = make_correlated_eeg(n_eeg, n_samples, 256.0)
        # Add EOG channel (will be correlated with frontal)
        eog_data = data[0] * 1.5 + np.random.randn(n_samples) * 5e-6
        data = np.vstack([data, eog_data[np.newaxis, :]])

        # Add blink artifacts to frontal channels and EOG
        blink_times = [5, 15, 25, 35, 45, 55]  # seconds
        for t in blink_times:
            idx = int(t * 256)
            blink = 100e-6 * np.exp(-np.arange(128) / 30)  # ~0.5s blink
            for ch in [0, 1, 15]:  # Fp1, Fp2, EOG
                data[ch, idx:idx+128] += blink

        raw = mne.io.RawArray(data, info)
        return raw

    def test_module_initialization(self):
        """Test module initialization with config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "l_freq": 1.0,
                "h_freq": 40.0,
                "ica_method": "infomax",
            }
            module = PreprocessingModule(config, Path(tmpdir))

            assert module.l_freq == 1.0
            assert module.h_freq == 40.0
            assert module.ica_method == "infomax"
            assert module.name == "preprocessing"

    def test_default_config(self):
        """Test module with default configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = PreprocessingModule({}, Path(tmpdir))

            assert module.l_freq == 0.5
            assert module.h_freq == 40.0
            assert module.ica_method == "infomax"
            assert module.use_asr is False

    def test_process_basic(self, sample_raw):
        """Test basic preprocessing pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "l_freq": 0.5,
                "h_freq": 40.0,
                "ica_method": "infomax",
                "ica_n_components": 10,
            }
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(sample_raw)

            assert isinstance(result, ModuleResult)
            assert result.success is True
            assert result.module_name == "preprocessing"
            assert result.execution_time_seconds > 0
            assert "data" in result.outputs
            assert "ica" in result.outputs

    def test_process_output_raw(self, sample_raw):
        """Test that output is valid Raw object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"ica_n_components": 10}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(sample_raw)

            output_raw = result.outputs["data"]
            assert isinstance(output_raw, mne.io.BaseRaw)
            assert len(output_raw.ch_names) == len(sample_raw.ch_names)
            assert output_raw.info["sfreq"] == sample_raw.info["sfreq"]

    def test_process_output_ica(self, sample_raw):
        """Test that output includes ICA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"ica_n_components": 10}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(sample_raw)

            ica = result.outputs["ica"]
            assert isinstance(ica, mne.preprocessing.ICA)
            assert ica.n_components_ == 10

    def test_filtering_applied(self, sample_raw):
        """Test that filtering is applied correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "l_freq": 1.0,
                "h_freq": 30.0,
                "ica_n_components": 5,
            }
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(sample_raw)

            output_raw = result.outputs["data"]
            # Check that high-pass removed very low frequencies
            # This is a basic check - the data should be different after filtering
            assert not np.allclose(
                sample_raw.get_data()[:, :1000],
                output_raw.get_data()[:, :1000]
            )

    def test_notch_filter(self, sample_raw):
        """Test notch filter application."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "l_freq": 0.5,
                "h_freq": 100.0,  # High enough to include line noise
                "notch_freq": 50.0,
                "ica_n_components": 5,
            }
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(sample_raw)

            assert result.success is True

    def test_output_files_created(self, sample_raw):
        """Test that output files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"ica_n_components": 5}
            module = PreprocessingModule(config, Path(tmpdir))

            # Create mock subject
            class MockSubject:
                id = "sub-001"

            result = module.process(sample_raw, subject=MockSubject())

            assert len(result.output_files) == 2  # Raw and ICA
            assert all(f.exists() for f in result.output_files)

            # Check file names
            file_names = [f.name for f in result.output_files]
            assert any("preprocessed_raw.fif" in n for n in file_names)
            assert any("ica.fif" in n for n in file_names)

    def test_metadata_returned(self, sample_raw):
        """Test that metadata is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"ica_n_components": 5}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(sample_raw)

            assert result.metadata is not None
            assert "n_channels" in result.metadata
            assert "sfreq" in result.metadata
            assert "duration_s" in result.metadata
            # ICA metadata is nested under 'ica' key
            assert "ica" in result.metadata
            assert "n_components" in result.metadata["ica"]
            assert result.metadata["n_channels"] == 19

    def test_eog_artifact_detection(self, sample_raw_with_artifacts):
        """Test that EOG artifacts can be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"ica_n_components": 8}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(sample_raw_with_artifacts)

            assert result.success is True
            # EOG channel should help identify blink components
            ica = result.outputs["ica"]
            assert ica.n_components_ == 8

    def test_get_output_spec(self):
        """Test output specification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = PreprocessingModule({}, Path(tmpdir))
            spec = module.get_output_spec()

            assert "data" in spec
            assert "ica" in spec

    def test_process_preserves_channel_info(self, sample_raw):
        """Test that channel information is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"ica_n_components": 5}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(sample_raw)

            output_raw = result.outputs["data"]
            assert output_raw.ch_names == sample_raw.ch_names


class TestPreprocessingEdgeCases:
    """Test edge cases and error handling."""

    def test_minimal_channels(self):
        """Test with minimal number of channels (ICA skipped due to low rank)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002", "EEG 003"],
                sfreq=256.0,
                ch_types="eeg",
            )
            np.random.seed(42)
            # Use correlated data to avoid bad channel detection marking all channels bad
            n_samples = 256 * 30
            t = np.arange(n_samples) / 256.0
            common_signal = 15e-6 * np.sin(2 * np.pi * 10 * t)
            data = np.array([
                common_signal + np.random.randn(n_samples) * 5e-6,
                common_signal * 0.9 + np.random.randn(n_samples) * 5e-6,
                common_signal * 0.8 + np.random.randn(n_samples) * 5e-6,
            ])
            raw = mne.io.RawArray(data, info)

            # With only 3 channels, ICA will be skipped due to low rank (rank < 10)
            config = {"ica_n_components": 2}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(raw)

            assert result.success is True
            # ICA is None because rank is too low
            assert result.outputs["ica"] is None
            assert result.metadata["ica"]["success"] is False
            assert result.metadata["ica"]["reason"] == "insufficient_rank"

    def test_short_recording(self):
        """Test with short recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002", "EEG 003", "EEG 004"],
                sfreq=256.0,
                ch_types="eeg",
            )
            np.random.seed(42)
            # Only 5 seconds, use correlated data
            n_samples = 256 * 5
            t = np.arange(n_samples) / 256.0
            common_signal = 15e-6 * np.sin(2 * np.pi * 10 * t)
            data = np.array([
                common_signal + np.random.randn(n_samples) * 5e-6,
                common_signal * 0.9 + np.random.randn(n_samples) * 5e-6,
                common_signal * 0.8 + np.random.randn(n_samples) * 5e-6,
                common_signal * 0.7 + np.random.randn(n_samples) * 5e-6,
            ])
            raw = mne.io.RawArray(data, info)

            config = {"ica_n_components": 3}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(raw)

            assert result.success is True

    def test_high_sampling_rate(self):
        """Test with high sampling rate data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002", "EEG 003"],
                sfreq=1000.0,  # 1kHz
                ch_types="eeg",
            )
            np.random.seed(42)
            # Use correlated data
            n_samples = 1000 * 30
            t = np.arange(n_samples) / 1000.0
            common_signal = 15e-6 * np.sin(2 * np.pi * 10 * t)
            data = np.array([
                common_signal + np.random.randn(n_samples) * 5e-6,
                common_signal * 0.9 + np.random.randn(n_samples) * 5e-6,
                common_signal * 0.8 + np.random.randn(n_samples) * 5e-6,
            ])
            raw = mne.io.RawArray(data, info)

            config = {"ica_n_components": 2}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(raw)

            assert result.success is True

    def test_without_subject_info(self):
        """Test processing without subject info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002", "EEG 003"],
                sfreq=256.0,
                ch_types="eeg",
            )
            np.random.seed(42)
            # Use correlated data
            n_samples = 256 * 30
            t = np.arange(n_samples) / 256.0
            common_signal = 15e-6 * np.sin(2 * np.pi * 10 * t)
            data = np.array([
                common_signal + np.random.randn(n_samples) * 5e-6,
                common_signal * 0.9 + np.random.randn(n_samples) * 5e-6,
                common_signal * 0.8 + np.random.randn(n_samples) * 5e-6,
            ])
            raw = mne.io.RawArray(data, info)

            config = {"ica_n_components": 2}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(raw)  # No subject provided

            assert result.success is True
            # Should use "unknown" as subject ID
            file_names = [f.name for f in result.output_files]
            assert any("unknown" in n for n in file_names)


class TestPreprocessingICA:
    """Test ICA-specific functionality."""

    def test_ica_infomax(self):
        """Test ICA with infomax method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Need at least 10 channels for ICA feasibility check
            info = mne.create_info(
                ch_names=[f"EEG {i:03d}" for i in range(15)],
                sfreq=256.0,
                ch_types="eeg",
            )
            np.random.seed(42)
            data = make_correlated_eeg(15, 256 * 30, 256.0)
            raw = mne.io.RawArray(data, info)

            config = {"ica_method": "infomax", "ica_n_components": 10}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(raw)

            assert result.success is True
            assert result.outputs["ica"].method == "infomax"

    def test_ica_fastica(self):
        """Test ICA with fastica method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Need at least 10 channels for ICA feasibility check
            info = mne.create_info(
                ch_names=[f"EEG {i:03d}" for i in range(15)],
                sfreq=256.0,
                ch_types="eeg",
            )
            np.random.seed(42)
            data = make_correlated_eeg(15, 256 * 30, 256.0)
            raw = mne.io.RawArray(data, info)

            config = {"ica_method": "fastica", "ica_n_components": 10}
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(raw)

            assert result.success is True
            assert result.outputs["ica"].method == "fastica"

    def test_ica_auto_components(self):
        """Test ICA with automatic component detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=[f"EEG {i:03d}" for i in range(30)],
                sfreq=256.0,
                ch_types="eeg",
            )
            np.random.seed(42)
            data = make_correlated_eeg(30, 256 * 60, 256.0)
            raw = mne.io.RawArray(data, info)

            config = {}  # No ica_n_components specified
            module = PreprocessingModule(config, Path(tmpdir))
            result = module.process(raw)

            assert result.success is True
            # Should automatically use rank-based recommendation or min(20, n_channels-1)
            assert result.outputs["ica"].n_components_ <= 20


class TestPreprocessingValidation:
    """Test input validation."""

    def test_module_name(self):
        """Test module name property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = PreprocessingModule({}, Path(tmpdir))
            assert module.name == "preprocessing"

    def test_module_version(self):
        """Test module version property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = PreprocessingModule({}, Path(tmpdir))
            assert module.version == "0.2.0"

    def test_module_description(self):
        """Test module description property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = PreprocessingModule({}, Path(tmpdir))
            assert "preprocessing" in module.description.lower()


class TestZaplineStep:
    """Test ZaplineStep for line noise removal."""

    @pytest.fixture
    def sample_raw_with_line_noise(self):
        """Create raw data with simulated 60 Hz line noise."""
        ch_names = [
            "Fp1", "Fp2", "F3", "Fz", "F4",
            "C3", "Cz", "C4", "P3", "Pz", "P4",
            "O1", "O2"
        ]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=256.0,
            ch_types="eeg",
        )
        np.random.seed(42)
        n_samples = 256 * 30  # 30 seconds
        n_ch = len(ch_names)
        t = np.arange(n_samples) / 256.0

        # Generate EEG-like data
        data = np.zeros((n_ch, n_samples))
        for i in range(n_ch):
            # Brain signals (alpha, beta)
            data[i] = (
                10e-6 * np.sin(2 * np.pi * 10 * t)  # Alpha
                + 5e-6 * np.sin(2 * np.pi * 20 * t)  # Beta
                + np.random.randn(n_samples) * 3e-6  # Noise
            )
            # Add 60 Hz line noise (varying amplitude per channel)
            line_amplitude = 8e-6 * (0.5 + np.random.rand())
            data[i] += line_amplitude * np.sin(2 * np.pi * 60 * t)
            # Add 120 Hz harmonic
            data[i] += line_amplitude * 0.3 * np.sin(2 * np.pi * 120 * t)

        raw = mne.io.RawArray(data, info)
        return raw

    def test_zapline_initialization(self):
        """Test ZaplineStep initialization with defaults."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep

        step = ZaplineStep()
        assert step.fline == 60.0
        assert step.nremove is None
        assert step.adaptive is True
        assert step.enabled is True

    def test_zapline_initialization_custom(self):
        """Test ZaplineStep with custom parameters."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep

        step = ZaplineStep(fline=50, nremove=2, adaptive=False)
        assert step.fline == 50
        assert step.nremove == 2
        assert step.adaptive is False

    def test_zapline_get_config(self):
        """Test ZaplineStep configuration export."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep

        step = ZaplineStep(fline=60, nremove=1, adaptive=True)
        config = step.get_config()

        assert config['fline'] == 60
        assert config['nremove'] == 1
        assert config['adaptive'] is True
        assert 'nfft' in config

    def test_zapline_validate_success(self, sample_raw_with_line_noise):
        """Test ZaplineStep validation passes for valid data."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep

        step = ZaplineStep(fline=60)
        # Should not raise
        step.validate(sample_raw_with_line_noise)

    def test_zapline_validate_nyquist_error(self):
        """Test ZaplineStep validation fails when fline > Nyquist."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep

        # Create data with low sampling rate (100 Hz -> Nyquist = 50 Hz)
        info = mne.create_info(ch_names=["C3", "C4", "Cz"], sfreq=100.0, ch_types="eeg")
        data = np.random.randn(3, 1000) * 10e-6
        raw = mne.io.RawArray(data, info)

        step = ZaplineStep(fline=60)  # 60 Hz > 50 Hz Nyquist
        with pytest.raises(ValueError, match="Nyquist"):
            step.validate(raw)

    def test_zapline_validate_insufficient_channels(self):
        """Test ZaplineStep validation fails with < 2 channels."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep

        info = mne.create_info(ch_names=["Cz"], sfreq=256.0, ch_types="eeg")
        data = np.random.randn(1, 1000) * 10e-6
        raw = mne.io.RawArray(data, info)

        step = ZaplineStep(fline=60)
        with pytest.raises(ValueError, match="at least 2"):
            step.validate(raw)

    def test_zapline_process_reduces_line_noise(self, sample_raw_with_line_noise):
        """Test ZaplineStep actually reduces 60 Hz power."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep
        from scipy import signal

        step = ZaplineStep(fline=60, verbose=False)
        raw = sample_raw_with_line_noise.copy()

        # Measure 60 Hz power before
        data_before = raw.get_data()
        freqs, psd_before = signal.welch(data_before, fs=256, nperseg=512)
        idx_60hz = np.argmin(np.abs(freqs - 60))
        power_60hz_before = np.mean(psd_before[:, idx_60hz])

        # Process
        raw_clean, metadata = step.process(raw, {})

        # Measure 60 Hz power after
        data_after = raw_clean.get_data()
        _, psd_after = signal.welch(data_after, fs=256, nperseg=512)
        power_60hz_after = np.mean(psd_after[:, idx_60hz])

        # 60 Hz power should be reduced
        assert power_60hz_after < power_60hz_before
        assert metadata['applied'] is True
        assert metadata['fline'] == 60
        assert 'line_reduction_db' in metadata

    def test_zapline_process_metadata(self, sample_raw_with_line_noise):
        """Test ZaplineStep returns proper metadata."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep

        step = ZaplineStep(fline=60, adaptive=True)
        raw = sample_raw_with_line_noise.copy()

        _, metadata = step.process(raw, {})

        assert metadata['applied'] is True
        assert metadata['fline'] == 60
        assert metadata['adaptive'] is True
        assert 'n_components_removed' in metadata
        assert 'n_channels_processed' in metadata
        assert 'noise_reduction_db' in metadata
        assert 'line_reduction_db' in metadata

    def test_zapline_disabled(self, sample_raw_with_line_noise):
        """Test ZaplineStep when disabled."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep

        step = ZaplineStep(fline=60, enabled=False)
        raw = sample_raw_with_line_noise.copy()

        # Disabled steps should be skipped by pipeline, but process should work
        raw_out, metadata = step.process(raw, {})
        # Step still processes when called directly
        assert metadata['applied'] is True

    def test_zapline_registry(self):
        """Test ZaplineStep is registered in STEP_REGISTRY."""
        from eegcpm.modules.preprocessing.steps import STEP_REGISTRY

        assert 'zapline' in STEP_REGISTRY
        assert STEP_REGISTRY['zapline'].__name__ == 'ZaplineStep'

    def test_zapline_50hz(self):
        """Test ZaplineStep with 50 Hz line frequency."""
        from eegcpm.modules.preprocessing.steps import ZaplineStep

        # Create data with 50 Hz noise
        info = mne.create_info(
            ch_names=["C3", "C4", "Cz", "Pz", "O1"],
            sfreq=256.0,
            ch_types="eeg"
        )
        np.random.seed(42)
        n_samples = 256 * 30
        t = np.arange(n_samples) / 256.0

        data = np.zeros((5, n_samples))
        for i in range(5):
            data[i] = (
                np.random.randn(n_samples) * 5e-6
                + 10e-6 * np.sin(2 * np.pi * 50 * t)  # 50 Hz line noise
            )

        raw = mne.io.RawArray(data, info)

        step = ZaplineStep(fline=50)
        step.validate(raw)  # Should pass

        _, metadata = step.process(raw, {})
        assert metadata['applied'] is True
        assert metadata['fline'] == 50
