"""Tests for data loaders module."""

import pytest
import tempfile
from pathlib import Path

import numpy as np
import mne

from eegcpm.data.loaders import load_raw, load_epochs, load_events, get_montage


class TestLoadRaw:
    """Test load_raw function."""

    @pytest.fixture
    def sample_raw(self):
        """Create a sample raw object for testing."""
        info = mne.create_info(
            ch_names=["EEG 001", "EEG 002", "EEG 003"],
            sfreq=256.0,
            ch_types="eeg",
        )
        data = np.random.randn(3, 256 * 10)  # 10 seconds of data
        return mne.io.RawArray(data, info)

    def test_load_raw_fif(self, sample_raw):
        """Test loading .fif file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fif_path = Path(tmpdir) / "test_raw.fif"
            sample_raw.save(fif_path, overwrite=True)

            loaded = load_raw(fif_path, verbose=False)

            assert isinstance(loaded, mne.io.BaseRaw)
            assert len(loaded.ch_names) == 3
            assert loaded.info["sfreq"] == 256.0

    def test_load_raw_with_preload_false(self, sample_raw):
        """Test loading with preload=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fif_path = Path(tmpdir) / "test_raw.fif"
            sample_raw.save(fif_path, overwrite=True)

            loaded = load_raw(fif_path, preload=False, verbose=False)

            assert isinstance(loaded, mne.io.BaseRaw)
            # Data should not be preloaded
            assert not loaded.preload

    def test_load_raw_unsupported_format(self):
        """Test that unsupported formats raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_raw(Path("/tmp/test.xyz"), verbose=False)

    def test_load_raw_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_raw(Path("/nonexistent/path/file.fif"), verbose=False)

    def test_load_raw_string_path(self, sample_raw):
        """Test loading with string path instead of Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fif_path = Path(tmpdir) / "test_raw.fif"
            sample_raw.save(fif_path, overwrite=True)

            loaded = load_raw(str(fif_path), verbose=False)

            assert isinstance(loaded, mne.io.BaseRaw)


class TestLoadEpochs:
    """Test load_epochs function."""

    @pytest.fixture
    def sample_epochs(self):
        """Create sample epochs for testing."""
        info = mne.create_info(
            ch_names=["EEG 001", "EEG 002", "EEG 003"],
            sfreq=256.0,
            ch_types="eeg",
        )
        # Create 10 epochs of 1 second each
        data = np.random.randn(10, 3, 256)
        events = np.array([[i * 256, 0, 1] for i in range(10)])
        return mne.EpochsArray(data, info, events=events, tmin=0)

    def test_load_epochs_fif(self, sample_epochs):
        """Test loading epochs from .fif file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            epo_path = Path(tmpdir) / "test-epo.fif"
            sample_epochs.save(epo_path, overwrite=True)

            loaded = load_epochs(epo_path, verbose=False)

            # mne.read_epochs returns EpochsFIF which is a subclass of BaseEpochs
            assert isinstance(loaded, mne.BaseEpochs)
            assert len(loaded) == 10
            assert len(loaded.ch_names) == 3

    def test_load_epochs_with_preload_false(self, sample_epochs):
        """Test loading epochs with preload=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            epo_path = Path(tmpdir) / "test-epo.fif"
            sample_epochs.save(epo_path, overwrite=True)

            loaded = load_epochs(epo_path, preload=False, verbose=False)

            assert isinstance(loaded, mne.BaseEpochs)

    def test_load_epochs_file_not_found(self):
        """Test loading non-existent epochs file."""
        with pytest.raises(FileNotFoundError):
            load_epochs(Path("/nonexistent/path/file-epo.fif"), verbose=False)


class TestLoadEvents:
    """Test load_events function."""

    @pytest.fixture
    def sample_raw_with_stim(self):
        """Create raw with stim channel for event extraction."""
        info = mne.create_info(
            ch_names=["EEG 001", "EEG 002", "STI 014"],
            sfreq=256.0,
            ch_types=["eeg", "eeg", "stim"],
        )
        data = np.zeros((3, 256 * 10))
        # Add some events to stim channel
        data[2, 256] = 1  # Event at 1 second
        data[2, 512] = 2  # Event at 2 seconds
        data[2, 768] = 1  # Event at 3 seconds
        return mne.io.RawArray(data, info)

    def test_load_events_from_tsv(self, sample_raw_with_stim):
        """Test loading events from TSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "events.tsv"

            # Create BIDS-style events TSV
            with open(tsv_path, "w") as f:
                f.write("onset\tduration\ttrial_type\n")
                f.write("1.0\t0.0\ttarget\n")
                f.write("2.0\t0.0\tstandard\n")
                f.write("3.0\t0.0\ttarget\n")

            events, event_id = load_events(tsv_path, raw=sample_raw_with_stim)

            assert events.shape[0] == 3
            assert events.shape[1] == 3
            assert "target" in event_id
            assert "standard" in event_id

    def test_load_events_from_raw_stim(self, sample_raw_with_stim):
        """Test extracting events from raw stim channel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the raw file
            fif_path = Path(tmpdir) / "raw.fif"
            sample_raw_with_stim.save(fif_path, overwrite=True)

            events, event_id = load_events(fif_path, raw=sample_raw_with_stim)

            assert events.shape[0] == 3
            assert "1" in event_id
            assert "2" in event_id

    def test_load_events_no_raw_no_tsv(self):
        """Test that ValueError is raised when no raw and no valid tsv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-tsv file
            txt_path = Path(tmpdir) / "events.txt"
            txt_path.touch()

            with pytest.raises(ValueError, match="Either provide events file or raw object"):
                load_events(txt_path)

    def test_load_events_tsv_without_raw(self):
        """Test loading TSV without raw (sample indices instead of times)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "events.tsv"

            with open(tsv_path, "w") as f:
                f.write("onset\tduration\ttrial_type\n")
                f.write("256\t0.0\ttarget\n")
                f.write("512\t0.0\tstandard\n")

            events, event_id = load_events(tsv_path)

            # Without raw, onset is treated as sample index
            assert events[0, 0] == 256
            assert events[1, 0] == 512


class TestGetMontage:
    """Test get_montage function."""

    def test_standard_1020_montage(self):
        """Test getting standard 10-20 montage."""
        montage = get_montage("standard_1020")

        assert isinstance(montage, mne.channels.DigMontage)
        ch_names = montage.ch_names
        assert "Fp1" in ch_names
        assert "Fp2" in ch_names
        assert "Fz" in ch_names
        assert "Cz" in ch_names

    def test_standard_1005_montage(self):
        """Test getting standard 10-05 montage."""
        montage = get_montage("standard_1005")

        assert isinstance(montage, mne.channels.DigMontage)
        # 10-05 has more channels than 10-20
        assert len(montage.ch_names) > 60

    def test_biosemi64_montage(self):
        """Test getting biosemi64 montage."""
        montage = get_montage("biosemi64")

        assert isinstance(montage, mne.channels.DigMontage)
        assert len(montage.ch_names) == 64

    def test_invalid_montage(self):
        """Test that invalid montage name raises error."""
        with pytest.raises(ValueError):
            get_montage("nonexistent_montage_xyz")

    def test_default_montage(self):
        """Test default montage is standard_1020."""
        default = get_montage()
        explicit = get_montage("standard_1020")

        # Both should have the same channels
        assert set(default.ch_names) == set(explicit.ch_names)
