"""Tests for CONN ROI module."""

import numpy as np

from eegcpm.data.conn_rois import (
    get_conn_rois,
    get_roi_names,
    get_mni_coordinates,
    get_network_indices,
    get_rois_by_network,
    CONN_NETWORKS,
)


class TestConnROIs:
    """Test CONN ROI definitions."""

    def test_roi_count(self):
        """Test we have 32 ROIs."""
        rois = get_conn_rois()
        assert len(rois) == 32

    def test_network_count(self):
        """Test we have 8 networks."""
        assert len(CONN_NETWORKS) == 8

    def test_network_roi_counts(self):
        """Test ROI counts per network match."""
        expected = {
            "DefaultMode": 4,
            "SensoriMotor": 3,
            "Visual": 4,
            "Salience": 7,
            "DorsalAttention": 4,
            "FrontoParietal": 4,
            "Language": 4,
            "Cerebellar": 2,
        }
        assert CONN_NETWORKS == expected

        # Verify actual ROIs match counts
        total = 0
        for network, count in expected.items():
            network_rois = get_rois_by_network(network)
            assert len(network_rois) == count
            total += count
        assert total == 32

    def test_roi_names(self):
        """Test ROI name format."""
        names = get_roi_names()
        assert len(names) == 32
        assert names[0] == "DefaultMode.MPFC"
        assert "Salience.ACC" in names

    def test_mni_coordinates(self):
        """Test MNI coordinate shape."""
        coords = get_mni_coordinates()
        assert coords.shape == (32, 3)
        # Coordinates are integers in MNI space
        assert np.issubdtype(coords.dtype, np.number)

    def test_network_indices(self):
        """Test network index mapping."""
        indices = get_network_indices()
        assert len(indices) == 8
        assert len(indices["DefaultMode"]) == 4
        assert len(indices["Salience"]) == 7

        # Indices should cover all 32 ROIs
        all_indices = []
        for idx_list in indices.values():
            all_indices.extend(idx_list)
        assert sorted(all_indices) == list(range(32))
