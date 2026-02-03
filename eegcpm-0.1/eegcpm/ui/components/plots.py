"""Plotting components for EEGCPM UI."""

from typing import Any, Dict, List, Optional

import numpy as np


def plot_connectivity_matrix(
    matrix: np.ndarray,
    roi_names: Optional[List[str]] = None,
    title: str = "Connectivity Matrix",
):
    """
    Create connectivity matrix heatmap.

    Args:
        matrix: Connectivity matrix (n_rois, n_rois)
        roi_names: List of ROI names
        title: Plot title

    Returns:
        Plotly figure
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        if roi_names is None:
            roi_names = [f"ROI_{i}" for i in range(matrix.shape[0])]

        fig = px.imshow(
            matrix,
            x=roi_names,
            y=roi_names,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=title,
        )

        fig.update_layout(
            width=800,
            height=700,
            xaxis_tickangle=-45,
        )

        return fig

    except ImportError:
        return None


def plot_erp(
    times: np.ndarray,
    data: np.ndarray,
    channel_names: Optional[List[str]] = None,
    title: str = "ERP",
):
    """
    Create ERP line plot.

    Args:
        times: Time vector
        data: ERP data (n_channels, n_times)
        channel_names: List of channel names
        title: Plot title

    Returns:
        Plotly figure
    """
    try:
        import plotly.graph_objects as go

        if channel_names is None:
            channel_names = [f"Ch_{i}" for i in range(data.shape[0])]

        fig = go.Figure()

        for i, ch_name in enumerate(channel_names):
            fig.add_trace(go.Scatter(
                x=times,
                y=data[i],
                mode="lines",
                name=ch_name,
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (µV)",
            width=800,
            height=400,
        )

        # Add vertical line at t=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        return fig

    except ImportError:
        return None


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs. Actual",
):
    """
    Create prediction scatter plot.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title

    Returns:
        Plotly figure
    """
    try:
        import plotly.graph_objects as go
        from scipy import stats

        r, p = stats.pearsonr(y_true, y_pred)

        fig = go.Figure()

        # Scatter points
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(size=10, opacity=0.7),
            name="Subjects",
        ))

        # Identity line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Identity",
        ))

        fig.update_layout(
            title=f"{title} (r = {r:.3f}, p = {p:.3e})",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            width=600,
            height=500,
        )

        return fig

    except ImportError:
        return None


def plot_network_summary(
    network_values: Dict[str, float],
    title: str = "Network Connectivity",
):
    """
    Create network-level bar chart.

    Args:
        network_values: Dict mapping network name to value
        title: Plot title

    Returns:
        Plotly figure
    """
    try:
        import plotly.express as px

        networks = list(network_values.keys())
        values = list(network_values.values())

        fig = px.bar(
            x=networks,
            y=values,
            title=title,
            labels={"x": "Network", "y": "Connectivity"},
        )

        fig.update_layout(
            width=700,
            height=400,
        )

        return fig

    except ImportError:
        return None


def plot_topography(
    data: np.ndarray,
    info,
    title: str = "Topography",
):
    """
    Create EEG topography plot using MNE.

    Args:
        data: Data values per channel
        info: MNE Info object
        title: Plot title

    Returns:
        Matplotlib figure
    """
    try:
        import mne
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
        )

        ax.set_title(title)

        return fig

    except ImportError:
        return None
