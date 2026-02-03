"""
Event-Related Potential (ERP) Quality Control Module

This module generates comprehensive QC reports for event-related potentials,
including condition comparisons, waveform plots, and topographic maps.

The reports use MNE-Python's built-in plotting functions for standardized
ERP visualizations familiar to EEG researchers.

Author: EEGCPM Development Team
Created: 2025-01
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
import io

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend for report generation
matplotlib.use('Agg')


class ERPQC:
    """
    Generate Event-Related Potential quality control reports.

    This class creates HTML reports with embedded visualizations showing:
    - Individual condition ERP waveforms
    - Topographic maps at key timepoints
    - Condition comparisons
    - Peak detection metrics (P1, N1, P3, etc.)

    The reports help users validate preprocessing quality before proceeding
    to connectivity analysis.

    Parameters
    ----------
    output_dir : Path
        Directory where QC reports will be saved

    Examples
    --------
    >>> from pathlib import Path
    >>> import mne
    >>>
    >>> # Create QC generator
    >>> qc = ERPQC(Path("derivatives/qc/erp"))
    >>>
    >>> # Generate report for condition-based ERPs
    >>> evokeds = {
    ...     'left_stimulus': evoked_left,
    ...     'right_stimulus': evoked_right
    ... }
    >>> report_path = qc.generate_report(
    ...     evokeds=evokeds,
    ...     subject_id='sub-001',
    ...     session_id='ses-01'
    ... )
    """

    def __init__(self, output_dir: Path):
        """
        Initialize ERP QC report generator.

        Parameters
        ----------
        output_dir : Path
            Directory where reports will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        evokeds: Dict[str, mne.Evoked],
        subject_id: str,
        session_id: Optional[str] = None,
        roi_channels: Optional[List[str]] = None,
        peak_windows: Optional[Dict[str, Tuple[float, float]]] = None,
        epochs_dict: Optional[Dict[str, mne.Epochs]] = None
    ) -> Path:
        """
        Generate comprehensive ERP QC report.

        Creates an HTML report with embedded visualizations showing:
        1. Individual condition waveforms with spatial colors
        2. Topographic maps at key latencies
        3. Condition comparison overlays
        4. Extracted peak metrics (latency, amplitude)

        Parameters
        ----------
        evokeds : dict of {str: mne.Evoked}
            Dictionary mapping condition names to Evoked objects
            Example: {'left': evoked_left, 'right': evoked_right}
        subject_id : str
            Subject identifier (e.g., 'sub-001')
        session_id : str, optional
            Session identifier (e.g., 'ses-01')
        roi_channels : list of str, optional
            Specific channels to highlight (e.g., occipital electrodes
            for visual ERPs). If None, uses all EEG channels.
        peak_windows : dict of {str: tuple}, optional
            Time windows (in seconds) for peak detection
            Default: {'P1': (0.08, 0.13), 'N1': (0.13, 0.20), 'P3': (0.30, 0.50)}
        epochs_dict : dict of {str: mne.Epochs}, optional
            Dictionary mapping condition names to Epochs objects (single trials).
            If provided, enables statistical testing of condition differences.
            Example: {'left': epochs_left, 'right': epochs_right}

        Returns
        -------
        Path
            Path to generated HTML report

        Notes
        -----
        The report includes:
        - Epoch counts per condition
        - Detected peak latencies and amplitudes
        - Global field power (GFP) for overall signal strength
        - Visual comparison across conditions

        All figures are embedded as base64-encoded PNG images for
        self-contained HTML reports.
        """
        # Set default peak detection windows
        if peak_windows is None:
            peak_windows = {
                'P1': (0.08, 0.13),   # 80-130ms: Early visual component
                'N1': (0.13, 0.20),   # 130-200ms: Visual discrimination
                'P3': (0.30, 0.50)    # 300-500ms: Cognitive processing
            }

        # Generate all figures
        figures = {}
        metrics = {}

        # Process each condition
        for condition, evoked in evokeds.items():
            # 1. Waveform plot with spatial colors
            fig_waveform = self._plot_waveforms(
                evoked,
                condition,
                roi_channels
            )
            figures[f'{condition}_waveform'] = self._fig_to_base64(fig_waveform)
            plt.close(fig_waveform)

            # 2. Topographic maps at key latencies
            fig_topo = self._plot_topomaps(evoked, condition, peak_windows)
            if fig_topo is not None:
                figures[f'{condition}_topomap'] = self._fig_to_base64(fig_topo)
                plt.close(fig_topo)

            # 3. Extract peak metrics
            metrics[condition] = self._extract_peak_metrics(
                evoked,
                peak_windows,
                roi_channels
            )

        # 4. Condition comparison (if multiple conditions)
        if len(evokeds) > 1:
            fig_compare = self._plot_comparison(evokeds, roi_channels)
            if fig_compare is not None:
                figures['comparison'] = self._fig_to_base64(fig_compare)
                plt.close(fig_compare)

            # 4b. Difference waves with statistical testing (if exactly 2 conditions)
            if len(evokeds) == 2:
                conditions = list(evokeds.keys())
                condition1, condition2 = conditions[0], conditions[1]

                # Get epochs if provided
                epochs1 = epochs_dict.get(condition1) if epochs_dict else None
                epochs2 = epochs_dict.get(condition2) if epochs_dict else None

                fig_diff = self._plot_difference_waves_with_stats(
                    condition1=condition1,
                    condition2=condition2,
                    evoked1=evokeds[condition1],
                    evoked2=evokeds[condition2],
                    epochs1=epochs1,
                    epochs2=epochs2,
                    roi_channels=roi_channels
                )
                if fig_diff is not None:
                    figures['difference_waves'] = self._fig_to_base64(fig_diff)
                    plt.close(fig_diff)

        # 5. Generate HTML report
        html_content = self._create_html_report(
            subject_id=subject_id,
            session_id=session_id,
            figures=figures,
            metrics=metrics,
            evokeds=evokeds
        )

        # Save HTML file
        session_str = f"_ses-{session_id}" if session_id else ""
        output_file = self.output_dir / f"sub-{subject_id}{session_str}_erp_qc.html"
        output_file.write_text(html_content, encoding='utf-8')

        return output_file

    def _plot_waveforms(
        self,
        evoked: mne.Evoked,
        condition: str,
        roi_channels: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Plot ERP waveforms with spatial colors and global field power.

        Uses MNE's plot() function to create publication-quality waveforms.

        Parameters
        ----------
        evoked : mne.Evoked
            Evoked response to plot
        condition : str
            Condition name for title
        roi_channels : list of str, optional
            Specific channels to plot

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        # Select channels to plot
        picks = roi_channels if roi_channels else 'eeg'

        try:
            # Create waveform plot
            fig = evoked.plot(
                picks=picks,
                spatial_colors=True,  # Color lines by sensor position
                gfp=True,             # Show global field power
                titles=dict(eeg=f'{condition.replace("_", " ").title()} - ERP Waveforms'),
                show=False,
                time_unit='ms'
            )
            return fig
        except Exception as e:
            print(f"Warning: Could not create waveform plot for {condition}: {e}")
            return plt.figure()

    def _plot_topomaps(
        self,
        evoked: mne.Evoked,
        condition: str,
        peak_windows: Dict[str, Tuple[float, float]]
    ) -> Optional[plt.Figure]:
        """
        Plot topographic maps at peak latencies.

        Shows spatial distribution of activity at key timepoints
        (e.g., P1, N1, P3 peaks).

        Parameters
        ----------
        evoked : mne.Evoked
            Evoked response to plot
        condition : str
            Condition name for title
        peak_windows : dict
            Time windows for peaks

        Returns
        -------
        Figure or None
            Matplotlib figure object, or None if plotting fails
        """
        # Get center of each peak window
        times = [np.mean(window) for window in peak_windows.values()]

        try:
            fig = evoked.plot_topomap(
                times=times,
                show=False,
                time_unit='ms',
                colorbar=True
            )
            # Add condition name to figure title
            fig.suptitle(f'{condition.replace("_", " ").title()} - Topographic Maps')
            return fig
        except Exception as e:
            print(f"Warning: Could not create topomap for {condition}: {e}")
            return None

    def _plot_comparison(
        self,
        evokeds: Dict[str, mne.Evoked],
        roi_channels: Optional[List[str]] = None
    ) -> Optional[plt.Figure]:
        """
        Plot condition comparison with overlaid waveforms.

        Uses MNE's plot_compare_evokeds() for standardized comparison plots.

        Parameters
        ----------
        evokeds : dict
            Dictionary of condition name -> Evoked pairs
        roi_channels : list of str, optional
            Channels to include in comparison

        Returns
        -------
        Figure or None
            Matplotlib figure object, or None if plotting fails
        """
        # Select channels
        picks = roi_channels if roi_channels else 'eeg'

        try:
            fig = mne.viz.plot_compare_evokeds(
                evokeds,
                picks=picks,
                title='Condition Comparison',
                show=False,
                time_unit='ms',
                legend='upper left'
            )
            return fig[0] if isinstance(fig, list) else fig
        except Exception as e:
            print(f"Warning: Could not create comparison plot: {e}")
            return None

    def _plot_difference_waves_with_stats(
        self,
        condition1: str,
        condition2: str,
        evoked1: mne.Evoked,
        evoked2: mne.Evoked,
        epochs1: Optional[mne.Epochs] = None,
        epochs2: Optional[mne.Epochs] = None,
        roi_channels: Optional[List[str]] = None
    ) -> Optional[plt.Figure]:
        """
        Plot ERP difference waves with statistical significance masks.

        Computes condition1 - condition2 and performs cluster-based
        permutation testing to identify significant time windows.

        This is a standard approach in ERP research for identifying
        condition effects with proper statistical control.

        Parameters
        ----------
        condition1 : str
            Name of first condition
        condition2 : str
            Name of second condition
        evoked1 : mne.Evoked
            Averaged ERP for condition 1
        evoked2 : mne.Evoked
            Averaged ERP for condition 2
        epochs1 : mne.Epochs, optional
            Single-trial data for condition 1 (needed for statistics)
            If None, skips statistical testing
        epochs2 : mne.Epochs, optional
            Single-trial data for condition 2 (needed for statistics)
        roi_channels : list of str, optional
            Channels to include in analysis

        Returns
        -------
        Figure or None
            Matplotlib figure with difference waves and significance masks

        Notes
        -----
        Uses MNE's cluster permutation test with default parameters:
        - 1000 permutations
        - Cluster-forming threshold: p < 0.05
        - Family-wise error rate control via cluster mass

        Significant time windows are highlighted with shaded regions.
        """
        # Compute difference wave (condition1 - condition2)
        diff_evoked = mne.combine_evoked([evoked1, evoked2], weights=[1, -1])

        # Select channels
        if roi_channels:
            available_channels = [ch for ch in roi_channels if ch in diff_evoked.ch_names]
            if available_channels:
                picks = [diff_evoked.ch_names.index(ch) for ch in available_channels]
            else:
                picks = mne.pick_types(diff_evoked.info, eeg=True)
        else:
            picks = mne.pick_types(diff_evoked.info, eeg=True)

        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Subplot 1: Individual condition ERPs + difference wave
        ax1 = axes[0]

        # Get times and data
        times = evoked1.times * 1000  # Convert to ms
        data1 = evoked1.get_data(picks=picks).mean(axis=0) * 1e6  # Average across channels, convert to µV
        data2 = evoked2.get_data(picks=picks).mean(axis=0) * 1e6
        data_diff = diff_evoked.get_data(picks=picks).mean(axis=0) * 1e6

        # Plot individual conditions (lighter)
        ax1.plot(times, data1, label=condition1, color='#377EB8', alpha=0.5, linewidth=1.5)
        ax1.plot(times, data2, label=condition2, color='#E41A1C', alpha=0.5, linewidth=1.5)

        # Plot difference wave (bold)
        ax1.plot(times, data_diff, label=f'{condition1} - {condition2}',
                color='black', linewidth=2.5)

        # Add zero lines
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

        ax1.set_xlabel('Time (ms)', fontsize=11)
        ax1.set_ylabel('Amplitude (µV)', fontsize=11)
        ax1.set_title(f'ERP Difference Wave: {condition1} vs {condition2}', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Difference wave with statistical significance
        ax2 = axes[1]

        # Plot difference wave
        ax2.plot(times, data_diff, color='black', linewidth=2)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

        # Perform cluster permutation test if epochs are available
        if epochs1 is not None and epochs2 is not None:
            try:
                # Get single-trial data for selected channels
                X1 = epochs1.get_data(picks=picks)  # (n_epochs1, n_channels, n_times)
                X2 = epochs2.get_data(picks=picks)  # (n_epochs2, n_channels, n_times)

                # Average across channels for simpler visualization
                X1_avg = X1.mean(axis=1)  # (n_epochs1, n_times)
                X2_avg = X2.mean(axis=1)  # (n_epochs2, n_times)

                # Ensure same number of time points
                n_times = min(X1_avg.shape[1], X2_avg.shape[1])
                X1_avg = X1_avg[:, :n_times]
                X2_avg = X2_avg[:, :n_times]

                # Cluster-based permutation test
                from mne.stats import permutation_cluster_test

                # Prepare data for test: (n_observations, n_times)
                # We need to test if X1 - X2 is significantly different from 0
                X = [X1_avg, X2_avg]

                # Perform test
                T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                    X,
                    n_permutations=1000,
                    threshold=None,  # Use automatic threshold (t-test)
                    tail=0,  # Two-tailed test
                    n_jobs=1,
                    out_type='mask',
                    verbose=False
                )

                # Highlight significant clusters (p < 0.05)
                y_min, y_max = ax2.get_ylim()
                for i_cluster, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
                    if p_val < 0.05:
                        # cluster is a boolean array of time points in the cluster
                        # Find start and end of cluster
                        cluster_times = times[:n_times][cluster]
                        if len(cluster_times) > 0:
                            t_start = cluster_times[0]
                            t_end = cluster_times[-1]

                            # Shade significant time window
                            ax2.axvspan(t_start, t_end, alpha=0.3, color='gold',
                                      label=f'p < 0.05' if i_cluster == 0 else '')

                            # Add p-value annotation
                            t_center = (t_start + t_end) / 2
                            ax2.text(t_center, y_max * 0.9, f'p={p_val:.3f}',
                                   ha='center', va='top', fontsize=8,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                if any(p < 0.05 for p in cluster_p_values):
                    ax2.legend(loc='upper right', fontsize=9)
                    stats_text = f"Cluster permutation test: {len([p for p in cluster_p_values if p < 0.05])}/{len(clusters)} significant clusters"
                else:
                    stats_text = "Cluster permutation test: No significant differences"

            except Exception as e:
                stats_text = f"Statistical testing failed: {str(e)[:80]}"
                print(f"Warning: Cluster permutation test failed: {e}")

        else:
            stats_text = "Statistical testing skipped (epochs not provided)"

        ax2.set_xlabel('Time (ms)', fontsize=11)
        ax2.set_ylabel('Amplitude (µV)', fontsize=11)
        ax2.set_title('Difference Wave with Significance Mask', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add statistics info box
        ax2.text(0.02, 0.02, stats_text,
                transform=ax2.transAxes, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.tight_layout()
        return fig

    def _extract_peak_metrics(
        self,
        evoked: mne.Evoked,
        peak_windows: Dict[str, Tuple[float, float]],
        roi_channels: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract peak latency and amplitude metrics.

        Finds positive/negative peaks within specified time windows
        and returns their latencies and amplitudes.

        Parameters
        ----------
        evoked : mne.Evoked
            Evoked response to analyze
        peak_windows : dict
            Time windows for each component
        roi_channels : list of str, optional
            Channels to average for peak detection

        Returns
        -------
        dict
            Nested dictionary with structure:
            {
                'P1': {'latency_ms': 105.2, 'amplitude_uv': 2.3},
                'N1': {'latency_ms': 178.1, 'amplitude_uv': -3.1},
                ...
            }
        """
        # Get data averaged across ROI channels
        if roi_channels:
            # Convert channel names to indices
            available_channels = [ch for ch in roi_channels if ch in evoked.ch_names]
            if available_channels:
                picks = [evoked.ch_names.index(ch) for ch in available_channels]
            else:
                picks = mne.pick_types(evoked.info, eeg=True)
        else:
            picks = mne.pick_types(evoked.info, eeg=True)

        data = evoked.data[picks, :].mean(axis=0) * 1e6  # Convert to µV
        times_ms = evoked.times * 1000  # Convert to ms

        metrics = {}
        metrics['n_epochs'] = evoked.nave

        # Detect peaks in each window
        for peak_name, (tmin, tmax) in peak_windows.items():
            tmin_ms, tmax_ms = tmin * 1000, tmax * 1000
            mask = (times_ms >= tmin_ms) & (times_ms <= tmax_ms)

            if not mask.any():
                continue

            window_data = data[mask]
            window_times = times_ms[mask]

            # Detect positive (P) or negative (N) peak
            if peak_name.startswith('P'):
                peak_idx = window_data.argmax()
            else:  # N components
                peak_idx = window_data.argmin()

            metrics[peak_name] = {
                'latency_ms': float(window_times[peak_idx]),
                'amplitude_uv': float(window_data[peak_idx])
            }

        return metrics

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64-encoded PNG.

        This allows embedding figures directly in HTML without external files.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to convert

        Returns
        -------
        str
            Data URI with base64-encoded PNG
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return f"data:image/png;base64,{img_base64}"

    def _create_html_report(
        self,
        subject_id: str,
        session_id: Optional[str],
        figures: Dict[str, str],
        metrics: Dict[str, Dict],
        evokeds: Dict[str, mne.Evoked]
    ) -> str:
        """
        Create HTML report with embedded figures and metrics.

        Parameters
        ----------
        subject_id : str
            Subject identifier
        session_id : str, optional
            Session identifier
        figures : dict
            Figure name -> base64 image data
        metrics : dict
            Condition -> metrics dictionary
        evokeds : dict
            Condition -> Evoked objects

        Returns
        -------
        str
            Complete HTML document
        """
        # Build metrics table
        metrics_html = self._create_metrics_table(metrics)

        # Build HTML document
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ERP QC Report - {subject_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .info p {{
            margin: 5px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .figure {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .figure h3 {{
            margin-top: 0;
            color: #2980b9;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px auto;
            border-radius: 4px;
        }}
        .note {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .note strong {{
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Event-Related Potential QC Report</h1>

        <div class="info">
            <p><strong>Subject:</strong> {subject_id}</p>
            <p><strong>Session:</strong> {session_id if session_id else 'N/A'}</p>
            <p><strong>Conditions:</strong> {', '.join(evokeds.keys())}</p>
            <p><strong>Generated:</strong> {self._get_timestamp()}</p>
        </div>

        <div class="note">
            <strong>Note:</strong> This report shows averaged event-related potentials (ERPs)
            by experimental condition. Review waveforms and topographies to verify preprocessing
            quality before proceeding to connectivity analysis.
        </div>

        <h2>ERP Metrics Summary</h2>
        {metrics_html}

        <h2>Condition-Specific Waveforms & Topographies</h2>
"""

        # Add individual condition plots
        for condition in evokeds.keys():
            waveform_key = f'{condition}_waveform'
            topo_key = f'{condition}_topomap'

            html += f"""
        <div class="figure">
            <h3>{condition.replace('_', ' ').title()}</h3>
"""
            if waveform_key in figures:
                html += f'            <img src="{figures[waveform_key]}" alt="{condition} waveforms"/>\n'

            if topo_key in figures:
                html += f'            <img src="{figures[topo_key]}" alt="{condition} topomaps"/>\n'

            html += "        </div>\n"

        # Add comparison plot if available
        if 'comparison' in figures:
            html += f"""
        <h2>Condition Comparison</h2>
        <div class="figure">
            <img src="{figures['comparison']}" alt="Condition comparison"/>
        </div>
"""

        # Add difference waves with statistics if available
        if 'difference_waves' in figures:
            html += f"""
        <h2>Difference Waves with Statistical Testing</h2>
        <div class="figure">
            <img src="{figures['difference_waves']}" alt="Difference waves with cluster permutation statistics"/>
            <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                <strong>Note:</strong> Gold shading indicates statistically significant time windows
                (p < 0.05, cluster-based permutation test with family-wise error correction).
                This analysis identifies when the two conditions differ significantly.
            </p>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""

        return html

    def _create_metrics_table(self, metrics: Dict[str, Dict]) -> str:
        """
        Create HTML table from metrics dictionary.

        Parameters
        ----------
        metrics : dict
            Nested dictionary of condition -> metric -> value

        Returns
        -------
        str
            HTML table markup
        """
        # Extract all peak names (P1, N1, etc.)
        peak_names = set()
        for condition_metrics in metrics.values():
            peak_names.update(k for k in condition_metrics.keys() if k != 'n_epochs')
        peak_names = sorted(peak_names)

        # Build table header
        html = '<table>\n<thead>\n<tr>\n'
        html += '<th>Condition</th><th>N Epochs</th>'
        for peak in peak_names:
            html += f'<th>{peak} Latency (ms)</th><th>{peak} Amplitude (µV)</th>'
        html += '\n</tr>\n</thead>\n<tbody>\n'

        # Build table rows
        for condition, condition_metrics in metrics.items():
            html += f'<tr>\n<td><strong>{condition}</strong></td>'
            html += f'<td>{condition_metrics.get("n_epochs", "N/A")}</td>'

            for peak in peak_names:
                if peak in condition_metrics:
                    latency = condition_metrics[peak]['latency_ms']
                    amplitude = condition_metrics[peak]['amplitude_uv']
                    html += f'<td>{latency:.1f}</td><td>{amplitude:.2f}</td>'
                else:
                    html += '<td>-</td><td>-</td>'

            html += '\n</tr>\n'

        html += '</tbody>\n</table>'

        return html

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in readable format."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
