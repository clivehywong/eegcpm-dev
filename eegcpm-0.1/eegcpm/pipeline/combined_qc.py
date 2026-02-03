"""QC reporting for combined epoch analysis.

This module generates QC reports for epoch combination, showing:
- Per-run quality metrics
- Channel harmonization details
- Combined epochs statistics
- Event distribution across runs
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mne
from io import BytesIO
import base64


@dataclass
class CombinedQCReport:
    """QC report for combined epochs."""
    success: bool
    html_path: Optional[Path] = None
    error: Optional[str] = None


class CombinedQC:
    """Generate QC reports for epoch combination."""

    def __init__(self, output_dir: Path):
        """
        Initialize combined QC generator.

        Parameters
        ----------
        output_dir : Path
            Output directory for QC reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        combination_result,
        run_results: List[Any],
        subject_id: str,
        session: str,
        task: str
    ) -> CombinedQCReport:
        """
        Generate combined QC HTML report.

        Parameters
        ----------
        combination_result : EpochCombinationResult
            Result from epoch combination
        run_results : List[RunProcessingResult]
            Individual run processing results
        subject_id : str
            Subject identifier
        session : str
            Session identifier
        task : str
            Task name

        Returns
        -------
        CombinedQCReport
            QC report result
        """
        try:
            # Generate figures
            figures = {}

            # Figure 1: Run quality summary
            figures['run_quality'] = self._plot_run_quality(run_results, combination_result)

            # Figure 2: Channel harmonization
            figures['channels'] = self._plot_channel_harmonization(run_results, combination_result)

            # Figure 3: Event distribution
            if combination_result.success and combination_result.combined_epochs:
                figures['events'] = self._plot_event_distribution(combination_result.combined_epochs)

            # Generate HTML
            html_path = self.output_dir / f"{subject_id}_ses-{session}_task-{task}_combined_qc.html"
            self._write_html_report(
                html_path,
                subject_id,
                session,
                task,
                combination_result,
                run_results,
                figures
            )

            return CombinedQCReport(success=True, html_path=html_path)

        except Exception as e:
            return CombinedQCReport(success=False, error=str(e))

    def _plot_run_quality(self, run_results, combination_result) -> str:
        """Plot run quality metrics comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Extract metrics
        runs = []
        bad_pct = []
        clustering = []
        included = []

        for result in run_results:
            if result.quality_metrics:
                runs.append(f"Run {result.run}")
                bad_pct.append(result.quality_metrics.pct_bad_channels)
                clustering.append(result.quality_metrics.pct_clustered)
                included.append(result.run in combination_result.runs_included)

        # Plot 1: Bad channel percentage
        colors = ['green' if inc else 'red' for inc in included]
        axes[0].bar(runs, bad_pct, color=colors, alpha=0.7)
        axes[0].set_ylabel('Bad Channels (%)')
        axes[0].set_title('Bad Channel Percentage by Run')
        axes[0].axhline(y=10, color='orange', linestyle='--', label='Good threshold')
        axes[0].axhline(y=20, color='red', linestyle='--', label='Review threshold')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2: Clustering percentage
        axes[1].bar(runs, clustering, color=colors, alpha=0.7)
        axes[1].set_ylabel('Clustered Bad Channels (%)')
        axes[1].set_title('Bad Channel Clustering by Run')
        axes[1].tick_params(axis='x', rotation=45)

        # Plot 3: Quality status
        quality_scores = {'excellent': 4, 'good': 3, 'acceptable': 2, 'poor': 1}
        quality_vals = []
        for result in run_results:
            if result.quality_metrics:
                quality_vals.append(quality_scores.get(result.quality_metrics.quality_status, 0))
            else:
                quality_vals.append(0)

        axes[2].bar(runs, quality_vals, color=colors, alpha=0.7)
        axes[2].set_ylabel('Quality Score')
        axes[2].set_title('Overall Quality by Run')
        axes[2].set_yticks([1, 2, 3, 4])
        axes[2].set_yticklabels(['Poor', 'Acceptable', 'Good', 'Excellent'])
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_channel_harmonization(self, run_results, combination_result) -> str:
        """Plot channel harmonization details."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Extract channel counts
        runs = []
        original_counts = []
        final_counts = []

        for result in run_results:
            if result.run in combination_result.runs_included:
                runs.append(f"Run {result.run}")
                if result.quality_metrics:
                    original = result.quality_metrics.n_original_channels
                    bad = result.quality_metrics.n_bad_channels
                    original_counts.append(original - bad)
                else:
                    original_counts.append(0)

                # All runs have same final count (common channels)
                if result.raw_preprocessed:
                    final_counts.append(len(result.raw_preprocessed.ch_names))
                else:
                    final_counts.append(0)

        # If we have the actual common channel count from combination
        if len(runs) > 0 and combination_result.combined_epochs:
            common_count = len(combination_result.combined_epochs.ch_names)
            final_counts = [common_count] * len(runs)

        x = np.arange(len(runs))
        width = 0.35

        bars1 = ax.bar(x - width/2, original_counts, width, label='After preprocessing', color='steelblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, final_counts, width, label='Common channels', color='green', alpha=0.7)

        ax.set_xlabel('Run')
        ax.set_ylabel('Number of Channels')
        ax.set_title('Channel Harmonization Across Runs')
        ax.set_xticks(x)
        ax.set_xticklabels(runs)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_event_distribution(self, epochs: mne.Epochs) -> str:
        """Plot event type distribution in combined epochs."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Count events per type
        event_counts = {}
        for event_name, event_code in epochs.event_id.items():
            n_events = sum(epochs.events[:, 2] == event_code)
            event_counts[event_name] = n_events

        # Plot 1: Bar chart
        event_names = list(event_counts.keys())
        counts = list(event_counts.values())

        ax1.bar(event_names, counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Event Type')
        ax1.set_ylabel('Number of Epochs')
        ax1.set_title('Event Distribution')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (name, count) in enumerate(zip(event_names, counts)):
            ax1.text(i, count, str(count), ha='center', va='bottom')

        # Plot 2: Pie chart
        ax2.pie(counts, labels=event_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Event Type Proportions')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def _write_html_report(
        self,
        output_path: Path,
        subject_id: str,
        session: str,
        task: str,
        combination_result,
        run_results: List[Any],
        figures: Dict[str, str]
    ):
        """Write HTML report to file."""

        # Build run summary table
        run_rows = []
        for result in run_results:
            if result.quality_metrics:
                m = result.quality_metrics
                included = "✓" if result.run in combination_result.runs_included else "✗"
                run_rows.append(f"""
                <tr>
                    <td>{result.run}</td>
                    <td>{m.quality_status}</td>
                    <td>{m.pct_bad_channels:.1f}%</td>
                    <td>{m.clustering_severity}</td>
                    <td>{m.recommended_action}</td>
                    <td>{included}</td>
                </tr>
                """)

        run_table = "".join(run_rows)

        # Build event summary table if we have combined epochs
        event_table = ""
        if combination_result.success and combination_result.combined_epochs:
            event_rows = []
            epochs = combination_result.combined_epochs
            for event_name, event_code in epochs.event_id.items():
                n_events = sum(epochs.events[:, 2] == event_code)
                pct = (n_events / len(epochs)) * 100
                event_rows.append(f"""
                <tr>
                    <td>{event_name}</td>
                    <td>{n_events}</td>
                    <td>{pct:.1f}%</td>
                </tr>
                """)
            event_table = "".join(event_rows)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Combined Epochs QC - {subject_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #555;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #2c3e50;
        }}
        .success {{
            color: green;
        }}
        .warning {{
            color: orange;
        }}
        .error {{
            color: red;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Combined Epochs QC Report</h1>
        <p>Subject: {subject_id} | Session: {session} | Task: {task}</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Status:</span>
            <span class="metric-value {'success' if combination_result.success else 'error'}">
                {'✓ Success' if combination_result.success else '✗ Failed'}
            </span>
        </div>
        <div class="metric">
            <span class="metric-label">Runs Combined:</span>
            <span class="metric-value">{combination_result.n_runs_combined}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Epochs:</span>
            <span class="metric-value">{combination_result.n_total_epochs}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Runs Included:</span>
            <span class="metric-value">{', '.join(combination_result.runs_included)}</span>
        </div>
    </div>

    <div class="section">
        <h2>Run Quality Assessment</h2>
        <table>
            <thead>
                <tr>
                    <th>Run</th>
                    <th>Quality</th>
                    <th>Bad Channels</th>
                    <th>Clustering</th>
                    <th>Recommendation</th>
                    <th>Included</th>
                </tr>
            </thead>
            <tbody>
                {run_table}
            </tbody>
        </table>

        <h3>Run Quality Comparison</h3>
        <img src="data:image/png;base64,{figures.get('run_quality', '')}" alt="Run Quality">
    </div>

    <div class="section">
        <h2>Channel Harmonization</h2>
        <p>Common channels kept across all runs to ensure compatibility.</p>
        <img src="data:image/png;base64,{figures.get('channels', '')}" alt="Channel Harmonization">
    </div>

    {'<div class="section"><h2>Event Distribution</h2>' +
     '<table><thead><tr><th>Event Type</th><th>Count</th><th>Percentage</th></tr></thead>' +
     '<tbody>' + event_table + '</tbody></table>' +
     '<h3>Event Type Distribution</h3>' +
     f'<img src="data:image/png;base64,{figures.get("events", "")}" alt="Event Distribution">' +
     '</div>' if event_table else ''}

    <div class="section">
        <h2>Output Files</h2>
        <p><strong>Combined epochs:</strong> {combination_result.output_path}</p>
    </div>
</body>
</html>
        """

        output_path.write_text(html_content)
