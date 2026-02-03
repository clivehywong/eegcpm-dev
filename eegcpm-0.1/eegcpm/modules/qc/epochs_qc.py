"""Epochs Quality Control Report Generation.

Unified QC report generation for epochs, used by both interactive and batch modes.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import mne
from datetime import datetime


def generate_epochs_qc_report(
    epochs: mne.Epochs,
    task_config: Dict[str, Any],
    output_path: Path,
    runs_included: Optional[List[str]] = None,
    runs_excluded: Optional[List[str]] = None,
    subject_id: str = "unknown",
    session: str = "01",
    task: str = "unknown",
) -> Path:
    """
    Generate standardized HTML QC report for epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data
    task_config : dict
        Task configuration used for epoching
    output_path : Path
        Output HTML file path
    runs_included : list, optional
        List of run IDs that were included
    runs_excluded : list, optional
        List of run IDs that were excluded
    subject_id : str
        Subject identifier
    session : str
        Session identifier
    task : str
        Task name

    Returns
    -------
    Path
        Path to generated HTML report
    """

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect QC metrics
    n_epochs = len(epochs)
    n_conditions = len(epochs.event_id)
    drop_log_stats = epochs.drop_log_stats()

    # Per-condition trial counts
    condition_counts = {}
    for condition in epochs.event_id:
        if condition in epochs.event_id:
            condition_counts[condition] = len(epochs[condition])

    # Generate HTML report
    html = _generate_html_template(
        subject_id=subject_id,
        session=session,
        task=task,
        task_config=task_config,
        n_epochs=n_epochs,
        n_conditions=n_conditions,
        condition_counts=condition_counts,
        drop_log_stats=drop_log_stats,
        runs_included=runs_included,
        runs_excluded=runs_excluded,
        epochs=epochs,
    )

    # Write to file
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def _generate_html_template(
    subject_id: str,
    session: str,
    task: str,
    task_config: Dict[str, Any],
    n_epochs: int,
    n_conditions: int,
    condition_counts: Dict[str, int],
    drop_log_stats: float,
    runs_included: Optional[List[str]],
    runs_excluded: Optional[List[str]],
    epochs: mne.Epochs,
) -> str:
    """Generate HTML template for epochs QC report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build run info section
    run_info_html = ""
    if runs_included:
        run_info_html += f"<p><strong>Runs Included:</strong> {', '.join(runs_included)}</p>\n"
    if runs_excluded:
        run_info_html += f"<p><strong>Runs Excluded:</strong> {', '.join(runs_excluded)}</p>\n"

    # Build condition table
    condition_rows = ""
    for condition, count in sorted(condition_counts.items()):
        condition_rows += f"""
        <tr>
            <td>{condition}</td>
            <td>{count}</td>
            <td>{count / n_epochs * 100:.1f}%</td>
        </tr>
        """

    # Task config details
    config_details = f"""
    <p><strong>Time Window:</strong> {task_config.get('tmin', 'N/A')} to {task_config.get('tmax', 'N/A')} s</p>
    <p><strong>Baseline:</strong> {task_config.get('baseline', 'N/A')}</p>
    """

    if 'conditions' in task_config:
        config_details += "<p><strong>Conditions Defined:</strong></p><ul>"
        for cond in task_config['conditions']:
            config_details += f"<li>{cond.get('name', 'Unknown')}: {cond.get('event_codes', [])}</li>"
        config_details += "</ul>"

    # Generate ERP plots as base64 (simple placeholder for now)
    erp_plots_html = _generate_erp_plots_html(epochs, condition_counts.keys())

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Epochs QC Report - {subject_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 150px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        .success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Epochs Quality Control Report</h1>

        <div class="timestamp">Generated: {timestamp}</div>

        <h2>Subject Information</h2>
        <p><strong>Subject:</strong> {subject_id}</p>
        <p><strong>Session:</strong> {session}</p>
        <p><strong>Task:</strong> {task}</p>
        {run_info_html}

        <h2>Epoch Summary</h2>
        <div class="metric">
            <div class="metric-label">Total Epochs</div>
            <div class="metric-value">{n_epochs}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Conditions</div>
            <div class="metric-value">{n_conditions}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Drop Rate</div>
            <div class="metric-value">{drop_log_stats:.1f}%</div>
        </div>

        <h2>Task Configuration</h2>
        <div class="info-box">
            {config_details}
        </div>

        <h2>Per-Condition Trial Counts</h2>
        <table>
            <thead>
                <tr>
                    <th>Condition</th>
                    <th>Trial Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                {condition_rows}
            </tbody>
        </table>

        <h2>Event-Related Potentials</h2>
        {erp_plots_html}

        <h2>Rejection Statistics</h2>
        <div class="info-box">
            <p><strong>Drop Rate:</strong> {drop_log_stats:.1f}% of epochs rejected</p>
            <p><strong>Criteria:</strong> {epochs.reject if hasattr(epochs, 'reject') else 'N/A'}</p>
        </div>

        <div class="info-box success">
            <p><strong>✓ Quality Check Complete</strong></p>
            <p>Epochs extracted successfully. Review ERP plots and trial counts above to verify data quality.</p>
        </div>
    </div>
</body>
</html>
    """

    return html


def _generate_erp_plots_html(epochs: mne.Epochs, conditions: List[str]) -> str:
    """Generate HTML with ERP plots for each condition.

    For now, returns placeholder. Full implementation would use matplotlib
    to generate plots and convert to base64 images.
    """

    plots_html = '<div class="info-box">'
    plots_html += '<p><em>ERP plots would be displayed here for each condition:</em></p>'
    plots_html += '<ul>'

    for condition in conditions:
        if condition in epochs.event_id:
            evoked = epochs[condition].average()
            n_channels = len(evoked.ch_names)
            times = evoked.times
            plots_html += f'<li><strong>{condition}</strong>: {len(epochs[condition])} trials, {n_channels} channels, {times[0]:.2f} to {times[-1]:.2f} s</li>'

    plots_html += '</ul>'
    plots_html += '<p><em>Note: To add actual ERP plots, install matplotlib and update this function.</em></p>'
    plots_html += '</div>'

    return plots_html
