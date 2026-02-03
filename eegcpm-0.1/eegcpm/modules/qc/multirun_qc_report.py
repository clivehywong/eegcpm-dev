"""Unified Multi-Run QC Report Generator.

Generates a single HTML report containing QC results for all runs of a task,
with collapsible sections for detailed per-run reports.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from .base import QCResult


class MultiRunQCReport:
    """Generate unified QC report for multiple runs of a task.

    Creates a single HTML with:
    - Overview table showing summary metrics for all runs
    - Collapsible detailed sections for each run (hidden by default)
    - JavaScript for toggling visibility
    """

    # CSS for multi-run report
    CSS_STYLES = """
    * {
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        margin: 0;
        padding: 20px;
        background: #f9fafb;
        color: #1f2937;
        line-height: 1.6;
    }

    h1 {
        margin-top: 0;
        margin-bottom: 20px;
        font-size: 28px;
        color: #111827;
    }

    h2 {
        margin-top: 30px;
        margin-bottom: 15px;
        font-size: 20px;
        color: #111827;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 10px;
    }

    h3 {
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 16px;
        color: #374151;
    }

    /* Overview table */
    .overview-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        background: white;
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .overview-table th {
        background: #f3f4f6;
        padding: 12px;
        text-align: left;
        font-weight: 600;
        font-size: 13px;
        color: #374151;
        border-bottom: 1px solid #e5e7eb;
    }

    .overview-table td {
        padding: 12px;
        border-bottom: 1px solid #e5e7eb;
        font-size: 14px;
    }

    .overview-table tr:last-child td {
        border-bottom: none;
    }

    .overview-table tr:hover {
        background: #f9fafb;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }

    .status-badge.excellent {
        background: #dcfce7;
        color: #166534;
    }

    .status-badge.good {
        background: #dbeafe;
        color: #1e40af;
    }

    .status-badge.acceptable {
        background: #fef3c7;
        color: #92400e;
    }

    .status-badge.poor {
        background: #fee2e2;
        color: #991b1b;
    }

    .status-badge.ok {
        background: #dcfce7;
        color: #166534;
    }

    .status-badge.warning {
        background: #fef3c7;
        color: #92400e;
    }

    .status-badge.bad {
        background: #fee2e2;
        color: #991b1b;
    }

    /* Run sections */
    .run-section {
        margin: 30px 0;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        background: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .run-header {
        padding: 15px 20px;
        background: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        user-select: none;
    }

    .run-header:hover {
        background: #f3f4f6;
    }

    .run-header h3 {
        margin: 0;
        font-size: 16px;
        color: #111827;
    }

    .run-header .toggle-icon {
        font-size: 14px;
        color: #6b7280;
        transition: transform 0.2s;
    }

    .run-header .toggle-icon.expanded {
        transform: rotate(180deg);
    }

    .run-content {
        padding: 20px;
        display: none;
    }

    .run-content.expanded {
        display: block;
    }

    /* Summary stats boxes */
    .summary-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }

    .stat-box {
        background: white;
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .stat-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .stat-value {
        font-size: 24px;
        font-weight: 600;
        color: #111827;
    }

    .stat-unit {
        font-size: 14px;
        color: #6b7280;
        margin-left: 4px;
    }

    /* Clickable action text */
    .expand-all-link {
        color: #2563eb;
        text-decoration: none;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
    }

    .expand-all-link:hover {
        text-decoration: underline;
    }

    /* Embedded iframe styling */
    .run-content iframe {
        width: 100%;
        min-height: 800px;
        border: none;
        border-radius: 4px;
    }
    """

    # JavaScript for collapsible sections
    JS_SCRIPT = """
    function toggleRun(runId) {
        const content = document.getElementById('content-' + runId);
        const icon = document.getElementById('icon-' + runId);

        if (content.classList.contains('expanded')) {
            content.classList.remove('expanded');
            icon.classList.remove('expanded');
        } else {
            content.classList.add('expanded');
            icon.classList.add('expanded');
        }
    }

    function expandAll() {
        const contents = document.querySelectorAll('.run-content');
        const icons = document.querySelectorAll('.toggle-icon');

        contents.forEach(c => c.classList.add('expanded'));
        icons.forEach(i => i.classList.add('expanded'));
    }

    function collapseAll() {
        const contents = document.querySelectorAll('.run-content');
        const icons = document.querySelectorAll('.toggle-icon');

        contents.forEach(c => c.classList.remove('expanded'));
        icons.forEach(i => i.classList.remove('expanded'));
    }
    """

    def __init__(self, title: str = "Multi-Run QC Report"):
        """Initialize multi-run QC report builder.

        Parameters
        ----------
        title : str
            Report title
        """
        self.title = title
        self.run_summaries: List[Dict[str, Any]] = []
        self.run_reports: List[Dict[str, str]] = []  # [{run_id, html_content}, ...]

    def add_run(
        self,
        run_id: str,
        qc_result: QCResult,
        detailed_html: Optional[str] = None
    ):
        """Add a run's QC results to the report.

        Parameters
        ----------
        run_id : str
            Run identifier (e.g., "run-1")
        qc_result : QCResult
            QC result object with metrics
        detailed_html : str, optional
            Full detailed HTML report for this run (will be embedded in iframe)
        """
        # Extract summary metrics
        summary = {
            'run_id': run_id,
            'status': qc_result.status,
            'metrics': {}
        }

        # Extract key metrics for overview table
        for metric in qc_result.metrics:
            summary['metrics'][metric.name] = {
                'value': metric.value,
                'unit': metric.unit,
                'status': metric.status
            }

        self.run_summaries.append(summary)

        # Store detailed HTML if provided
        if detailed_html:
            self.run_reports.append({
                'run_id': run_id,
                'html': detailed_html
            })

    def build(self) -> str:
        """Build the unified multi-run HTML report.

        Returns
        -------
        str
            Complete HTML document
        """
        html = "<!DOCTYPE html>\n"
        html += '<html lang="en">\n'
        html += "<head>\n"
        html += '  <meta charset="UTF-8">\n'
        html += '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        html += f"  <title>{self._escape_html(self.title)}</title>\n"
        html += "  <style>\n"
        html += self.CSS_STYLES
        html += "  </style>\n"
        html += "</head>\n"
        html += "<body>\n"

        # Title
        html += f"<h1>{self._escape_html(self.title)}</h1>\n"

        # Summary statistics
        html += self._build_summary_stats()

        # Overview table
        html += "<h2>Run Overview</h2>\n"
        html += self._build_overview_table()

        # Expand/collapse controls
        html += '<p style="margin: 20px 0;">\n'
        html += '  <a href="javascript:void(0)" onclick="expandAll()" class="expand-all-link">Expand All</a>\n'
        html += '  | \n'
        html += '  <a href="javascript:void(0)" onclick="collapseAll()" class="expand-all-link">Collapse All</a>\n'
        html += '</p>\n'

        # Detailed run sections (collapsible)
        html += "<h2>Detailed Reports</h2>\n"
        html += self._build_run_sections()

        # JavaScript
        html += "<script>\n"
        html += self.JS_SCRIPT
        html += "</script>\n"

        html += "</body>\n"
        html += "</html>"

        return html

    def _build_summary_stats(self) -> str:
        """Build summary statistics boxes."""
        if not self.run_summaries:
            return ""

        total_runs = len(self.run_summaries)
        excellent = sum(1 for r in self.run_summaries if r['status'] == 'ok')
        warnings = sum(1 for r in self.run_summaries if r['status'] == 'warning')
        poor = sum(1 for r in self.run_summaries if r['status'] == 'bad')

        html = '<div class="summary-stats">\n'

        html += '  <div class="stat-box">\n'
        html += '    <div class="stat-label">Total Runs</div>\n'
        html += f'    <div class="stat-value">{total_runs}</div>\n'
        html += '  </div>\n'

        html += '  <div class="stat-box">\n'
        html += '    <div class="stat-label">Excellent/Good</div>\n'
        html += f'    <div class="stat-value">{excellent}<span class="stat-unit">({100*excellent//total_runs}%)</span></div>\n'
        html += '  </div>\n'

        html += '  <div class="stat-box">\n'
        html += '    <div class="stat-label">Warnings</div>\n'
        html += f'    <div class="stat-value">{warnings}<span class="stat-unit">({100*warnings//total_runs if total_runs > 0 else 0}%)</span></div>\n'
        html += '  </div>\n'

        html += '  <div class="stat-box">\n'
        html += '    <div class="stat-label">Poor/Failed</div>\n'
        html += f'    <div class="stat-value">{poor}<span class="stat-unit">({100*poor//total_runs if total_runs > 0 else 0}%)</span></div>\n'
        html += '  </div>\n'

        html += '</div>\n'

        return html

    def _build_overview_table(self) -> str:
        """Build overview table with key metrics for all runs."""
        if not self.run_summaries:
            return "<p>No run data available.</p>\n"

        html = '<table class="overview-table">\n'

        # Header
        html += "  <tr>\n"
        html += "    <th>Run</th>\n"
        html += "    <th>Status</th>\n"
        html += "    <th>Channels</th>\n"
        html += "    <th>Bad Channels</th>\n"
        html += "    <th>Duration</th>\n"
        html += "    <th>Bad Segments (%)</th>\n"
        html += "    <th>ICA Excluded</th>\n"
        html += "  </tr>\n"

        # Rows
        for summary in self.run_summaries:
            run_id = summary['run_id']
            status = summary['status']
            metrics = summary['metrics']

            html += "  <tr>\n"
            html += f'    <td><strong>{self._escape_html(run_id)}</strong></td>\n'
            html += f'    <td><span class="status-badge {status}">{status.upper()}</span></td>\n'

            # Channels
            n_channels = metrics.get('N EEG Channels', {}).get('value', '—')
            html += f'    <td>{n_channels}</td>\n'

            # Bad channels
            n_bad = metrics.get('Bad Channels', {}).get('value', '—')
            html += f'    <td>{n_bad}</td>\n'

            # Duration
            duration = metrics.get('Duration', {}).get('value', '—')
            if duration != '—':
                duration = f"{duration:.1f}s"
            html += f'    <td>{duration}</td>\n'

            # Bad segments
            bad_pct = metrics.get('Bad Segment %', {}).get('value', '—')
            if bad_pct != '—':
                bad_pct = f"{bad_pct:.1f}%"
            html += f'    <td>{bad_pct}</td>\n'

            # ICA excluded
            ica_excl = metrics.get('ICA Excluded', {}).get('value', '—')
            html += f'    <td>{ica_excl}</td>\n'

            html += "  </tr>\n"

        html += "</table>\n"

        return html

    def _build_run_sections(self) -> str:
        """Build collapsible sections for each run's detailed report."""
        if not self.run_reports:
            return "<p>No detailed reports available.</p>\n"

        html = ""

        for report in self.run_reports:
            run_id = report['run_id']
            report_html = report['html']

            # Find corresponding summary for status badge
            summary = next((s for s in self.run_summaries if s['run_id'] == run_id), None)
            status = summary['status'] if summary else 'unknown'

            html += f'<div class="run-section" id="section-{run_id}">\n'

            # Header (clickable)
            html += f'  <div class="run-header" onclick="toggleRun(\'{run_id}\')">\n'
            html += f'    <h3>{self._escape_html(run_id)} <span class="status-badge {status}">{status.upper()}</span></h3>\n'
            html += f'    <span class="toggle-icon" id="icon-{run_id}">▼</span>\n'
            html += '  </div>\n'

            # Content (hidden by default)
            html += f'  <div class="run-content" id="content-{run_id}">\n'
            html += f'    {report_html}\n'  # Embed the full HTML directly
            html += '  </div>\n'

            html += '</div>\n'

        return html

    def save(self, path: Path) -> Path:
        """Save the multi-run report to file.

        Parameters
        ----------
        path : Path
            Output file path

        Returns
        -------
        Path
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build())
        return path

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
