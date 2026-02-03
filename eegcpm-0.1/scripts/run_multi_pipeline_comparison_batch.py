#!/usr/bin/env python3
"""
Multi-Pipeline Preprocessing Comparison - Batch Processing

Runs multiple preprocessing pipelines on multiple subjects and generates
comprehensive comparison reports.

Features:
- Compares 5 different preprocessing pipelines
- Processes 10 subjects from HBN dataset
- Generates individual and summary comparison reports
- Tracks pipeline performance statistics

Usage:
    python scripts/run_multi_pipeline_comparison_batch.py

Author: EEGCPM Development Team
Date: 2025-12-01
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import mne
import yaml
import numpy as np
import pandas as pd

from eegcpm.modules.qc.multi_pipeline_comparison import (
    MultiPipelineComparison,
    create_pipeline_configs_from_yaml
)

# Configuration
BIDS_ROOT = Path("/Volumes/Work/data/hbn/bids")
CONFIG_PATH = Path(__file__).parent.parent / "config/multi_pipeline_comparison.yaml"
OUTPUT_ROOT = Path("/Volumes/Work/data/hbn/derivatives/multi-pipeline-comparison")

# Task to process
TASK = "task-saiit2afcblock1"


def get_available_subjects(bids_root: Path, task: str, n_subjects: int = 10) -> List[str]:
    """
    Get list of subjects with available data for the task.

    Args:
        bids_root: BIDS root directory
        task: Task name
        n_subjects: Number of subjects to return

    Returns:
        List of subject IDs
    """
    subjects = []

    for subject_dir in sorted(bids_root.glob("sub-*")):
        subject_id = subject_dir.name

        # Check if EEG file exists
        eeg_file = subject_dir / "ses-01/eeg" / f"{subject_id}_ses-01_{task}_eeg.fif"

        if eeg_file.exists():
            subjects.append(subject_id)

            if len(subjects) >= n_subjects:
                break

    return subjects


def process_subject(
    subject_id: str,
    task: str,
    pipelines: List,
    comparison_config: Dict,
    reference_channels: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Process a single subject through all pipelines.

    Args:
        subject_id: Subject identifier
        task: Task name
        pipelines: List of pipeline configurations
        comparison_config: Comparison QC configuration
        reference_channels: Channels to drop before processing
        output_dir: Output directory for this subject

    Returns:
        Dictionary with processing results and statistics
    """
    print("\n" + "=" * 80)
    print(f"Processing Subject: {subject_id}")
    print(f"Task: {task}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    subject_short = subject_id.replace('sub-', '')
    subject_output = output_dir / subject_short
    subject_output.mkdir(parents=True, exist_ok=True)

    # Load EEG data
    eeg_file = BIDS_ROOT / subject_id / "ses-01/eeg" / f"{subject_id}_ses-01_{task}_eeg.fif"

    if not eeg_file.exists():
        print(f"ERROR: File not found: {eeg_file}")
        return {'success': False, 'error': f'File not found: {eeg_file}'}

    print(f"\nLoading: {eeg_file.name}")
    raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    print(f"  Channels: {len(raw.ch_names)}, Duration: {raw.times[-1]:.1f}s, Sfreq: {raw.info['sfreq']} Hz")

    # Drop reference channels
    if reference_channels:
        print(f"  Dropping reference channels: {reference_channels}")
        raw.drop_channels(reference_channels)
        print(f"  Channels after reference drop: {len(raw.ch_names)}")

    # Keep copy of original
    raw_original = raw.copy()

    # Initialize multi-pipeline comparison
    comparison = MultiPipelineComparison(
        pipelines=pipelines,
        output_dir=subject_output,
        comparison_config=comparison_config
    )

    # Run all pipelines
    results = comparison.run_all_pipelines(
        raw_original=raw_original,
        subject_id=subject_short,
        task=task.replace('task-', '')
    )

    # Generate comparison report
    html_path = comparison.generate_comparison_report(
        raw_original=raw_original,
        results=results,
        subject_id=subject_short,
        task=task.replace('task-', '')
    )

    # Collect statistics
    stats = {
        'subject_id': subject_short,
        'task': task,
        'success': True,
        'n_pipelines': len(pipelines),
        'n_successful': sum(1 for r in results if r.success),
        'n_failed': sum(1 for r in results if not r.success),
        'html_report': str(html_path) if html_path else None,
        'pipeline_results': []
    }

    for result in results:
        pipeline_stats = {
            'name': result.name,
            'type': result.pipeline_type,
            'success': result.success,
            'execution_time_s': result.execution_time_s,
        }

        if result.success:
            pipeline_stats.update({
                'n_channels_final': len(result.raw_processed.ch_names),
                'n_channels_removed': len(raw_original.ch_names) - len(result.raw_processed.ch_names),
                'duration_final_s': result.raw_processed.times[-1],
            })

            # Add specific stats from result.stats
            for key in ['ica_components', 'ica_components_removed', 'data_removed_pct']:
                if key in result.stats:
                    pipeline_stats[key] = result.stats[key]
        else:
            pipeline_stats['error'] = result.error

        stats['pipeline_results'].append(pipeline_stats)

    # Save subject statistics
    stats_file = subject_output / f"{subject_short}_{task.replace('task-', '')}_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Subject {subject_short} complete")
    print(f"  Successful pipelines: {stats['n_successful']}/{stats['n_pipelines']}")
    print(f"  Statistics saved: {stats_file}")
    if html_path:
        print(f"  HTML report: {html_path}")

    return stats


def generate_summary_report(all_stats: List[Dict], output_dir: Path):
    """
    Generate summary report across all subjects with two-panel layout.

    Args:
        all_stats: List of statistics dictionaries from all subjects
        output_dir: Output directory for summary
    """
    print("\n" + "=" * 80)
    print("Generating Summary Report")
    print("=" * 80)

    # Convert to DataFrame for analysis
    records = []
    for subject_stats in all_stats:
        if not subject_stats.get('success'):
            continue

        for pipeline_result in subject_stats['pipeline_results']:
            record = {
                'subject_id': subject_stats['subject_id'],
                'task': subject_stats['task'],
                'pipeline_name': pipeline_result['name'],
                'pipeline_type': pipeline_result['type'],
                'success': pipeline_result['success'],
                'execution_time_s': pipeline_result.get('execution_time_s', 0),
            }

            if pipeline_result['success']:
                record.update({
                    'n_channels_final': pipeline_result.get('n_channels_final', 0),
                    'n_channels_removed': pipeline_result.get('n_channels_removed', 0),
                    'duration_final_s': pipeline_result.get('duration_final_s', 0),
                    'ica_components': pipeline_result.get('ica_components', 0),
                    'ica_components_removed': pipeline_result.get('ica_components_removed', 0),
                    'data_removed_pct': pipeline_result.get('data_removed_pct', 0),
                })

            records.append(record)

    df = pd.DataFrame(records)

    # Save full results
    csv_path = output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Full results saved: {csv_path}")

    # Compute summary statistics per pipeline
    summary_stats = []

    for pipeline_name in df['pipeline_name'].unique():
        pipeline_df = df[df['pipeline_name'] == pipeline_name]
        successful = pipeline_df[pipeline_df['success'] == True]

        if len(successful) == 0:
            continue

        stats = {
            'pipeline_name': pipeline_name,
            'pipeline_type': successful['pipeline_type'].iloc[0],
            'n_subjects_total': len(pipeline_df),
            'n_subjects_success': len(successful),
            'success_rate_pct': 100 * len(successful) / len(pipeline_df),
            'mean_execution_time_s': successful['execution_time_s'].mean(),
            'std_execution_time_s': successful['execution_time_s'].std(),
            'mean_channels_removed': successful['n_channels_removed'].mean(),
            'std_channels_removed': successful['n_channels_removed'].std(),
            'mean_ica_components': successful['ica_components'].mean() if 'ica_components' in successful else 0,
            'mean_ica_removed': successful['ica_components_removed'].mean() if 'ica_components_removed' in successful else 0,
            'mean_data_removed_pct': successful['data_removed_pct'].mean() if 'data_removed_pct' in successful else 0,
        }

        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    # Save summary statistics
    summary_csv = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary statistics saved: {summary_csv}")

    # Print summary table
    print("\nPipeline Summary:")
    print(summary_df.to_string(index=False))

    # Generate summary content HTML
    summary_content = f"""
    <div class="summary-content">
        <h2>Overview</h2>
        <p>Processed {len(all_stats)} subjects with {len(df['pipeline_name'].unique())} pipelines</p>
        <p>Task: {all_stats[0]['task']}</p>
        <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Pipeline Statistics</h2>
        {summary_df.to_html(index=False, classes='table')}
    </div>
    """

    # Generate two-panel HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Multi-Pipeline Comparison Summary - {len(all_stats)} Subjects</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            height: 100vh;
            overflow: hidden;
        }}

        .container {{
            display: flex;
            height: 100vh;
        }}

        .left-panel {{
            width: 300px;
            background-color: #f5f5f5;
            border-right: 1px solid #ddd;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }}

        .right-panel {{
            flex: 1;
            overflow: auto;
            padding: 20px;
        }}

        .nav-item {{
            padding: 12px 20px;
            cursor: pointer;
            border-bottom: 1px solid #ddd;
            transition: background-color 0.2s;
        }}

        .nav-item:hover {{
            background-color: #e8e8e8;
        }}

        .nav-item.active {{
            background-color: #007bff;
            color: white;
        }}

        .nav-item.summary {{
            font-weight: bold;
            background-color: #28a745;
            color: white;
        }}

        .nav-item.summary:hover {{
            background-color: #218838;
        }}

        .table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}

        .table th, .table td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}

        .table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}

        .table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}

        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}

        h2 {{
            margin-top: 20px;
            margin-bottom: 10px;
            color: #333;
        }}

        p {{
            margin: 5px 0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <div class="nav-item summary active" onclick="showSummary()">
                📊 Summary
            </div>
"""

    # Add subject links
    for subject_stats in all_stats:
        if subject_stats.get('html_report'):
            subject_id = subject_stats['subject_id']
            html_report = Path(subject_stats['html_report'])
            if html_report.exists():
                rel_path = html_report.relative_to(output_dir)
                html_content += f"""
            <div class="nav-item" onclick="showReport('{rel_path}', this)">
                {subject_id}
            </div>
"""

    html_content += f"""
        </div>
        <div class="right-panel" id="content-panel">
            {summary_content}
        </div>
    </div>

    <script>
        function showSummary() {{
            const contentPanel = document.getElementById('content-panel');
            contentPanel.innerHTML = `{summary_content}`;

            // Update active state
            document.querySelectorAll('.nav-item').forEach(item => {{
                item.classList.remove('active');
            }});
            document.querySelector('.nav-item.summary').classList.add('active');
        }}

        function showReport(reportPath, element) {{
            const contentPanel = document.getElementById('content-panel');
            contentPanel.innerHTML = `<iframe src="${{reportPath}}"></iframe>`;

            // Update active state
            document.querySelectorAll('.nav-item').forEach(item => {{
                item.classList.remove('active');
            }});
            element.classList.add('active');
        }}
    </script>
</body>
</html>
"""

    # Save summary HTML
    summary_html_path = output_dir / "summary_report.html"
    with open(summary_html_path, 'w') as f:
        f.write(html_content)

    print(f"✓ Summary HTML saved: {summary_html_path}")


def main():
    """Run multi-pipeline comparison on batch of subjects."""

    print("=" * 80)
    print("Multi-Pipeline Preprocessing Comparison - Batch Processing")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load configuration
    print(f"Loading configuration: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Parse pipeline configurations
    pipelines = create_pipeline_configs_from_yaml(config)
    print(f"Configured {len(pipelines)} pipelines:")
    for pipeline in pipelines:
        status = "✓ Enabled" if pipeline.enabled else "✗ Disabled"
        print(f"  {status}: {pipeline.name} ({pipeline.pipeline_type}) - {pipeline.description}")

    # Get batch settings
    batch_config = config.get('batch', {})
    n_subjects = batch_config.get('n_subjects', 10)
    reference_channels = batch_config.get('reference_channels', [])

    # Get comparison QC settings
    comparison_config = config.get('comparison_qc', {})

    # Get subjects
    print(f"\nFinding subjects with {TASK} data...")
    subjects = get_available_subjects(BIDS_ROOT, TASK, n_subjects)
    print(f"Found {len(subjects)} subjects: {', '.join([s.replace('sub-', '') for s in subjects])}")

    if len(subjects) == 0:
        print("ERROR: No subjects found")
        return

    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Process all subjects
    all_stats = []

    for i, subject_id in enumerate(subjects, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Subject {i}/{len(subjects)}: {subject_id}")
        print(f"{'#'*80}")

        try:
            stats = process_subject(
                subject_id=subject_id,
                task=TASK,
                pipelines=pipelines,
                comparison_config=comparison_config,
                reference_channels=reference_channels,
                output_dir=OUTPUT_ROOT
            )
            all_stats.append(stats)

        except Exception as e:
            print(f"\nERROR processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()

            all_stats.append({
                'subject_id': subject_id.replace('sub-', ''),
                'task': TASK,
                'success': False,
                'error': str(e)
            })

    # Generate summary report
    generate_summary_report(all_stats, OUTPUT_ROOT)

    # Save overall statistics
    overall_stats = {
        'processing_date': datetime.now().isoformat(),
        'task': TASK,
        'n_subjects_total': len(subjects),
        'n_subjects_success': sum(1 for s in all_stats if s.get('success')),
        'n_subjects_failed': sum(1 for s in all_stats if not s.get('success')),
        'pipelines': [p.name for p in pipelines],
        'subject_statistics': all_stats
    }

    overall_file = OUTPUT_ROOT / "overall_statistics.json"
    with open(overall_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Subjects processed: {overall_stats['n_subjects_success']}/{overall_stats['n_subjects_total']}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Summary report: {OUTPUT_ROOT / 'summary_report.html'}")
    print()


if __name__ == "__main__":
    main()
