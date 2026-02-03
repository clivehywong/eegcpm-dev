#!/usr/bin/env python3
"""Test run-level preprocessing on HBN data."""

import sys
from pathlib import Path
import yaml
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from eegcpm.pipeline.run_processor import RunProcessor
from eegcpm.workflow.state import WorkflowStateManager


def test_run_processing():
    """Test run-level preprocessing."""

    print("=" * 80)
    print("Testing Run-Level Preprocessing")
    print("=" * 80)

    # Paths
    bids_root = Path("/Volumes/Work/data/hbn/bids")
    output_root = Path("/Volumes/Work/data/hbn/derivatives/pipeline-run-test")
    config_file = Path("/Users/clive/eegcpm/config/cli_test_config.yaml")

    if not bids_root.exists():
        print(f"Error: BIDS root not found: {bids_root}")
        return False

    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Setup state manager
    state_db = output_root / ".eegcpm" / "state.db"
    state_db.parent.mkdir(parents=True, exist_ok=True)
    state_manager = WorkflowStateManager(state_db)

    # Initialize processor
    processor = RunProcessor(
        bids_root=bids_root,
        output_root=output_root,
        config=config,
        state_manager=state_manager
    )

    # Test on one subject with multiple runs
    subject = "NDARAB678VYW"
    task = "saiit"

    print(f"\nProcessing {subject} task {task}")
    print("-" * 80)

    results = processor.process_subject_task(
        subject_id=subject,
        task=task,
        session="01",
        pipeline="run-test"
    )

    print("\n" + "=" * 80)
    print("Run Processing Results")
    print("=" * 80)

    for result in results:
        print(f"\nRun {result.run}:")
        print(f"  Success: {result.success}")

        if result.quality_metrics:
            metrics = result.quality_metrics
            print(f"  Quality: {metrics.quality_status}")
            print(f"  Recommendation: {metrics.recommended_action}")
            print(f"  Bad channels: {metrics.n_bad_channels}/{metrics.n_original_channels} ({metrics.pct_bad_channels:.1f}%)")
            print(f"  Clustering: {metrics.clustering_severity} ({metrics.pct_clustered:.1f}% clustered)")
            print(f"  ICA: {'Success' if metrics.ica_success else 'Failed'} ({metrics.n_ica_components_rejected} components rejected)")

            if metrics.warnings:
                print(f"  Warnings:")
                for warning in metrics.warnings:
                    print(f"    - {warning}")

            if result.qc_path:
                print(f"  QC report: {result.qc_path}")

        if result.error:
            print(f"  Error: {result.error}")

    # Get recommendations
    print("\n" + "=" * 80)
    print("Run Selection Recommendations")
    print("=" * 80)

    recommendations = processor.get_run_selection_recommendations(results)

    print("\nAutomatic recommendations:")
    for run, should_include in recommendations.items():
        status = "✓ INCLUDE" if should_include else "✗ EXCLUDE"
        print(f"  run-{run}: {status}")

    # Save summary
    summary_file = output_root / subject / f"ses-01" / f"task-{task}" / "run_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'subject': subject,
        'task': task,
        'session': '01',
        'n_runs': len(results),
        'n_successful': sum(1 for r in results if r.success),
        'n_recommended': sum(1 for include in recommendations.values() if include),
        'runs': [r.quality_metrics.to_dict() if r.quality_metrics else {'run': r.run, 'error': r.error} for r in results],
        'recommendations': recommendations,
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")

    # Check workflow state
    print("\n" + "=" * 80)
    print("Workflow State")
    print("=" * 80)

    states = state_manager.get_all_states(subject_id=subject, task=task)
    print(f"\nFound {len(states)} workflow records:")
    for state in states:
        run_label = f"run-{state.run}" if state.run else "combined"
        print(f"  {run_label}: {state.status.value} ({len(state.get_completed_steps())}/{len(state.steps)} steps)")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_run_processing()
    sys.exit(0 if success else 1)
