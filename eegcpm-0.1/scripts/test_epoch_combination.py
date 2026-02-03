#!/usr/bin/env python3
"""Test epoch combination on runs with accepted quality."""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from eegcpm.pipeline.run_processor import RunProcessor
from eegcpm.pipeline.epoch_combiner import EpochCombiner
from eegcpm.workflow.state import WorkflowStateManager


def test_epoch_combination():
    """Test combining epochs from multiple runs."""

    print("=" * 80)
    print("Testing Epoch Combination")
    print("=" * 80)

    # Paths
    bids_root = Path("/Volumes/Work/data/hbn/bids")
    output_root = Path("/Volumes/Work/data/hbn/derivatives/pipeline-asr-test")
    config_file = Path("/Users/clive/eegcpm/config/asr_pipeline_config.yaml")

    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Add epoching config if not present
    if 'epochs' not in config:
        config['epochs'] = {
            'tmin': -0.2,
            'tmax': 0.5,
            'baseline': (None, 0),
            'reject': None  # No automatic rejection for now
        }

    # Setup state manager
    state_db = output_root / ".eegcpm" / "state.db"
    state_manager = WorkflowStateManager(state_db)

    # Test on NDARBA404HR9 which has 2 accepted runs
    subject = "NDARBA404HR9"
    task = "saiit"
    session = "01"
    pipeline = "asr-test"

    print(f"\nSubject: {subject}")
    print(f"Task: {task}")
    print(f"Session: {session}")
    print("-" * 80)

    # First, run preprocessing to ensure we have the runs
    processor = RunProcessor(
        bids_root=bids_root,
        output_root=output_root,
        config=config,
        state_manager=state_manager
    )

    results = processor.process_subject_task(
        subject_id=subject,
        task=task,
        session=session,
        pipeline=pipeline
    )

    # Get recommendations
    recommendations = processor.get_run_selection_recommendations(results)

    print("\n" + "=" * 80)
    print("Run Quality Assessment")
    print("=" * 80)

    for result in results:
        if result.quality_metrics:
            m = result.quality_metrics
            status = "✓ ACCEPT" if recommendations.get(result.run) else "✗ REJECT"
            print(f"Run {result.run}: {status}")
            print(f"  Quality: {m.quality_status}")
            print(f"  Bad channels: {m.pct_bad_channels:.1f}%")
            print(f"  Clustering: {m.clustering_severity}")
            print(f"  Recommendation: {m.recommended_action}")

    # Combine accepted runs
    accepted_runs = [r for r in results if recommendations.get(r.run, False)]

    if len(accepted_runs) == 0:
        print("\n✗ No runs accepted for combination")
        return False

    print(f"\n{len(accepted_runs)} runs accepted for combination")

    # Create epoch combiner
    combined_output = output_root / subject / f"ses-{session}" / f"task-{task}" / "combined"
    combiner = EpochCombiner(
        config=config,
        output_dir=combined_output,
        state_manager=state_manager
    )

    # Combine epochs
    print("\n" + "=" * 80)
    print("Combining Epochs")
    print("=" * 80)

    combination_result = combiner.combine_with_selection(
        run_results=results,
        selection=recommendations,
        subject_id=subject,
        session=session,
        task=task,
        pipeline=pipeline
    )

    # Display results
    print("\n" + "=" * 80)
    print("Combination Results")
    print("=" * 80)

    if combination_result.success:
        print(f"✓ Success!")
        print(f"  Runs combined: {combination_result.n_runs_combined}")
        print(f"  Runs included: {', '.join(combination_result.runs_included)}")
        print(f"  Total epochs: {combination_result.n_total_epochs}")
        print(f"  Output file: {combination_result.output_path}")

        print(f"\n  Epochs per run (approximate):")
        for run_id, n_epochs in combination_result.n_epochs_per_run.items():
            print(f"    run-{run_id}: {n_epochs} epochs")

        # Check workflow state
        combined_state = state_manager.load_state(
            subject_id=subject,
            task=task,
            pipeline=pipeline,
            session=session,
            run=None  # Combined workflow
        )

        if combined_state:
            print(f"\n  Workflow state:")
            print(f"    Status: {combined_state.status.value}")
            print(f"    Steps completed: {len(combined_state.get_completed_steps())}")
            for step in combined_state.steps:
                print(f"      - {step.step_name}: {step.status.value}")
    else:
        print(f"✗ Failed: {combination_result.error}")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return combination_result.success


if __name__ == "__main__":
    success = test_epoch_combination()
    sys.exit(0 if success else 1)
