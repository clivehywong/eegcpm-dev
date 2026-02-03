#!/usr/bin/env python3
"""
Batch processing script for comparing clustering strategies on multiple subjects.

Uses the multi-pipeline comparison framework to test different clustering actions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import mne
from eegcpm.modules.qc.multi_pipeline_comparison import (
    MultiPipelineComparison,
    create_pipeline_configs_from_yaml
)

# Configuration
# IMPORTANT: Modify these paths to match your system
BIDS_ROOT = Path("/Volumes/Work/data/hbn/bids")
CONFIG_PATH = Path(__file__).parent.parent / "config/clustering_comparison.yaml"
OUTPUT_ROOT = Path("/Volumes/Work/data/hbn/derivatives/clustering-comparison")
TASK = "task-saiit2afcblock1"

# Check if BIDS_ROOT exists
if not BIDS_ROOT.exists():
    print(f"ERROR: BIDS_ROOT does not exist: {BIDS_ROOT}")
    print("\nPlease modify the paths in this script to match your system:")
    print(f"  Line 17: BIDS_ROOT = Path('/Volumes/Work/data/hbn/bids')")
    print(f"  Line 19: OUTPUT_ROOT = Path('/Volumes/Work/data/hbn/derivatives/clustering-comparison')")
    print("\nOr mount the /Volumes/Work drive if you're using an external drive.")
    sys.exit(1)

# Subjects to process (you can modify this list)
SUBJECTS = [
    "sub-NDARAB678VYW",
    "sub-NDARAC296UCB",
    "sub-NDARAK770XEW",
    "sub-NDARAP457WB5",
    "sub-NDARAR358RHK",
    "sub-NDARBA404HR9",
    "sub-NDARBF851NH6",
    "sub-NDARCD401HGZ",
    "sub-NDARCT889DMB",
    "sub-NDARAA306NT2",
]


def main():
    """Run clustering comparison on multiple subjects."""

    print("=" * 80)
    print("Clustering Strategy Comparison - Batch Processing")
    print("=" * 80)

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Create pipeline configs
    pipelines = create_pipeline_configs_from_yaml(config)

    print(f"\nPipelines to test ({len(pipelines)}):")
    for p in pipelines:
        print(f"  - {p.name}: {p.description}")

    print(f"\nSubjects to process: {len(SUBJECTS)}")
    print(f"Output directory: {OUTPUT_ROOT}")

    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Initialize comparison
    comparison = MultiPipelineComparison(
        pipelines=pipelines,
        output_dir=OUTPUT_ROOT,
        comparison_config=config.get('comparison_qc', {})
    )

    # Process each subject
    all_stats = []

    for i, subject_id in enumerate(SUBJECTS, 1):
        print(f"\n{'='*80}")
        print(f"Subject {i}/{len(SUBJECTS)}: {subject_id}")
        print(f"{'='*80}")

        # Load data
        eeg_file = BIDS_ROOT / subject_id / "ses-01/eeg" / f"{subject_id}_ses-01_{TASK}_eeg.fif"

        if not eeg_file.exists():
            print(f"  ⚠️  File not found: {eeg_file}")
            continue

        try:
            raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)

            # Drop reference channel
            ref_channels = config.get('batch', {}).get('reference_channels', ['E129'])
            raw.drop_channels(ref_channels, on_missing='ignore')

            print(f"  Loaded: {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")

            # Run all pipelines on this subject
            results = comparison.run_all_pipelines(
                raw_original=raw,
                subject_id=subject_id.replace('sub-', ''),
                task=TASK.replace('task-', '')
            )

            # Extract statistics
            for result in results:
                stats = {
                    'subject_id': subject_id.replace('sub-', ''),
                    'pipeline_name': result.name,
                    'pipeline_type': result.pipeline_type,
                    'success': result.success,
                    'execution_time_s': result.execution_time_s,
                    'error': result.error if not result.success else None,
                }

                if result.success:
                    stats.update({
                        'n_channels_final': len(result.raw_processed.ch_names),
                        'n_channels_removed': result.stats.get('n_channels_removed', 0),
                        'n_bad_detected': result.stats.get('n_bad_channels', 0),
                        'n_clustered': result.stats.get('n_clustered_channels', 0),
                        'pct_clustered': result.stats.get('pct_clustered', 0),
                        'clustering_severity': result.stats.get('clustering_severity', 'unknown'),
                        'n_dropped_clustered': result.stats.get('n_dropped_clustered', 0),
                        'n_interpolated': result.stats.get('n_interpolated', 0),
                    })

                all_stats.append(stats)

            # Display summary for this subject
            print(f"\n  Results:")
            for result in results:
                status = "✓" if result.success else "✗"
                if result.success:
                    n_ch = len(result.raw_processed.ch_names)
                    n_clustered = result.stats.get('n_clustered_channels', 0)
                    n_dropped_clust = result.stats.get('n_dropped_clustered', 0)
                    print(f"    {status} {result.name:25s} → {n_ch:3d} ch "
                          f"(clustered: {n_clustered:2d}, dropped: {n_dropped_clust:2d})")
                else:
                    print(f"    {status} {result.name:25s} → Failed: {result.error}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary report
    print(f"\n{'='*80}")
    print("Generating Summary Report")
    print(f"{'='*80}")

    comparison.generate_summary_report(all_stats)

    print(f"\n✓ Batch processing complete!")
    print(f"\nOutput directory: {OUTPUT_ROOT}")
    print(f"Summary report: {OUTPUT_ROOT / 'summary_report.html'}")


if __name__ == "__main__":
    main()
