#!/usr/bin/env python3
"""Generate aggregate QC report across all subjects."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eegcpm.pipeline.aggregate_reports import AggregateReportGenerator


def main():
    """Generate aggregate report."""

    print("=" * 80)
    print("Generating Aggregate QC Report")
    print("=" * 80)

    # Paths
    derivatives_root = Path("/Volumes/Work/data/hbn/derivatives")
    pipeline_name = "pipeline-asr-test"

    # Create generator
    generator = AggregateReportGenerator(derivatives_root, pipeline_name)

    # Scan all subjects
    print("\nScanning subjects...")
    summaries = generator.scan_subjects()

    print(f"  Found {len(summaries)} processed subjects")

    if len(summaries) == 0:
        print("  No processed subjects found!")
        return False

    # Print summary statistics
    print("\nSummary Statistics:")
    total_runs = sum(s.n_runs_processed for s in summaries)
    total_accepted = sum(s.n_runs_accepted for s in summaries)
    total_combined = sum(1 for s in summaries if s.combined_epochs is not None)

    print(f"  Total runs processed: {total_runs}")
    print(f"  Total runs accepted: {total_accepted}")
    print(f"  Subjects with combined epochs: {total_combined}")

    # Quality breakdown
    quality_totals = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}
    for summary in summaries:
        for quality, count in summary.quality_summary.items():
            quality_totals[quality] += count

    print(f"\n  Run quality breakdown:")
    for quality, count in sorted(quality_totals.items()):
        if count > 0:
            print(f"    {quality.capitalize()}: {count}")

    # Generate index HTML
    print("\nGenerating index page...")
    index_path = generator.generate_index(summaries)
    print(f"  ✓ Index: {index_path}")

    # Generate summary JSON
    print("\nGenerating summary JSON...")
    json_path = generator.generate_summary_json(summaries)
    print(f"  ✓ JSON: {json_path}")

    print("\n" + "=" * 80)
    print("Aggregate Report Generated Successfully!")
    print("=" * 80)
    print(f"\nOpen in browser: file://{index_path}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
