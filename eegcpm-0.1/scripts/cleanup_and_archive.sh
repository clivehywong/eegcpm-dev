#!/bin/bash
# Cleanup and Archive Old Preprocessing Results
# This script archives old preprocessing attempts before implementing
# the new data quality detection system
#
# Author: EEGCPM Development Team
# Date: 2025-11-27

set -e  # Exit on error

DERIVATIVES="/Volumes/Work/data/hbn/derivatives"
ARCHIVE_DIR="/Volumes/Work/data/hbn/derivatives_archive_$(date +%Y%m%d_%H%M%S)"

echo "================================================================"
echo "EEGCPM Preprocessing Cleanup & Archive"
echo "================================================================"
echo ""
echo "This will:"
echo "  1. Archive old preprocessing results to: $ARCHIVE_DIR"
echo "  2. Clean derivatives directory"
echo "  3. Preserve raw-qc (keep for comparison)"
echo "  4. Preserve behavioral data (unaffected by preprocessing)"
echo ""
echo "Directories to archive:"
echo "  - pipeline-mne-bids/        (412M - old baseline run)"
echo "  - preprocessed-qc/          (168K - old QC)"
echo "  - preprocessing-batch/      (117M - old batch test)"
echo ""
echo "Total to archive: ~529M"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Creating archive directory..."
mkdir -p "$ARCHIVE_DIR"

echo "Step 2: Archiving old pipeline results..."
if [ -d "$DERIVATIVES/pipeline-mne-bids" ]; then
    echo "  - Moving pipeline-mne-bids/ (412M)..."
    mv "$DERIVATIVES/pipeline-mne-bids" "$ARCHIVE_DIR/"
fi

if [ -d "$DERIVATIVES/preprocessed-qc" ]; then
    echo "  - Moving preprocessed-qc/ (168K)..."
    mv "$DERIVATIVES/preprocessed-qc" "$ARCHIVE_DIR/"
fi

if [ -d "$DERIVATIVES/preprocessing-batch" ]; then
    echo "  - Moving preprocessing-batch/ (117M)..."
    mv "$DERIVATIVES/preprocessing-batch" "$ARCHIVE_DIR/"
fi

echo "Step 3: Creating archive manifest..."
cat > "$ARCHIVE_DIR/README.md" << 'EOF'
# Archived Preprocessing Results

**Archive Date**: $(date)
**Reason**: Implementing data quality detection system (flatline/bridged channels + rank-based ICA)

## Contents

### pipeline-mne-bids/
- **Config**: MNE-BIDS equivalent (RANSAC bad channels, extended-infomax ICA)
- **Subjects**: Last 10 from HBN dataset
- **Success Rate**: 3/10 (30%)
- **Issues**: 5 subjects failed with LAPACK "work array" error
- **Preprocessed**: 3 files (~300 MB total)
- **QC Reports**: 10 raw, 3 preprocessed

### preprocessed-qc/
- Old QC reports before API standardization
- Replaced by pipeline-mne-bids/qc/

### preprocessing-batch/
- Early batch preprocessing test
- Before MNE-BIDS configuration

## Why Archived

These results represent the baseline BEFORE implementing:
1. Flatline channel detection
2. Bridged channel detection
3. Rank-based ICA component selection
4. Comprehensive data quality QC

Expected improvement: 30% → 80-100% success rate

## Access

Results preserved for:
- Comparison with new implementation
- Documentation of improvement
- Troubleshooting reference

To restore:
```bash
cp -r $ARCHIVE_DIR/pipeline-mne-bids $DERIVATIVES/
```
EOF

echo "Step 4: Verifying archive..."
ls -lh "$ARCHIVE_DIR"

echo "Step 5: Checking remaining derivatives..."
echo ""
echo "Remaining in derivatives/:"
ls -lh "$DERIVATIVES/"

echo ""
echo "================================================================"
echo "Cleanup Complete!"
echo "================================================================"
echo ""
echo "Archived to: $ARCHIVE_DIR"
echo ""
echo "Preserved:"
echo "  ✓ raw-qc/        (5.1M - for comparison with new QC)"
echo "  ✓ behavioral/    (580K - unaffected)"
echo ""
echo "Ready for new implementation with:"
echo "  1. Data quality detection (flatline/bridged channels)"
echo "  2. Rank-based ICA"
echo "  3. Enhanced QC reports"
echo ""
