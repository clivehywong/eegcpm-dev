"""
Preprocessing Module for EEG Data

Provides a comprehensive, config-driven preprocessing pipeline following industry standards
(MNE-Python, PREP, ASR best practices):

1. Set montage (channel locations)
2. Drop extreme channels (flatline, bridged, high variance)
3. Filter (bandpass + optional notch) - EARLY filtering for stationary data
4. Resample (optional downsampling)
5. Annotate artifacts (mark bad segments)
6. Detect bad channels (RANSAC on filtered data, mark only)
7. ASR (Artifact Subspace Reconstruction - repairs high-amplitude artifacts)
8. Interpolate bad channels (AFTER ASR to avoid artifacts)
9. Re-reference (average reference on complete channel set)
10. ICA (Independent Component Analysis - removes residual artifacts)

Key improvements:
- Filter BEFORE bad channel detection (RANSAC needs stationary data)
- Defer interpolation until AFTER ASR (prevents NaN/Inf artifacts)
- Re-reference AFTER interpolation (needs complete channel set)

All steps are configurable via YAML/JSON configuration files or dictionaries.

Author: EEGCPM Development Team
Created: 2025-01
Updated: 2025-12 (Pipeline order refactored to industry standards)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np

from eegcpm.pipeline.base import BaseModule, ModuleResult, RawDataModule

# Import preprocessing submodules
from .referencing import EEGReferencing, apply_reference_from_config
from .bad_channels import BadChannelDetector, detect_and_interpolate_bad_channels
from .artifacts import ArtifactAnnotator, annotate_artifacts_from_config
from .channel_clustering import compute_bad_channel_clustering


class PreprocessingModule(RawDataModule):
    """
    Comprehensive EEG preprocessing pipeline with config-driven processing.

    This module provides a complete preprocessing workflow that can be fully
    controlled via YAML/JSON configuration files. Each step is optional and
    can be enabled/disabled independently.

    Pipeline Order (Industry Standard):
    1. Set montage (channel locations for topographic plots and source localization)
    2. Drop extreme channels (flatline, bridged, high variance)
    3. **Filter EARLY** (highpass 0.5-1 Hz, lowpass ~40 Hz) - stationary data for detection
    4. Resample (optional downsampling for faster processing)
    5. Annotate artifacts (mark bad segments for ICA exclusion)
    6. Detect bad channels (RANSAC on filtered data, **mark only, don't interpolate**)
    7. ASR (Artifact Subspace Reconstruction - repairs high-amplitude artifacts)
    8. **Interpolate bad channels** (AFTER ASR to avoid creating NaN/Inf artifacts)
    9. Re-reference (common average on complete channel set)
    10. ICA (Independent Component Analysis - removes residual eye/muscle/heart artifacts)

    Parameters
    ----------
    config : dict
        Configuration dictionary with preprocessing parameters.
        Can be loaded from YAML file using yaml.safe_load().
        See config/preprocessing_config.yaml for full specification.
    output_dir : Path
        Directory for saving preprocessed data and ICA files.

    Examples
    --------
    Using default configuration:
    >>> config = {'l_freq': 0.5, 'h_freq': 40.0}
    >>> module = PreprocessingModule(config, Path("output/"))
    >>> result = module.process(raw, subject=subject)

    Using YAML configuration:
    >>> import yaml
    >>> with open("config/preprocessing_config.yaml") as f:
    ...     config = yaml.safe_load(f)
    >>> module = PreprocessingModule(config, Path("output/"))
    >>> result = module.process(raw)

    With full pipeline configuration:
    >>> config = {
    ...     'montage': {'enabled': True, 'type': 'standard_1020'},
    ...     'bad_channels': {
    ...         'auto_detect': True,
    ...         'methods': ['variance', 'correlation'],
    ...         'interpolate': True
    ...     },
    ...     'artifact_detection': {
    ...         'enabled': True,
    ...         'amplitude_threshold': 150e-6
    ...     },
    ...     'reference': {'type': 'average'},
    ...     'filter': {'l_freq': 0.5, 'h_freq': 40.0},
    ...     'ica': {'enabled': True, 'method': 'infomax'}
    ... }
    >>> module = PreprocessingModule(config, Path("output/"))
    """

    name = "preprocessing"
    version = "0.2.0"
    description = "Comprehensive EEG preprocessing with config-driven pipeline"

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize preprocessing module with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary. Supports both flat and nested formats:
            - Flat: {'l_freq': 0.5, 'h_freq': 40.0, 'ica_method': 'infomax'}
            - Nested: {'filter': {'l_freq': 0.5}, 'ica': {'method': 'infomax'}}
        output_dir : Path
            Output directory for preprocessed files
        """
        super().__init__(config, output_dir)

        # Store full config for nested access
        self._config = config

        # Parse configuration (supports both flat and nested formats)
        self._parse_config(config)

    def _parse_config(self, config: Dict[str, Any]) -> None:
        """
        Parse configuration dictionary into module parameters.

        Supports both flat format (legacy) and nested format (recommended).
        """
        # Montage configuration
        montage_cfg = config.get('montage', {})
        self.montage_enabled = montage_cfg.get('enabled', True) if isinstance(montage_cfg, dict) else True
        self.montage_type = montage_cfg.get('type', 'standard_1020') if isinstance(montage_cfg, dict) else 'standard_1020'
        self.montage_file = montage_cfg.get('file', None) if isinstance(montage_cfg, dict) else None

        # Bad channel configuration
        bad_cfg = config.get('bad_channels', {})
        self.bad_channels_config = {
            'auto_detect': bad_cfg.get('auto_detect', True),
            'methods': bad_cfg.get('methods', ['variance', 'correlation']),
            'variance_threshold': bad_cfg.get('variance_threshold', 5.0),
            'correlation_threshold': bad_cfg.get('correlation_threshold', 0.4),
            'ransac_sample_prop': bad_cfg.get('ransac_sample_prop', 0.25),
            'ransac_corr_threshold': bad_cfg.get('ransac_corr_threshold', 0.75),
            'deviation_threshold': bad_cfg.get('deviation_threshold', 5.0),
            'manual_bads': bad_cfg.get('manual_bads', []),
            'interpolate': bad_cfg.get('interpolate', True),
            'max_bad_channel_percent': bad_cfg.get('max_bad_channel_percent', 20.0),

            # Clustering detection and handling
            'check_clustering': bad_cfg.get('check_clustering', True),
            'clustering_action': bad_cfg.get('clustering_action', 'adaptive'),  # 'adaptive', 'interpolate_all', 'drop_clustered', 'fail_on_severe'
            'clustering_thresholds': {
                'moderate': bad_cfg.get('clustering_thresholds', {}).get('moderate', 20.0),
                'severe': bad_cfg.get('clustering_thresholds', {}).get('severe', 50.0),
            },
            'clustering_k_neighbors': bad_cfg.get('clustering_k_neighbors', 6),
            'clustering_neighbor_threshold': bad_cfg.get('clustering_neighbor_threshold', 0.5),
        }

        # Artifact detection configuration
        # Default thresholds for peak-to-peak amplitude in 100ms windows:
        # - amplitude_threshold: 500 µV (conservative, catches obvious artifacts)
        #   NOTE: Some datasets (e.g., HBN) may have scaling issues. If you see
        #   extreme amplitude values in QC reports (>1000 µV), check data import scaling.
        # - gradient_threshold: None (disabled by default, sample-to-sample change)
        # - flatline_duration: 5 seconds (catches disconnected electrodes)
        artifact_cfg = config.get('artifact_detection', {})
        self.artifact_config = {
            'enabled': artifact_cfg.get('enabled', True),
            'amplitude_threshold': artifact_cfg.get('amplitude_threshold', 500e-6),  # 500 µV default
            'gradient_threshold': artifact_cfg.get('gradient_threshold', None),      # Disabled by default
            'flatline_duration': artifact_cfg.get('flatline_duration', 5.0),         # 5 seconds default
            'muscle_threshold': artifact_cfg.get('muscle_threshold', None),          # Disabled by default
            'freq_muscle': artifact_cfg.get('freq_muscle', (110, 140)),
            'min_duration': artifact_cfg.get('min_duration', 0.1),
        }

        # Reference configuration
        ref_cfg = config.get('reference', {})
        self.reference_config = {
            'type': ref_cfg.get('type', 'average'),
            'channels': ref_cfg.get('channels', None),
            'projection': ref_cfg.get('projection', True),
            'exclude_bads': ref_cfg.get('exclude_bads', True),
        }

        # Resample configuration
        resample_cfg = config.get('resample', {})
        self.resample_enabled = resample_cfg.get('enabled', False) if isinstance(resample_cfg, dict) else False
        self.resample_sfreq = resample_cfg.get('sfreq', 250) if isinstance(resample_cfg, dict) else 250

        # Filter configuration (supports both flat and nested)
        filter_cfg = config.get('filter', {})
        if isinstance(filter_cfg, dict) and filter_cfg:
            # Nested format: config['filter']['l_freq']
            self.l_freq = filter_cfg.get('l_freq', 0.5)
            self.h_freq = filter_cfg.get('h_freq', 40.0)
            self.notch_freq = filter_cfg.get('notch_freq', None)
        else:
            # Flat format: config['l_freq']
            self.l_freq = config.get('l_freq', 0.5)
            self.h_freq = config.get('h_freq', 40.0)
            self.notch_freq = config.get('notch_freq', None)

        # ICA configuration (supports both flat and nested)
        ica_cfg = config.get('ica', {})
        if isinstance(ica_cfg, dict) and ica_cfg:
            # Nested format: config['ica']['method']
            self.ica_enabled = ica_cfg.get('enabled', True)
            self.ica_method = ica_cfg.get('method', 'infomax')
            self.ica_n_components = ica_cfg.get('n_components', None)
            self.ica_auto_detect = ica_cfg.get('auto_detect_artifacts', True)
            self.ica_manual_exclude = ica_cfg.get('manual_exclude', [])
            self.use_iclabel = ica_cfg.get('use_iclabel', False)
            self.iclabel_threshold = ica_cfg.get('iclabel_threshold', 0.8)

            # Parse method for extended-infomax support
            # MNE uses method='infomax' with fit_params=dict(extended=True) for extended-infomax
            self.ica_fit_params = {}
            if self.ica_method == 'extended-infomax':
                self.ica_method = 'infomax'
                self.ica_fit_params = {'extended': True}
            elif self.ica_method == 'extended_infomax':  # Also accept underscore variant
                self.ica_method = 'infomax'
                self.ica_fit_params = {'extended': True}
        else:
            # Flat format: config['ica_method']
            self.ica_enabled = True
            self.ica_method = config.get('ica_method', 'infomax')
            self.ica_n_components = config.get('ica_n_components', None)
            self.ica_auto_detect = True
            self.ica_manual_exclude = []
            self.use_iclabel = False
            self.iclabel_threshold = 0.8
            self.ica_fit_params = {}

        # ASR configuration
        asr_cfg = config.get('asr', {})
        if isinstance(asr_cfg, dict):
            self.use_asr = asr_cfg.get('enabled', False)
            self.asr_cutoff = asr_cfg.get('cutoff', 20.0)
            self.asr_train_duration = asr_cfg.get('train_duration', 60)
        else:
            # Flat format fallback
            self.use_asr = config.get('use_asr', False)
            self.asr_cutoff = config.get('asr_cutoff', 20.0)
            self.asr_train_duration = 60

    def process(
        self,
        data: mne.io.BaseRaw,
        subject: Optional[Any] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Run the complete preprocessing pipeline.

        Parameters
        ----------
        data : mne.io.BaseRaw
            Raw EEG data to preprocess
        subject : Subject, optional
            Subject object with ID and metadata
        **kwargs : dict
            Additional keyword arguments (e.g., session_id, task)

        Returns
        -------
        ModuleResult
            Contains preprocessed Raw, ICA object, and metadata including:
            - Number of bad channels detected/interpolated
            - Number of artifact annotations added
            - ICA components excluded
            - Processing parameters applied
        """
        start_time = time.time()
        raw = data.copy()
        warnings = []
        output_files = []
        metadata = {}

        try:
            # ==================================================================
            # Step 1: Set Montage
            # ==================================================================
            if self.montage_enabled:
                raw, montage_info = self._set_montage(raw)
                metadata['montage'] = montage_info
                if montage_info.get('warning'):
                    warnings.append(montage_info['warning'])

            # ==================================================================
            # Step 1.5: Data Quality Assessment
            # ==================================================================
            from eegcpm.modules.preprocessing.data_quality import detect_all_quality_issues

            # Track original channel count BEFORE any drops
            original_eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            n_original_channels = len(original_eeg_picks)

            quality_report = detect_all_quality_issues(raw)
            metadata['data_quality'] = quality_report

            # Add quality issues to warnings
            if quality_report['issues']:
                for issue in quality_report['issues']:
                    warnings.append(f"Data Quality: {issue}")

            # ==================================================================
            # Step 2: Drop problematic channels BEFORE filtering
            # ==================================================================
            # Drop extreme channels that are obviously bad (flatline, bridged, extreme variance)
            # These would corrupt filtering and bad channel detection

            # Drop flatline channels
            if quality_report['flatline_channels']:
                n_flatline = len(quality_report['flatline_channels'])
                raw.drop_channels(quality_report['flatline_channels'], on_missing='ignore')
                warnings.append(f"Dropped {n_flatline} flatline channels: {', '.join(quality_report['flatline_channels'][:5])}{' ...' if n_flatline > 5 else ''}")

            # Drop bridged channels
            if quality_report['bridged_channels']:
                n_bridged = len(quality_report['bridged_channels'])
                raw.drop_channels(quality_report['bridged_channels'], on_missing='ignore')
                warnings.append(f"Dropped {n_bridged} bridged channels: {', '.join(quality_report['bridged_channels'][:5])}{' ...' if n_bridged > 5 else ''}")

            # Drop high variance channels
            if quality_report['high_variance_channels']:
                n_high_var = len(quality_report['high_variance_channels'])
                raw.drop_channels(quality_report['high_variance_channels'], on_missing='ignore')
                warnings.append(f"Dropped {n_high_var} high variance channels: {', '.join(quality_report['high_variance_channels'][:5])}{' ...' if n_high_var > 5 else ''}")

            # ==================================================================
            # Step 3: Filter - EARLY (Industry Standard)
            # ==================================================================
            # Filter BEFORE bad channel detection (PREP, MNE, ASR best practices)
            # RANSAC correlation needs stationary, detrended data
            raw, filter_info = self._apply_filtering(raw)
            metadata['filter'] = filter_info

            # ==================================================================
            # Step 4: Resample (if enabled) - After filter, before bad channel detection
            # ==================================================================
            if self.resample_enabled:
                original_sfreq = raw.info['sfreq']
                raw.resample(self.resample_sfreq, verbose=False)
                metadata['resample'] = {
                    'original_sfreq': original_sfreq,
                    'new_sfreq': self.resample_sfreq
                }

            # ==================================================================
            # Step 5: Annotate Artifacts (on filtered data)
            # ==================================================================
            if self.artifact_config.get('enabled', True):
                raw, artifact_info = self._annotate_artifacts(raw)
                metadata['artifacts'] = artifact_info

            # ==================================================================
            # Step 6: Detect Bad Channels (MARK ONLY - don't interpolate yet!)
            # ==================================================================
            # RANSAC on filtered data, mark channels in raw.info['bads']
            # Interpolation happens AFTER ASR to avoid creating artifacts
            if self.bad_channels_config.get('auto_detect', True) or self.bad_channels_config.get('manual_bads'):
                raw, bad_channel_info = self._detect_bad_channels_mark_only(raw, n_original_channels)
                metadata['bad_channels'] = bad_channel_info

                # Log quality status
                if bad_channel_info['quality_status'] == 'exclude':
                    warnings.append(bad_channel_info['quality_message'])
                    print(f"      ⚠️ QUALITY WARNING: {bad_channel_info['quality_message']}")
                elif bad_channel_info['quality_status'] == 'warning':
                    warnings.append(bad_channel_info['quality_message'])
                    print(f"      ⚠️  {bad_channel_info['quality_message']}")

            # ==================================================================
            # Step 7: ASR (if enabled) - On filtered data with marked bads
            # ==================================================================
            # ASR repairs high-amplitude artifacts
            # Works with marked bad channels (doesn't need them interpolated)
            if self.use_asr:
                raw, asr_info = self._apply_asr(raw)
                metadata['asr'] = asr_info
                if asr_info.get('error'):
                    warnings.append(asr_info['error'])

            # ==================================================================
            # Step 8: Interpolate Bad Channels - AFTER ASR
            # ==================================================================
            # Now safe to interpolate after ASR has repaired artifacts
            # BUT: Don't interpolate if too many bad channels (not enough good references)
            if len(raw.info['bads']) > 0:
                n_interp = len(raw.info['bads'])
                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
                n_total_eeg = len(eeg_picks)
                pct_bad_for_interp = 100 * n_interp / n_total_eeg if n_total_eeg > 0 else 100

                # Safety threshold: only interpolate if <50% bad channels
                # With 50%+ bad channels, interpolation is unreliable
                if pct_bad_for_interp < 50.0:
                    raw.interpolate_bads(reset_bads=True, verbose=False)
                    metadata['bad_channels']['n_interpolated'] = n_interp
                    metadata['bad_channels']['interpolation_after_asr'] = True
                else:
                    # Too many bad channels - drop them instead of interpolating
                    warnings.append(
                        f"Cannot interpolate {n_interp}/{n_total_eeg} ({pct_bad_for_interp:.1f}%) bad channels - "
                        f"not enough good channels for reliable interpolation. Dropping bad channels instead."
                    )
                    raw.drop_channels(raw.info['bads'])
                    metadata['bad_channels']['n_interpolated'] = 0
                    metadata['bad_channels']['n_dropped_post_asr'] = n_interp
                    metadata['bad_channels']['drop_reason'] = 'too_many_for_interpolation'
                    metadata['bad_channels']['interpolation_after_asr'] = False

            # ==================================================================
            # Step 9: Re-reference - After interpolation
            # ==================================================================
            # Needs complete channel set for proper average reference
            raw, ref_info = self._apply_reference(raw)
            metadata['reference'] = ref_info

            # ==================================================================
            # Step 10: ICA - Final artifact removal
            # ==================================================================
            # ICA on filtered, ASR-repaired, interpolated, re-referenced data
            ica = None
            if self.ica_enabled:
                try:
                    raw, ica, ica_info, ica_warnings = self._run_ica(raw, quality_report)
                    metadata['ica'] = ica_info
                    warnings.extend(ica_warnings)
                except Exception as e:
                    # ICA failed but don't stop preprocessing
                    metadata['ica'] = {
                        'enabled': True,
                        'success': False,
                        'error': str(e)
                    }
                    warnings.append(f"ICA failed: {str(e)}")

            # ==================================================================
            # Save outputs
            # ==================================================================
            subject_id = subject.id if subject else kwargs.get('subject_id', 'unknown')
            session_id = kwargs.get('session_id', '')
            session_str = f"_ses-{session_id}" if session_id else ""

            # Save preprocessed data
            output_path = self.output_dir / f"{subject_id}{session_str}_preprocessed_raw.fif"
            raw.save(output_path, overwrite=True, verbose=False)
            output_files.append(output_path)

            # Save ICA if computed
            if ica is not None:
                ica_path = self.output_dir / f"{subject_id}{session_str}_ica.fif"
                ica.save(ica_path, overwrite=True)
                output_files.append(ica_path)

            # Summary metadata
            metadata.update({
                'n_channels': len(raw.ch_names),
                'sfreq': raw.info['sfreq'],
                'duration_s': raw.times[-1],
                'n_annotations': len(raw.annotations) if raw.annotations else 0,
            })

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={"data": raw, "ica": ica},
                output_files=output_files,
                warnings=warnings,
                metadata=metadata,
            )

        except Exception as e:
            import traceback
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)],
                metadata={'traceback': traceback.format_exc()},
            )

    def _set_montage(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, dict]:
        """Set channel montage for spatial information."""
        info = {'type': self.montage_type, 'applied': False}

        if raw.get_montage() is not None:
            info['applied'] = True
            info['note'] = 'Montage already set'
            return raw, info

        try:
            if self.montage_file:
                # Custom montage from file
                montage = mne.channels.read_custom_montage(self.montage_file)
                info['source'] = 'custom_file'
            else:
                # Standard montage
                montage = mne.channels.make_standard_montage(self.montage_type)
                info['source'] = 'standard'

            raw.set_montage(montage, on_missing='warn', verbose=False)
            info['applied'] = True

        except Exception as e:
            info['warning'] = f"Could not set montage: {e}"
            info['applied'] = False

        return raw, info

    def _detect_bad_channels_mark_only(
        self,
        raw: mne.io.BaseRaw,
        n_original_channels: int
    ) -> Tuple[mne.io.BaseRaw, dict]:
        """Detect bad channels and MARK ONLY (don't interpolate).

        For use with ASR: mark channels as bad, let ASR handle them,
        then interpolate after ASR.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data (after flatline/bridged channels removed, filtered)
        n_original_channels : int
            Original number of EEG channels before any drops

        Returns
        -------
        raw : mne.io.BaseRaw
            Raw with bad channels marked in raw.info['bads']
        info : dict
            Detection metadata
        """
        # Detect bad channels
        detector = BadChannelDetector(
            methods=self.bad_channels_config.get('methods', ['variance', 'correlation']),
            variance_threshold=self.bad_channels_config.get('variance_threshold', 5.0),
            correlation_threshold=self.bad_channels_config.get('correlation_threshold', 0.4),
            ransac_sample_prop=self.bad_channels_config.get('ransac_sample_prop', 0.25),
            ransac_corr_threshold=self.bad_channels_config.get('ransac_corr_threshold', 0.75),
            deviation_threshold=self.bad_channels_config.get('deviation_threshold', 5.0),
        )

        bad_channels = detector.detect(raw)

        # Add manual bads
        manual_bads = self.bad_channels_config.get('manual_bads', [])
        if manual_bads:
            bad_channels = list(set(bad_channels + manual_bads))

        # Clustering analysis (if enabled)
        clustering_result = None
        if (self.bad_channels_config.get('check_clustering', True) and
            len(bad_channels) > 0 and
            raw.get_montage() is not None):

            clustering_result = compute_bad_channel_clustering(
                raw,
                bad_channels,
                n_neighbors=self.bad_channels_config.get('clustering_k_neighbors', 6),
                cluster_threshold=self.bad_channels_config.get('clustering_neighbor_threshold', 0.5)
            )

        # Mark bad channels (don't interpolate yet!)
        raw.info['bads'] = bad_channels

        # Calculate metrics
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        n_current_channels = len(eeg_picks)
        n_dropped = n_original_channels - n_current_channels
        n_total_bad = n_dropped + len(bad_channels)
        pct_bad = 100 * n_total_bad / n_original_channels if n_original_channels > 0 else 0

        # Quality assessment
        max_bad_pct = self.bad_channels_config.get('max_bad_channel_percent', 20.0)
        clustering_severity = clustering_result['severity'] if clustering_result else 'none'

        if pct_bad > max_bad_pct or clustering_severity == 'severe':
            quality_status = 'exclude'
            quality_message = f"Subject has {n_total_bad}/{n_original_channels} ({pct_bad:.1f}%) bad channels"
            if clustering_severity == 'severe':
                quality_message += f", with severe clustering ({clustering_result['pct_clustered']:.1f}% clustered)"
            quality_message += f", exceeding quality threshold. Consider excluding this subject."
        elif pct_bad > max_bad_pct / 2 or clustering_severity == 'moderate':
            quality_status = 'warning'
            quality_message = f"Subject has {n_total_bad}/{n_original_channels} ({pct_bad:.1f}%) bad channels"
            if clustering_severity == 'moderate':
                quality_message += f", with moderate clustering ({clustering_result['pct_clustered']:.1f}% clustered)"
            quality_message += f". Data quality acceptable but monitor closely."
        else:
            quality_status = 'ok'
            quality_message = f"Subject has {n_total_bad}/{n_original_channels} ({pct_bad:.1f}%) bad channels. Data quality good."

        info = {
            'n_bad_channels': len(bad_channels),
            'bad_channels': bad_channels,
            'n_interpolated': 0,  # Will be updated after ASR
            'n_dropped_total': n_dropped,
            'pct_bad_channels': pct_bad,
            'clustering_severity': clustering_result['severity'] if clustering_result else 'none',
            'quality_status': quality_status,
            'quality_message': quality_message,
            'interpolation_deferred': True,  # Key flag
        }

        if clustering_result:
            info['clustering'] = clustering_result

        return raw, info

    def _handle_bad_channels(
        self,
        raw: mne.io.BaseRaw,
        n_original_channels: int
    ) -> Tuple[mne.io.BaseRaw, dict]:
        """Detect and interpolate bad channels with clustering-aware handling.

        DEPRECATED: Use _detect_bad_channels_mark_only for new pipeline.
        This method kept for backward compatibility.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data (after flatline/bridged channels removed)
        n_original_channels : int
            Original number of EEG channels before any drops
        """
        # Step 1: Detect bad channels (without interpolation yet)
        detector = BadChannelDetector(
            methods=self.bad_channels_config.get('methods', ['variance', 'correlation']),
            variance_threshold=self.bad_channels_config.get('variance_threshold', 5.0),
            correlation_threshold=self.bad_channels_config.get('correlation_threshold', 0.4),
            ransac_sample_prop=self.bad_channels_config.get('ransac_sample_prop', 0.25),
            ransac_corr_threshold=self.bad_channels_config.get('ransac_corr_threshold', 0.75),
            deviation_threshold=self.bad_channels_config.get('deviation_threshold', 5.0),
        )

        bad_channels = detector.detect(raw)

        # Add manual bads if specified
        manual_bads = self.bad_channels_config.get('manual_bads', [])
        if manual_bads:
            bad_channels = list(set(bad_channels + manual_bads))

        # Step 2: Clustering analysis (if enabled and montage available)
        clustering_result = None
        channels_to_drop = []
        channels_to_interpolate = []

        if (self.bad_channels_config.get('check_clustering', True) and
            len(bad_channels) > 0 and
            raw.get_montage() is not None):

            clustering_result = compute_bad_channel_clustering(
                raw,
                bad_channels,
                n_neighbors=self.bad_channels_config.get('clustering_k_neighbors', 6),
                cluster_threshold=self.bad_channels_config.get('clustering_neighbor_threshold', 0.5)
            )

            if 'error' not in clustering_result:
                # Determine action based on severity and config
                action = self.bad_channels_config.get('clustering_action', 'adaptive')
                severity = clustering_result['severity']
                clustered_chs = clustering_result.get('clustered_channels', [])
                isolated_chs = [ch for ch in bad_channels if ch not in clustered_chs]

                if action == 'interpolate_all':
                    # Ignore clustering, interpolate everything
                    channels_to_interpolate = bad_channels

                elif action == 'drop_clustered':
                    # Always drop clustered, interpolate isolated
                    channels_to_drop = clustered_chs
                    channels_to_interpolate = isolated_chs

                elif action == 'fail_on_severe':
                    # Fail if severe clustering
                    if severity == 'severe':
                        raise RuntimeError(
                            f"Severe bad channel clustering detected: "
                            f"{clustering_result['n_clustered_channels']}/{len(bad_channels)} "
                            f"({clustering_result['pct_clustered']:.1f}%) channels clustered. "
                            f"Data quality too poor for processing."
                        )
                    else:
                        channels_to_interpolate = bad_channels

                else:  # 'adaptive' (default)
                    # Mild: interpolate all
                    # Moderate: drop clustered, interpolate isolated
                    # Severe: drop all clustered
                    if severity == 'none' or severity == 'mild':
                        channels_to_interpolate = bad_channels
                    elif severity == 'moderate':
                        channels_to_drop = clustered_chs
                        channels_to_interpolate = isolated_chs
                    else:  # severe
                        channels_to_drop = clustered_chs
                        channels_to_interpolate = isolated_chs
        else:
            # No clustering check - use standard interpolation
            channels_to_interpolate = bad_channels

        # Safeguard: Check if dropping would leave too few channels
        current_n_channels = len(raw.ch_names)
        min_channels_required = 10  # Minimum viable for analysis

        if len(channels_to_drop) > 0:
            n_remaining_after_drop = current_n_channels - len(channels_to_drop)
            if n_remaining_after_drop < min_channels_required:
                # Would leave too few channels - fall back to interpolation
                print(f"      WARNING: Dropping {len(channels_to_drop)} channels would leave only "
                      f"{n_remaining_after_drop} channels (< {min_channels_required} minimum). "
                      f"Falling back to interpolation.")
                channels_to_interpolate = bad_channels
                channels_to_drop = []
                if clustering_result:
                    clustering_result['fallback_to_interpolation'] = True
                    clustering_result['fallback_reason'] = 'insufficient_channels_after_drop'

        # Step 3: Apply drops
        if channels_to_drop:
            print(f"      Dropping {len(channels_to_drop)} clustered bad channels: "
                  f"{', '.join(channels_to_drop[:5])}{' ...' if len(channels_to_drop) > 5 else ''}")
            raw.drop_channels(channels_to_drop, on_missing='ignore')

        # Step 4: Apply interpolation
        if channels_to_interpolate and self.bad_channels_config.get('interpolate', True):
            print(f"      Interpolating {len(channels_to_interpolate)} bad channels: "
                  f"{', '.join(channels_to_interpolate[:5])}{' ...' if len(channels_to_interpolate) > 5 else ''}")
            raw.info['bads'] = channels_to_interpolate
            raw.interpolate_bads(reset_bads=True, mode='accurate', verbose=False)

        # Calculate current EEG channels (after all processing)
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        n_current_channels = len(eeg_picks)

        # Total bad = all dropped (from data quality) + RANSAC bad
        n_ransac_bad = len(bad_channels)
        n_dropped = n_original_channels - n_current_channels - len(channels_to_interpolate)  # All drops
        n_total_bad = n_dropped + n_ransac_bad

        # Percentage based on ORIGINAL channel count
        pct_bad = 100 * n_total_bad / n_original_channels if n_original_channels > 0 else 0

        # Check against quality threshold
        max_bad_pct = self.bad_channels_config.get('max_bad_channel_percent', 20.0)

        # Determine quality status (including clustering severity)
        clustering_severity = clustering_result.get('severity', 'none') if clustering_result else 'none'

        if pct_bad > max_bad_pct or clustering_severity == 'severe':
            quality_status = 'exclude'
            quality_message = (
                f"Subject has {n_total_bad}/{n_original_channels} ({pct_bad:.1f}%) bad channels "
                f"({n_dropped} dropped + {len(channels_to_interpolate)} interpolated)"
            )
            if clustering_severity == 'severe':
                quality_message += f", with severe clustering ({clustering_result['pct_clustered']:.1f}% clustered)"
            quality_message += f", exceeding quality threshold. Consider excluding this subject."

        elif pct_bad > max_bad_pct / 2 or clustering_severity == 'moderate':
            quality_status = 'warning'
            quality_message = (
                f"Subject has {n_total_bad}/{n_original_channels} ({pct_bad:.1f}%) bad channels "
                f"({n_dropped} dropped + {len(channels_to_interpolate)} interpolated)"
            )
            if clustering_severity == 'moderate':
                quality_message += f", with moderate clustering ({clustering_result['pct_clustered']:.1f}% clustered)"
            quality_message += f". Data quality acceptable but monitor closely."
        else:
            quality_status = 'ok'
            quality_message = (
                f"Subject has {n_total_bad}/{n_original_channels} ({pct_bad:.1f}%) bad channels "
                f"({n_dropped} dropped + {len(channels_to_interpolate)} interpolated). Data quality good."
            )

        info = {
            'detected': bad_channels,
            'n_bad': n_ransac_bad,  # RANSAC bad channels
            'n_dropped': n_dropped + len(channels_to_drop),  # Dropped by data quality + clustering
            'n_dropped_clustered': len(channels_to_drop),  # Dropped due to clustering
            'n_interpolated': len(channels_to_interpolate),  # Actually interpolated
            'n_total_bad': n_total_bad,  # Total bad
            'n_total': n_original_channels,  # Original count
            'n_remaining': n_current_channels,  # After all processing
            'percent_bad': pct_bad,
            'threshold': max_bad_pct,
            'quality_status': quality_status,  # 'ok', 'warning', or 'exclude'
            'quality_message': quality_message,
            'methods': self.bad_channels_config.get('methods', []),
            'clustering': clustering_result,  # Include full clustering info
            'clustering_action': self.bad_channels_config.get('clustering_action', 'adaptive'),
        }

        return raw, info

    def _annotate_artifacts(
        self,
        raw: mne.io.BaseRaw
    ) -> Tuple[mne.io.BaseRaw, dict]:
        """Annotate artifact segments."""
        n_annotations_before = len(raw.annotations) if raw.annotations else 0

        raw = annotate_artifacts_from_config(
            raw,
            self.artifact_config,
            copy=False
        )

        n_annotations_after = len(raw.annotations) if raw.annotations else 0
        n_new = n_annotations_after - n_annotations_before

        # Count annotations by type
        annotation_counts = {}
        if raw.annotations:
            for desc in raw.annotations.description:
                if desc.startswith('BAD_'):
                    annotation_counts[desc] = annotation_counts.get(desc, 0) + 1

        info = {
            'n_annotations_added': n_new,
            'annotation_counts': annotation_counts,
            'amplitude_threshold': self.artifact_config.get('amplitude_threshold'),
            'gradient_threshold': self.artifact_config.get('gradient_threshold'),
        }

        return raw, info

    def _apply_reference(
        self,
        raw: mne.io.BaseRaw
    ) -> Tuple[mne.io.BaseRaw, dict]:
        """Apply re-referencing."""
        raw = apply_reference_from_config(
            raw,
            self.reference_config,
            copy=False
        )

        info = {
            'type': self.reference_config['type'],
            'channels': self.reference_config.get('channels'),
            'projection': self.reference_config.get('projection', True),
        }

        return raw, info

    def _apply_filtering(
        self,
        raw: mne.io.BaseRaw
    ) -> Tuple[mne.io.BaseRaw, dict]:
        """Apply bandpass and notch filtering."""
        # Bandpass filter
        raw.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            picks="eeg",
            verbose=False,
        )

        info = {
            'l_freq': self.l_freq,
            'h_freq': self.h_freq,
            'notch_freq': None,
        }

        # Notch filter (if specified)
        if self.notch_freq:
            notch_freqs = self.notch_freq if isinstance(self.notch_freq, list) else [self.notch_freq]
            raw.notch_filter(
                freqs=notch_freqs,
                picks="eeg",
                verbose=False,
            )
            info['notch_freq'] = notch_freqs

        return raw, info

    def _run_ica(
        self,
        raw: mne.io.BaseRaw,
        quality_report: Optional[Dict[str, Any]] = None
    ) -> Tuple[mne.io.BaseRaw, mne.preprocessing.ICA, dict, List[str]]:
        """Run ICA and remove artifact components.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG data
        quality_report : dict, optional
            Data quality report from detect_all_quality_issues()
            Contains rank information and ICA feasibility

        Returns
        -------
        raw : mne.io.BaseRaw
            Preprocessed raw data with ICA applied
        ica : mne.preprocessing.ICA or None
            Fitted ICA object, or None if ICA failed
        info : dict
            Metadata about ICA processing
        warnings : list of str
            Warning messages generated during ICA
        """
        # Initialize warnings list for this method
        ica_warnings = []

        # Check if ICA is feasible based on data quality
        if quality_report is not None:
            if not quality_report['ica_feasible']:
                rank = quality_report['rank'].get('eeg', 0)
                ica_warnings.append(f"Skipping ICA: data rank ({rank}) too low for reliable decomposition")
                return raw, None, {
                    'enabled': True,
                    'success': False,
                    'reason': 'insufficient_rank',
                    'rank': rank
                }, ica_warnings

        # Validate data before ICA (check for NaN/Inf)
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(eeg_picks) == 0:
            raise ValueError("No EEG channels available for ICA")

        data = raw.get_data(picks=eeg_picks)

        # Check for and handle NaN/Inf values
        if np.isnan(data).any() or np.isinf(data).any():
            # Find problematic channels
            has_nan = np.isnan(data).any(axis=1)
            has_inf = np.isinf(data).any(axis=1)
            bad_idx = np.where(has_nan | has_inf)[0]

            if len(bad_idx) > 0:
                bad_channel_names = [raw.ch_names[eeg_picks[i]] for i in bad_idx]
                print(f"Warning: Dropping {len(bad_idx)} channels with NaN/Inf values: {bad_channel_names}")
                raw.drop_channels(bad_channel_names)

                # Update eeg_picks after dropping channels
                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

                # Re-check data after dropping channels
                data = raw.get_data(picks=eeg_picks)
                if np.isnan(data).any() or np.isinf(data).any():
                    # Still have NaN/Inf in time domain - this data is too corrupted
                    raise ValueError(
                        f"Data contains NaN/Inf values in time domain even after dropping {len(bad_idx)} channels. "
                        "This subject's data quality is too poor for reliable ICA. Consider excluding this subject."
                    )

        # Determine number of components
        # Priority: 1) user-specified, 2) rank-based recommendation, 3) default
        n_components = self.ica_n_components

        # Handle 'rank-1' string option
        if n_components == 'rank-1':
            # Compute data rank and use rank - 1
            rank_dict = mne.compute_rank(raw, rank='info')
            data_rank = rank_dict.get('eeg', len(eeg_picks))
            n_components = max(5, data_rank - 1)  # At least 5 components
            ica_warnings.append(f"Using rank-1 mode: {n_components} components (data rank={data_rank})")

        # Handle variance-based float option (e.g., 0.99)
        elif isinstance(n_components, float) and 0 < n_components < 1:
            # MNE ICA supports this natively - will explain this much variance
            ica_warnings.append(f"Using variance mode: keeping {n_components*100:.1f}% variance")

        # Handle None (use recommendation or default)
        elif n_components is None:
            # User didn't specify, use rank-based or default
            if quality_report is not None and quality_report['recommended_ica_components'] is not None:
                n_components = quality_report['recommended_ica_components']
                ica_warnings.append(f"Using rank-based ICA components: {n_components} (rank={quality_report['rank']['eeg']})")
            else:
                n_components = min(20, len(eeg_picks) - 1)

        # Create ICA
        # Extended-infomax: method='infomax' with fit_params={'extended': True}
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=self.ica_method,
            fit_params=self.ica_fit_params if self.ica_fit_params else None,
            random_state=42,
            max_iter="auto",
        )

        # Fit ICA (exclude annotated bad segments)
        # Handle case where data quality is poor and n_components is too high
        try:
            ica.fit(raw, picks="eeg", reject_by_annotation=True, verbose=False)
        except (IndexError, ValueError) as e:
            # IndexError: "index N is out of bounds" - too many components
            # ValueError: "array must not contain infs or NaNs" - data quality issue
            error_msg = str(e)

            if "array must not contain infs or NaNs" in error_msg:
                # Data quality too poor for ICA
                ica_warnings.append(
                    "ICA failed due to NaN/Inf in data. Subject's data quality is too poor for ICA. "
                    "This can happen when bad channel interpolation creates artifacts. "
                    "Consider excluding this subject or using a pipeline without ICA."
                )
                return raw, None, {
                    'enabled': True,
                    'success': False,
                    'error': 'data_contains_nan_inf',
                    'message': error_msg
                }, ica_warnings

            # Try with fewer components for other errors
            if n_components > 10:
                ica_warnings.append(f"ICA failed with {n_components} components, retrying with 10")
                ica = mne.preprocessing.ICA(
                    n_components=10,
                    method=self.ica_method,
                    fit_params=self.ica_fit_params if self.ica_fit_params else None,
                    random_state=42,
                    max_iter="auto",
                )
                try:
                    ica.fit(raw, picks="eeg", reject_by_annotation=True, verbose=False)
                except Exception as retry_error:
                    ica_warnings.append(f"ICA failed even with 10 components: {str(retry_error)}")
                    return raw, None, {'enabled': True, 'success': False, 'error': str(retry_error)}, ica_warnings
            else:
                ica_warnings.append(f"ICA failed: {error_msg}")
                return raw, None, {'enabled': True, 'success': False, 'error': error_msg}, ica_warnings

        # Collect components to exclude
        exclude_indices = list(self.ica_manual_exclude)
        auto_detected = {'eog': [], 'ecg': [], 'iclabel': []}

        # Auto-detect EOG components
        if self.ica_auto_detect:
            eog_picks = mne.pick_types(raw.info, eog=True)
            if len(eog_picks) > 0:
                try:
                    eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
                    auto_detected['eog'] = eog_indices
                    exclude_indices.extend(eog_indices)
                except Exception:
                    pass

            # Auto-detect ECG components
            ecg_picks = mne.pick_types(raw.info, ecg=True)
            if len(ecg_picks) > 0:
                try:
                    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, verbose=False)
                    auto_detected['ecg'] = ecg_indices
                    exclude_indices.extend(ecg_indices)
                except Exception:
                    pass

        # ICLabel classification (if enabled and available)
        if self.use_iclabel:
            try:
                from mne_icalabel import label_components

                # Validate ICA components don't have NaN/Inf before ICLabel
                ica_data = ica.get_sources(raw).get_data()
                if np.isnan(ica_data).any() or np.isinf(ica_data).any():
                    ica_warnings.append("Skipping ICLabel: ICA components contain NaN/Inf values")
                else:
                    labels = label_components(raw, ica, method='iclabel')
                    # Get artifact components (not brain or other)
                    artifact_labels = ['eye blink', 'heart beat', 'muscle artifact', 'line noise', 'channel noise']

                    # Safely iterate over available labels and probabilities
                    n_labels = min(len(labels['labels']), labels['y_pred_proba'].shape[0])
                    for idx in range(n_labels):
                        label = labels['labels'][idx]
                        # Get max probability across all classes for this component
                        prob = labels['y_pred_proba'][idx].max() if labels['y_pred_proba'].ndim > 1 else labels['y_pred_proba'][idx]

                        if label in artifact_labels and prob > self.iclabel_threshold:
                            auto_detected['iclabel'].append(idx)
                            exclude_indices.append(idx)
            except ImportError:
                pass
            except Exception as e:
                ica_warnings.append(f"ICLabel failed: {str(e)}")

        # Remove duplicates
        ica.exclude = sorted(list(set(exclude_indices)))

        # Apply ICA
        raw = ica.apply(raw, verbose=False)

        info = {
            'method': self.ica_method,
            'n_components': ica.n_components_,
            'n_excluded': len(ica.exclude),
            'excluded_indices': ica.exclude,
            'auto_detected': auto_detected,
            'manual_excluded': list(self.ica_manual_exclude),
        }

        return raw, ica, info, ica_warnings

    def _apply_asr(
        self,
        raw: mne.io.BaseRaw
    ) -> Tuple[mne.io.BaseRaw, dict]:
        """
        Apply Artifact Subspace Reconstruction (experimental).

        Requires asrpy package: pip install asrpy

        ASR expects MNE Raw objects for both fit() and transform().
        """
        info = {'applied': False, 'cutoff': self.asr_cutoff}

        try:
            from asrpy import ASR

            # Check for NaN/Inf in data before attempting ASR
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            data_sample = raw.get_data(picks=eeg_picks, start=0, stop=min(1000, raw.n_times))

            if not np.isfinite(data_sample).all():
                info['error'] = "Data contains NaN/Inf values (likely from poor quality interpolation). Skipping ASR."
                info['skipped'] = True
                return raw, info

            # Initialize ASR with parameters
            asr = ASR(
                sfreq=raw.info["sfreq"],
                cutoff=self.asr_cutoff,
                max_bad_chans=0.1  # Max 10% bad channels during calibration
            )

            # Fit ASR on clean calibration window (first N seconds)
            # ASR.fit() expects a Raw object, not numpy array
            train_duration_samples = int(self.asr_train_duration * raw.info["sfreq"])
            train_duration_samples = min(train_duration_samples, raw.n_times)

            # Fit on calibration window
            asr.fit(raw, picks='eeg', start=0, stop=train_duration_samples)

            # Transform entire dataset
            # ASR.transform() expects a Raw object and returns a Raw object
            raw_cleaned = asr.transform(raw, picks='eeg')

            info['applied'] = True
            info['train_duration'] = self.asr_train_duration
            info['train_samples'] = train_duration_samples

        except ImportError:
            info['error'] = "ASR not available (install asrpy: pip install asrpy)"
            return raw, info
        except Exception as e:
            import traceback
            info['error'] = f"{str(e)}"
            info['skipped'] = True
            # Return original raw if ASR fails
            return raw, info

        return raw_cleaned, info

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration as dictionary.

        Returns configuration that can be saved to YAML and
        reused to reproduce the preprocessing.
        """
        return {
            'montage': {
                'enabled': self.montage_enabled,
                'type': self.montage_type,
                'file': self.montage_file,
            },
            'bad_channels': self.bad_channels_config,
            'artifact_detection': self.artifact_config,
            'reference': self.reference_config,
            'resample': {
                'enabled': self.resample_enabled,
                'sfreq': self.resample_sfreq,
            },
            'filter': {
                'l_freq': self.l_freq,
                'h_freq': self.h_freq,
                'notch_freq': self.notch_freq,
            },
            'ica': {
                'enabled': self.ica_enabled,
                'method': self.ica_method,
                'n_components': self.ica_n_components,
                'auto_detect_artifacts': self.ica_auto_detect,
                'manual_exclude': self.ica_manual_exclude,
                'use_iclabel': self.use_iclabel,
                'iclabel_threshold': self.iclabel_threshold,
            },
            'asr': {
                'enabled': self.use_asr,
                'cutoff': self.asr_cutoff,
                'train_duration': self.asr_train_duration,
            },
        }

    def get_output_spec(self) -> Dict[str, str]:
        """Get specification of module outputs."""
        return {
            "data": "Preprocessed mne.io.Raw object",
            "ica": "Fitted ICA object (if ICA enabled)",
        }


# Convenience functions for standalone use
def preprocess_raw(
    raw: mne.io.BaseRaw,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    subject_id: str = "unknown",
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """
    Preprocess raw EEG data using configuration dictionary.

    This is a convenience function for quick preprocessing without
    creating a full module instance.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    config : dict
        Preprocessing configuration
    output_dir : Path, optional
        Output directory (if None, data is not saved)
    subject_id : str
        Subject identifier for output files

    Returns
    -------
    raw : mne.io.BaseRaw
        Preprocessed data
    metadata : dict
        Processing metadata and statistics

    Examples
    --------
    >>> import yaml
    >>> with open("config/preprocessing_config.yaml") as f:
    ...     config = yaml.safe_load(f)
    >>> raw_clean, info = preprocess_raw(raw, config)
    >>> print(f"Excluded {info['ica']['n_excluded']} ICA components")
    """
    if output_dir is None:
        output_dir = Path(".")

    module = PreprocessingModule(config, output_dir)
    result = module.process(raw, subject_id=subject_id)

    if result.success:
        return result.outputs['data'], result.metadata
    else:
        raise RuntimeError(f"Preprocessing failed: {result.errors}")


# Export all public classes and functions
__all__ = [
    'PreprocessingModule',
    'preprocess_raw',
    'EEGReferencing',
    'apply_reference_from_config',
    'BadChannelDetector',
    'detect_and_interpolate_bad_channels',
    'ArtifactAnnotator',
    'annotate_artifacts_from_config',
]
