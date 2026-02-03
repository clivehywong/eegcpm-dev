"""
Source-Level ERP Feature Extraction Module

Extracts Event-Related Potential features from source-reconstructed ROI time courses.
This allows observing ERP patterns at the source level.

Features include:
- Peak amplitudes and latencies for defined components at each ROI
- Mean amplitudes in time windows per ROI
- Network-level ERP summaries
- Source-level ERP QC reports
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eegcpm.pipeline.base import BaseModule, ModuleResult


class SourceERPFeatureModule(BaseModule):
    """Extract ERP features from source-reconstructed ROI time courses."""

    name = "source_erp_features"
    version = "0.1.0"
    description = "ERP feature extraction (source-level)"

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)

        # Feature extraction parameters
        self.components = config.get("components", {})
        self.mean_amplitude_windows = config.get("mean_amplitude_windows", {})
        self.metrics = config.get("metrics", ["peak_amplitude", "peak_latency"])
        self.rois = config.get("rois", "all")  # Which ROIs to extract

        # Dependencies
        depends_on = config.get("depends_on", {})
        self.preprocessing = depends_on.get("preprocessing", "unknown")
        self.task = depends_on.get("task", "unknown")
        self.source_variant = depends_on.get("source", "unknown")

        # Variant name
        self.variant = config.get("variant", "default")

    def validate_input(self, data: Dict[str, np.ndarray]) -> bool:
        """Validate input is ROI time courses dictionary."""
        return isinstance(data, dict) and "roi_names" in data

    def process(
        self,
        roi_data: Dict[str, np.ndarray],
        subject: Optional[Any] = None,
        **kwargs
    ) -> ModuleResult:
        """
        Extract ERP features from ROI time courses.

        Parameters
        ----------
        roi_data : dict
            Dictionary with ROI time courses:
            - 'roi_names': list of ROI names
            - '{roi_name}': np.ndarray of shape (n_conditions, n_times)
        subject : object
            Subject object with id and session attributes
        **kwargs : dict
            Additional arguments (sampling_rate, etc.)

        Returns
        -------
        ModuleResult
            Results with extracted features DataFrame
        """
        start_time = time.time()

        try:
            if not self.validate_input(roi_data):
                return ModuleResult(
                    success=False,
                    module_name=self.name,
                    execution_time_seconds=time.time() - start_time,
                    errors=["Input must be ROI time courses dictionary with 'roi_names' key"]
                )

            subject_id = subject.id if subject else "unknown"
            session_id = subject.session if hasattr(subject, "session") and subject.session else "01"

            # Get ROI names
            roi_names = roi_data.get("roi_names", [])

            # Filter ROIs if specified
            if self.rois != "all":
                roi_names = [r for r in roi_names if r in self.rois]

            # Get sampling rate for time axis
            sfreq = kwargs.get("sfreq", 500.0)

            # Initialize feature dictionary
            features = {
                "subject_id": subject_id,
                "session_id": session_id
            }

            # Extract features for each ROI
            for roi_name in roi_names:
                if roi_name not in roi_data or roi_name == "roi_names":
                    continue

                roi_tc = roi_data[roi_name]

                # Average across conditions to get ROI ERP
                if len(roi_tc.shape) == 2:  # (n_conditions, n_times)
                    roi_erp = np.mean(roi_tc, axis=0)  # Average across conditions
                else:
                    roi_erp = roi_tc

                # Construct time axis
                n_times = len(roi_erp)
                times = np.arange(n_times) / sfreq

                # Extract component features for this ROI
                for comp_name, comp_config in self.components.items():
                    comp_features = self._extract_component_features(
                        roi_erp,
                        times,
                        roi_name,
                        comp_name,
                        comp_config
                    )
                    features.update(comp_features)

                # Extract mean amplitudes if specified
                if self.mean_amplitude_windows:
                    mean_amp_features = self._extract_mean_amplitudes(
                        roi_erp,
                        times,
                        roi_name,
                        self.mean_amplitude_windows
                    )
                    features.update(mean_amp_features)

            # Convert to DataFrame
            features_df = pd.DataFrame([features])

            # Save features
            output_files = []
            features_csv = self.output_dir / "features.csv"

            # Append or create
            if features_csv.exists():
                existing = pd.read_csv(features_csv)
                combined = pd.concat([existing, features_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["subject_id", "session_id"], keep="last")
                combined.to_csv(features_csv, index=False)
            else:
                features_df.to_csv(features_csv, index=False)

            output_files.append(features_csv)

            # Save config
            import yaml
            config_file = self.output_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            output_files.append(config_file)

            # Generate QC report if enabled
            if self.config.get("generate_qc", True):
                qc_html = self._generate_qc_report(
                    roi_data,
                    features_df,
                    subject_id,
                    session_id,
                    times
                )
                if qc_html:
                    output_files.append(qc_html)

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={
                    "features": features_df,
                    "n_features": len(features) - 2,
                    "n_rois": len(roi_names),
                    "components": list(self.components.keys())
                },
                output_files=output_files,
                metadata={
                    "variant": self.variant,
                    "source_variant": self.source_variant,
                    "n_components": len(self.components)
                }
            )

        except Exception as e:
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)]
            )

    def _extract_component_features(
        self,
        roi_erp: np.ndarray,
        times: np.ndarray,
        roi_name: str,
        comp_name: str,
        comp_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract ERP component features for one ROI."""

        features = {}

        # Get component parameters
        search_window = comp_config.get("search_window", [0.0, 0.5])
        polarity = comp_config.get("polarity", "positive")

        # Find time indices in search window
        time_mask = (times >= search_window[0]) & (times <= search_window[1])
        win_times = times[time_mask]
        win_data = roi_erp[time_mask]

        if len(win_data) == 0:
            return {}

        # Find peak
        if polarity == "positive":
            peak_idx = np.argmax(win_data)
        else:
            peak_idx = np.argmin(win_data)

        peak_amplitude = win_data[peak_idx]
        peak_latency = win_times[peak_idx]

        # Store features with ROI-specific naming
        feature_prefix = f"{roi_name}_{comp_name}"

        if "peak_amplitude" in self.metrics:
            features[f"{feature_prefix}_amplitude"] = peak_amplitude

        if "peak_latency" in self.metrics:
            features[f"{feature_prefix}_latency"] = peak_latency

        if "mean_amplitude" in self.metrics:
            features[f"{feature_prefix}_mean"] = np.mean(win_data)

        if "area_under_curve" in self.metrics:
            dt = np.mean(np.diff(win_times)) if len(win_times) > 1 else 1.0
            auc = np.trapz(win_data, dx=dt)
            features[f"{feature_prefix}_auc"] = auc

        return features

    def _extract_mean_amplitudes(
        self,
        roi_erp: np.ndarray,
        times: np.ndarray,
        roi_name: str,
        windows: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Extract mean amplitudes in time windows for one ROI."""

        features = {}

        for window_name, window in windows.items():
            tmin, tmax = window

            # Find time indices
            time_mask = (times >= tmin) & (times <= tmax)
            win_data = roi_erp[time_mask]

            if len(win_data) > 0:
                mean_amp = np.mean(win_data)
                features[f"{roi_name}_mean_amp_{window_name}"] = mean_amp

        return features

    def _generate_qc_report(
        self,
        roi_data: Dict[str, np.ndarray],
        features_df: pd.DataFrame,
        subject_id: str,
        session_id: str,
        times: np.ndarray
    ) -> Optional[Path]:
        """Generate source-level ERP QC report."""

        try:
            import matplotlib.pyplot as plt
            from eegcpm.modules.qc.html_report import HTMLReportBuilder

            # Create HTML report
            builder = HTMLReportBuilder(title=f"Source ERP Features: {subject_id}")

            builder.add_section("Overview")
            builder.add_text(f"**Subject**: {subject_id}")
            builder.add_text(f"**Session**: {session_id}")
            builder.add_text(f"**Source Variant**: {self.source_variant}")
            builder.add_text(f"**Components**: {', '.join(self.components.keys())}")

            # Plot ROI ERPs with component markers
            roi_names = roi_data.get("roi_names", [])
            if self.rois != "all":
                plot_rois = [r for r in roi_names if r in self.rois]
            else:
                plot_rois = roi_names[:12]  # Top 12 ROIs

            if plot_rois:
                fig = self._plot_roi_erps(roi_data, plot_rois, times)
                if fig:
                    from eegcpm.modules.qc.base import BaseQC
                    fig_bytes = BaseQC.fig_to_base64(fig)
                    builder.add_section("ROI Event-Related Potentials")
                    builder.add_figure(fig_bytes, "ERP waveforms for top ROIs")
                    plt.close(fig)

            # Save HTML
            qc_html = self.output_dir / f"{subject_id}_source_erp_qc.html"
            html = builder.build()
            with open(qc_html, 'w') as f:
                f.write(html)

            return qc_html

        except Exception as e:
            print(f"Warning: QC report generation failed: {e}")
            return None

    def _plot_roi_erps(
        self,
        roi_data: Dict[str, np.ndarray],
        roi_names: List[str],
        times: np.ndarray
    ) -> Optional[plt.Figure]:
        """Plot ERP waveforms for selected ROIs."""

        import matplotlib.pyplot as plt

        n_rois = len(roi_names)
        n_cols = 3
        n_rows = (n_rois + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_rois > 1 else [axes]

        for idx, roi_name in enumerate(roi_names):
            if roi_name not in roi_data or roi_name == "roi_names":
                continue

            ax = axes[idx]
            roi_tc = roi_data[roi_name]

            # Average across conditions
            if len(roi_tc.shape) == 2:
                roi_erp = np.mean(roi_tc, axis=0)
            else:
                roi_erp = roi_tc

            # Plot
            ax.plot(times, roi_erp, linewidth=1.5)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.5)

            # Mark component windows
            for comp_name, comp_config in self.components.items():
                window = comp_config.get("search_window", [])
                if window:
                    ax.axvspan(window[0], window[1], alpha=0.2, label=comp_name)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (a.u.)')
            ax.set_title(f'{roi_name}')
            ax.grid(True, alpha=0.3)

            if idx == 0:
                ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_rois, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Source-Level Event-Related Potentials', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig
