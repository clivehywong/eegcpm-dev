"""
Multi-Pipeline Comparison Module

Compares multiple preprocessing pipelines on the same data:
- Standard MNE-Python pipeline
- EEGPrep 3-step pipeline
- Custom pipelines with different parameters

Generates comprehensive comparison reports showing differences in:
- Channel exclusion
- Artifact removal
- Signal characteristics
- Processing statistics

Author: EEGCPM Development Team
Date: 2025-12-01
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np

from .preprocessing_comparison_qc import PreprocessingComparisonQC
from .base import QCResult
from .html_report import HTMLReportBuilder, QCIndexBuilder

try:
    import eegprep
    EEGPREP_AVAILABLE = True
except ImportError:
    EEGPREP_AVAILABLE = False


@dataclass
class PipelineConfig:
    """Configuration for a single preprocessing pipeline."""
    name: str
    pipeline_type: str  # 'mne', 'eegprep', or 'custom'
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""


@dataclass
class PipelineResult:
    """Result from running a single pipeline."""
    name: str
    pipeline_type: str
    raw_processed: Optional[mne.io.Raw] = None
    raw_annotated: Optional[mne.io.Raw] = None
    success: bool = False
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    execution_time_s: float = 0.0


class MultiPipelineComparison:
    """
    Compare multiple preprocessing pipelines on the same data.

    Supports:
    - MNE-Python pipelines with different parameters
    - EEGPrep pipeline (if available)
    - Side-by-side comparison plots
    - Statistical summaries
    """

    def __init__(
        self,
        pipelines: List[PipelineConfig],
        output_dir: Path,
        comparison_config: Optional[Dict] = None
    ):
        """
        Initialize multi-pipeline comparison.

        Args:
            pipelines: List of pipeline configurations to compare
            output_dir: Base output directory
            comparison_config: QC comparison configuration
        """
        self.pipelines = [p for p in pipelines if p.enabled]
        self.output_dir = Path(output_dir)
        self.comparison_config = comparison_config or {}

        # Create subdirectories for each pipeline
        for pipeline in self.pipelines:
            (self.output_dir / pipeline.name).mkdir(parents=True, exist_ok=True)

        self.qc_dir = self.output_dir / "qc"
        self.qc_dir.mkdir(parents=True, exist_ok=True)

    def run_all_pipelines(
        self,
        raw_original: mne.io.Raw,
        subject_id: str,
        **kwargs
    ) -> List[PipelineResult]:
        """
        Run all configured pipelines on the same data.

        Args:
            raw_original: Original unprocessed data
            subject_id: Subject identifier
            **kwargs: Additional arguments (task, session, etc.)

        Returns:
            List of PipelineResult objects
        """
        results = []

        for pipeline in self.pipelines:
            print(f"\n{'='*60}")
            print(f"Running Pipeline: {pipeline.name}")
            print(f"Type: {pipeline.pipeline_type}")
            if pipeline.description:
                print(f"Description: {pipeline.description}")
            print(f"{'='*60}")

            try:
                if pipeline.pipeline_type == 'mne':
                    result = self._run_mne_pipeline(
                        raw_original.copy(),
                        pipeline,
                        subject_id,
                        **kwargs
                    )
                elif pipeline.pipeline_type == 'eegprep':
                    result = self._run_eegprep_pipeline(
                        raw_original.copy(),
                        pipeline,
                        subject_id,
                        **kwargs
                    )
                else:
                    raise ValueError(f"Unknown pipeline type: {pipeline.pipeline_type}")

                results.append(result)
                print(f"✓ {pipeline.name} completed successfully")

            except Exception as e:
                print(f"✗ {pipeline.name} failed: {e}")
                results.append(PipelineResult(
                    name=pipeline.name,
                    pipeline_type=pipeline.pipeline_type,
                    success=False,
                    error=str(e)
                ))

        return results

    def _run_mne_pipeline(
        self,
        raw: mne.io.Raw,
        pipeline: PipelineConfig,
        subject_id: str,
        **kwargs
    ) -> PipelineResult:
        """Run MNE-Python based pipeline."""
        import time
        from eegcpm.modules.preprocessing import PreprocessingModule

        start_time = time.time()
        result = PipelineResult(
            name=pipeline.name,
            pipeline_type='mne'
        )

        # Create subject object
        class Subject:
            def __init__(self, id):
                self.id = id
        subject = Subject(subject_id)

        # Run preprocessing
        output_dir = self.output_dir / pipeline.name
        preprocessor = PreprocessingModule(
            config=pipeline.parameters,
            output_dir=output_dir
        )

        preprocess_result = preprocessor.process(data=raw, subject=subject)

        if preprocess_result.success:
            result.raw_processed = preprocess_result.outputs['data']
            result.raw_annotated = preprocess_result.outputs['data'].copy()
            result.success = True
            result.stats = preprocess_result.metadata
        else:
            result.success = False
            result.error = str(preprocess_result.errors)

        result.execution_time_s = time.time() - start_time
        return result

    def _run_eegprep_pipeline(
        self,
        raw: mne.io.Raw,
        pipeline: PipelineConfig,
        subject_id: str,
        **kwargs
    ) -> PipelineResult:
        """Run EEGPrep-based pipeline."""
        import time

        if not EEGPREP_AVAILABLE:
            raise ImportError("eegprep not installed")

        start_time = time.time()
        result = PipelineResult(
            name=pipeline.name,
            pipeline_type='eegprep'
        )

        # Use PreprocessingComparisonQC's EEGPrep implementation
        qc = PreprocessingComparisonQC(
            output_dir=self.qc_dir,
            config=pipeline.parameters
        )

        try:
            raw_processed, raw_annotated, stats = qc._preprocess_with_eegprep(raw)
            result.raw_processed = raw_processed
            result.raw_annotated = raw_annotated
            result.stats = stats
            result.success = True
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.execution_time_s = time.time() - start_time
        return result

    def generate_comparison_report(
        self,
        raw_original: mne.io.Raw,
        results: List[PipelineResult],
        subject_id: str,
        task: str = "unknown"
    ) -> Path:
        """
        Generate comprehensive comparison report.

        Args:
            raw_original: Original unprocessed data
            results: List of pipeline results
            subject_id: Subject identifier
            task: Task name

        Returns:
            Path to generated HTML report
        """
        # Generate pairwise comparisons
        successful_results = [r for r in results if r.success]

        if len(successful_results) < 2:
            print(f"Warning: Only {len(successful_results)} successful pipelines, need >= 2 for comparison")
            return None

        print(f"\nGenerating comparison report for {len(successful_results)} pipelines...")

        # Create comparison QC
        qc = PreprocessingComparisonQC(
            output_dir=self.qc_dir,
            config=self.comparison_config
        )

        # For multi-pipeline comparison, we'll create a custom plot
        # comparing all pipelines simultaneously
        fig = self._generate_multi_pipeline_plot(
            raw_original,
            successful_results,
            subject_id,
            task
        )

        # Convert to bytes
        fig_bytes = qc.fig_to_base64(fig, dpi=self.comparison_config.get('figure_dpi', 150))
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Build HTML report
        html_builder = HTMLReportBuilder(
            title=f"Multi-Pipeline Comparison - {subject_id} - {task}"
        )

        # Add summary table
        html_builder.add_header("Pipeline Summary", level=2)
        summary_html = self._generate_summary_table(results)
        html_builder.add_raw_html(summary_html)

        # Add comparison figure
        html_builder.add_header("Visual Comparison", level=2)
        data_uri = qc.bytes_to_data_uri(fig_bytes)
        html_builder.add_raw_html(f'<img src="{data_uri}" style="width:100%; max-width:1800px;"/>')

        # Add detailed statistics
        html_builder.add_header("Processing Statistics", level=2)
        stats_html = self._generate_stats_table(results)
        html_builder.add_raw_html(stats_html)

        # Save report
        html_path = self.qc_dir / f"{subject_id}_{task}_multi_pipeline_comparison.html"
        html_builder.save(html_path)

        print(f"✓ Comparison report saved: {html_path}")
        return html_path

    def _generate_multi_pipeline_plot(
        self,
        raw_original: mne.io.Raw,
        results: List[PipelineResult],
        subject_id: str,
        task: str
    ):
        """Generate multi-pipeline comparison plot."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        n_pipelines = len(results)
        fig = plt.figure(figsize=(20, 6 * n_pipelines))
        gs = GridSpec(n_pipelines, 3, figure=fig, hspace=0.4, wspace=0.3)

        fig.suptitle(
            f'{subject_id} - {task}\nMulti-Pipeline Comparison ({n_pipelines} pipelines)',
            fontsize=16,
            fontweight='bold'
        )

        qc = PreprocessingComparisonQC(output_dir=self.qc_dir, config={})

        for i, result in enumerate(results):
            # Channel locations
            ax = fig.add_subplot(gs[i, 0])
            excluded = list(set(raw_original.ch_names) - set(result.raw_processed.ch_names))
            qc._plot_channel_locations(
                raw_original, ax,
                f"{result.name}\n({len(result.raw_processed.ch_names)} kept, {len(excluded)} excluded)",
                excluded_channels=excluded
            )

            # Statistical comparison (violin plot)
            ax = fig.add_subplot(gs[i, 1])
            qc._plot_statistical_comparison(
                raw_original, None, result.raw_processed, ax
            )

            # Variance comparison
            ax = fig.add_subplot(gs[i, 2])
            qc._plot_variance_comparison(
                raw_original, None, result.raw_processed, ax
            )

        return fig

    def _generate_summary_table(self, results: List[PipelineResult]) -> str:
        """Generate HTML summary table."""
        rows = []
        for result in results:
            status = "✓ Success" if result.success else f"✗ Failed: {result.error}"
            n_channels = len(result.raw_processed.ch_names) if result.success else "N/A"
            exec_time = f"{result.execution_time_s:.1f}s" if result.success else "N/A"

            rows.append(f"""
                <tr>
                    <td><strong>{result.name}</strong></td>
                    <td>{result.pipeline_type}</td>
                    <td>{status}</td>
                    <td>{n_channels}</td>
                    <td>{exec_time}</td>
                </tr>
            """)

        return f"""
        <table>
            <thead>
                <tr>
                    <th>Pipeline Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Channels Remaining</th>
                    <th>Execution Time</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _generate_stats_table(self, results: List[PipelineResult]) -> str:
        """Generate detailed statistics table."""
        rows = []
        for result in results:
            if not result.success:
                continue

            stats = result.stats
            rows.append(f"""
                <tr>
                    <td><strong>{result.name}</strong></td>
                    <td>{stats.get('n_channels_before', 'N/A')}</td>
                    <td>{stats.get('n_channels_after', len(result.raw_processed.ch_names))}</td>
                    <td>{stats.get('channels_removed', 'N/A')}</td>
                    <td>{stats.get('ica_components', 'N/A')}</td>
                    <td>{stats.get('ica_components_removed', 'N/A')}</td>
                    <td>{stats.get('data_removed_pct', 0):.1f}%</td>
                </tr>
            """)

        return f"""
        <table>
            <thead>
                <tr>
                    <th>Pipeline</th>
                    <th>Channels Before</th>
                    <th>Channels After</th>
                    <th>Channels Removed</th>
                    <th>ICA Components</th>
                    <th>ICA Removed</th>
                    <th>Data Removed</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """


def create_pipeline_configs_from_yaml(config_dict: Dict) -> List[PipelineConfig]:
    """
    Create list of PipelineConfig objects from YAML configuration.

    Args:
        config_dict: Dictionary from YAML file with 'pipelines' key

    Returns:
        List of PipelineConfig objects
    """
    pipelines = []

    for pipeline_dict in config_dict.get('pipelines', []):
        pipeline = PipelineConfig(
            name=pipeline_dict['name'],
            pipeline_type=pipeline_dict['type'],
            parameters=pipeline_dict.get('parameters', {}),
            enabled=pipeline_dict.get('enabled', True),
            description=pipeline_dict.get('description', '')
        )
        pipelines.append(pipeline)

    return pipelines
