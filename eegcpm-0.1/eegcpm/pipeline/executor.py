"""
Pipeline executor for running analysis workflows.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from eegcpm.pipeline.base import BaseModule, ModuleResult
from eegcpm.pipeline.checkpoint import CheckpointManager
from eegcpm.core.config import Config
from eegcpm.core.models import Project, Subject


logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Execute analysis pipelines on EEG data.

    Handles:
    - Sequential module execution
    - Checkpoint/resume
    - Error handling
    - Logging
    """

    def __init__(
        self,
        project: Project,
        config: Config,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize executor.

        Args:
            project: Project to process
            config: Pipeline configuration
            output_dir: Base output directory
            checkpoint_dir: Directory for checkpoints
        """
        self.project = project
        self.config = config
        self.output_dir = output_dir or project.root_path / "derivatives"
        self.checkpoint_dir = checkpoint_dir or project.root_path / "checkpoints"

        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.modules: List[BaseModule] = []
        self.results: Dict[str, List[ModuleResult]] = {}

    def add_module(self, module: BaseModule) -> None:
        """Add a module to the pipeline."""
        self.modules.append(module)

    def run(
        self,
        subject_ids: Optional[List[str]] = None,
        resume: bool = True,
        parallel: bool = False,
    ) -> Dict[str, List[ModuleResult]]:
        """
        Run pipeline on specified subjects.

        Args:
            subject_ids: Subjects to process (None = all)
            resume: Resume from checkpoints if available
            parallel: Enable parallel processing (not implemented yet)

        Returns:
            Dict mapping subject_id to list of ModuleResults
        """
        subjects = self.project.subjects
        if subject_ids:
            subjects = [s for s in subjects if s.id in subject_ids]

        logger.info(f"Running pipeline on {len(subjects)} subjects")
        logger.info(f"Pipeline has {len(self.modules)} modules")

        for subject in subjects:
            logger.info(f"Processing subject: {subject.id}")

            # Check for checkpoint
            if resume:
                checkpoint = self.checkpoint_manager.load(subject.id)
                if checkpoint:
                    logger.info(f"Resuming from checkpoint for {subject.id}")
                    # Skip completed modules
                    # TODO: Implement checkpoint resume logic
                    pass

            subject_results = []
            current_data = None

            for module in self.modules:
                try:
                    logger.info(f"Running module: {module.name}")
                    start_time = time.time()

                    # Get input data for module
                    if current_data is None:
                        current_data = self._load_initial_data(subject)

                    # Validate input
                    module.validate_input(current_data)

                    # Process
                    result = module.process(current_data, subject=subject)

                    result.execution_time_seconds = time.time() - start_time
                    subject_results.append(result)

                    if result.success:
                        # Update current_data for next module
                        if "data" in result.outputs:
                            current_data = result.outputs["data"]

                        # Save checkpoint
                        self.checkpoint_manager.save(
                            subject.id,
                            module.name,
                            result,
                        )
                    else:
                        logger.error(f"Module {module.name} failed: {result.errors}")
                        break

                except Exception as e:
                    logger.exception(f"Error in module {module.name}")
                    result = ModuleResult(
                        success=False,
                        module_name=module.name,
                        execution_time_seconds=time.time() - start_time,
                        errors=[str(e)],
                    )
                    subject_results.append(result)
                    break

            self.results[subject.id] = subject_results

        return self.results

    def _load_initial_data(self, subject: Subject) -> Any:
        """Load initial data for a subject."""
        from eegcpm.data.loaders import load_raw

        # Get first run of first session
        if not subject.sessions or not subject.sessions[0].runs:
            raise ValueError(f"No data found for subject {subject.id}")

        run = subject.sessions[0].runs[0]
        return load_raw(run.eeg_file)

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        total_subjects = len(self.results)
        successful = sum(
            1 for results in self.results.values()
            if all(r.success for r in results)
        )
        failed = total_subjects - successful

        return {
            "total_subjects": total_subjects,
            "successful": successful,
            "failed": failed,
            "modules_run": len(self.modules),
            "results": self.results,
        }


def create_standard_pipeline(
    project: Project,
    config: Config,
) -> PipelineExecutor:
    """
    Create a standard CPM pipeline.

    Includes:
    1. Preprocessing (filtering, ICA)
    2. Epoch extraction
    3. Source reconstruction
    4. Connectivity analysis
    5. Feature extraction

    Args:
        project: Project to process
        config: Pipeline configuration

    Returns:
        Configured PipelineExecutor
    """
    from eegcpm.modules.preprocessing import PreprocessingModule
    from eegcpm.modules.epochs import EpochExtractionModule
    from eegcpm.modules.source import SourceReconstructionModule
    from eegcpm.modules.connectivity import ConnectivityModule
    from eegcpm.modules.features import FeatureExtractionModule

    executor = PipelineExecutor(project, config)

    # Add modules
    base_output = project.root_path / "derivatives"

    executor.add_module(PreprocessingModule(
        config=config.preprocessing.model_dump(),
        output_dir=base_output / "preprocessed",
    ))

    executor.add_module(EpochExtractionModule(
        config=config.epochs.model_dump(),
        output_dir=base_output / "epochs",
    ))

    executor.add_module(SourceReconstructionModule(
        config=config.source.model_dump(),
        output_dir=base_output / "source",
    ))

    executor.add_module(ConnectivityModule(
        config=config.connectivity.model_dump(),
        output_dir=base_output / "connectivity",
    ))

    executor.add_module(FeatureExtractionModule(
        config={},  # Uses connectivity config
        output_dir=base_output / "features",
    ))

    return executor
