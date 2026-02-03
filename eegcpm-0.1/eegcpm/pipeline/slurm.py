"""
SLURM HPC job submission utilities.
"""

import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from eegcpm.core.config import SlurmConfig
from eegcpm.core.models import Project


logger = logging.getLogger(__name__)


class SlurmJob:
    """Represents a SLURM job."""

    def __init__(
        self,
        job_id: str,
        name: str,
        script_path: Path,
        status: str = "PENDING",
    ):
        self.job_id = job_id
        self.name = name
        self.script_path = script_path
        self.status = status
        self.submitted_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def __repr__(self) -> str:
        return f"SlurmJob({self.job_id}, {self.name}, {self.status})"


class SlurmSubmitter:
    """
    Submit and manage SLURM jobs for EEGCPM pipelines.
    """

    def __init__(
        self,
        config: SlurmConfig,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize SLURM submitter.

        Args:
            config: SLURM configuration
            log_dir: Directory for job logs
        """
        self.config = config
        self.log_dir = log_dir or Path("./slurm_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, SlurmJob] = {}

    def generate_script(
        self,
        job_name: str,
        python_command: str,
        subject_id: Optional[str] = None,
        additional_modules: Optional[List[str]] = None,
    ) -> str:
        """
        Generate SLURM batch script.

        Args:
            job_name: Job name
            python_command: Python command to execute
            subject_id: Optional subject ID for job arrays
            additional_modules: Additional modules to load

        Returns:
            SLURM script content
        """
        modules = self.config.modules.copy()
        if additional_modules:
            modules.extend(additional_modules)

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={self.config.partition}
#SBATCH --time={self.config.time}
#SBATCH --mem={self.config.mem}
#SBATCH --cpus-per-task={self.config.cpus_per_task}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
"""

        if self.config.gpus:
            script += f"#SBATCH --gpus={self.config.gpus}\n"

        script += "\n# Load modules\n"
        for module in modules:
            script += f"module load {module}\n"

        script += f"""
# Activate conda environment if needed
# conda activate eegcpm

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Run command
{python_command}

echo "End time: $(date)"
"""

        return script

    def submit(
        self,
        job_name: str,
        python_command: str,
        dry_run: bool = False,
    ) -> Optional[SlurmJob]:
        """
        Submit a job to SLURM.

        Args:
            job_name: Job name
            python_command: Command to execute
            dry_run: If True, only generate script without submitting

        Returns:
            SlurmJob object or None if dry_run
        """
        script_content = self.generate_script(job_name, python_command)

        # Write script to temp file
        script_path = self.log_dir / f"{job_name}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        if dry_run:
            logger.info(f"Dry run - script saved to {script_path}")
            return None

        # Submit job
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            # Parse job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]

            job = SlurmJob(
                job_id=job_id,
                name=job_name,
                script_path=script_path,
            )
            job.submitted_at = datetime.now()
            self.jobs[job_id] = job

            logger.info(f"Submitted job {job_id}: {job_name}")
            return job

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job: {e.stderr}")
            raise

        except FileNotFoundError:
            logger.error("sbatch command not found - is SLURM installed?")
            raise

    def submit_subject_jobs(
        self,
        project: Project,
        config_path: Path,
        subject_ids: Optional[List[str]] = None,
    ) -> List[SlurmJob]:
        """
        Submit individual jobs for each subject.

        Args:
            project: Project to process
            config_path: Path to config YAML
            subject_ids: Specific subjects (None = all)

        Returns:
            List of submitted jobs
        """
        subjects = project.subjects
        if subject_ids:
            subjects = [s for s in subjects if s.id in subject_ids]

        jobs = []
        for subject in subjects:
            job_name = f"eegcpm_{subject.id}"
            command = (
                f"python -m eegcpm.cli run "
                f"--project {project.root_path / 'project.json'} "
                f"--config {config_path} "
                f"--subject {subject.id}"
            )
            job = self.submit(job_name, command)
            if job:
                jobs.append(job)

        return jobs

    def get_job_status(self, job_id: str) -> str:
        """
        Get status of a submitted job.

        Args:
            job_id: SLURM job ID

        Returns:
            Job status string
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
            )
            status = result.stdout.strip()
            return status if status else "COMPLETED"
        except Exception:
            return "UNKNOWN"

    def update_all_statuses(self) -> None:
        """Update status of all tracked jobs."""
        for job_id, job in self.jobs.items():
            job.status = self.get_job_status(job_id)

    def wait_for_completion(
        self,
        job_ids: Optional[List[str]] = None,
        poll_interval: int = 60,
    ) -> Dict[str, str]:
        """
        Wait for jobs to complete.

        Args:
            job_ids: Jobs to wait for (None = all tracked)
            poll_interval: Seconds between status checks

        Returns:
            Dict mapping job_id to final status
        """
        import time

        if job_ids is None:
            job_ids = list(self.jobs.keys())

        final_statuses = {}

        while job_ids:
            for job_id in job_ids.copy():
                status = self.get_job_status(job_id)

                if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", ""]:
                    final_statuses[job_id] = status or "COMPLETED"
                    job_ids.remove(job_id)
                    if job_id in self.jobs:
                        self.jobs[job_id].status = status
                        self.jobs[job_id].completed_at = datetime.now()

            if job_ids:
                logger.info(f"Waiting for {len(job_ids)} jobs...")
                time.sleep(poll_interval)

        return final_statuses

    def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            subprocess.run(["scancel", job_id], check=True)
            return True
        except Exception:
            return False

    def cancel_all(self) -> None:
        """Cancel all tracked jobs."""
        for job_id in self.jobs:
            self.cancel(job_id)
