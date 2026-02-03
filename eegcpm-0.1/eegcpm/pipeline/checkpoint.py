"""
Checkpoint management for pipeline resume capability.
"""

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eegcpm.pipeline.base import ModuleResult


class CheckpointManager:
    """
    Manage pipeline checkpoints for resume capability.

    Checkpoints are saved after each successful module execution,
    allowing interrupted pipelines to resume from the last completed step.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_subject_dir(self, subject_id: str) -> Path:
        """Get checkpoint directory for a subject."""
        return self.checkpoint_dir / subject_id

    def _get_checkpoint_path(self, subject_id: str, module_name: str) -> Path:
        """Get checkpoint file path."""
        return self._get_subject_dir(subject_id) / f"{module_name}.checkpoint"

    def save(
        self,
        subject_id: str,
        module_name: str,
        result: ModuleResult,
        data: Optional[Any] = None,
    ) -> Path:
        """
        Save checkpoint after module completion.

        Args:
            subject_id: Subject identifier
            module_name: Completed module name
            result: Module result
            data: Optional data to save (will use pickle)

        Returns:
            Path to saved checkpoint
        """
        subject_dir = self._get_subject_dir(subject_id)
        subject_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "subject_id": subject_id,
            "module_name": module_name,
            "timestamp": datetime.now().isoformat(),
            "success": result.success,
            "execution_time": result.execution_time_seconds,
            "output_files": [str(p) for p in result.output_files],
            "metadata": result.metadata,
        }

        # Save JSON metadata
        checkpoint_path = self._get_checkpoint_path(subject_id, module_name)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        # Save data separately if provided
        if data is not None:
            data_path = checkpoint_path.with_suffix(".data")
            with open(data_path, "wb") as f:
                pickle.dump(data, f)

        # Update manifest
        self._update_manifest(subject_id, module_name)

        return checkpoint_path

    def load(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint info for a subject.

        Args:
            subject_id: Subject identifier

        Returns:
            Dict with checkpoint info or None
        """
        manifest_path = self._get_subject_dir(subject_id) / "manifest.json"
        if not manifest_path.exists():
            return None

        with open(manifest_path) as f:
            return json.load(f)

    def load_module_checkpoint(
        self,
        subject_id: str,
        module_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load specific module checkpoint.

        Args:
            subject_id: Subject identifier
            module_name: Module name

        Returns:
            Checkpoint data or None
        """
        checkpoint_path = self._get_checkpoint_path(subject_id, module_name)
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path) as f:
            return json.load(f)

    def load_data(
        self,
        subject_id: str,
        module_name: str,
    ) -> Optional[Any]:
        """
        Load pickled data from checkpoint.

        Args:
            subject_id: Subject identifier
            module_name: Module name

        Returns:
            Loaded data or None
        """
        data_path = self._get_checkpoint_path(subject_id, module_name).with_suffix(".data")
        if not data_path.exists():
            return None

        with open(data_path, "rb") as f:
            return pickle.load(f)

    def get_completed_modules(self, subject_id: str) -> List[str]:
        """
        Get list of completed modules for a subject.

        Args:
            subject_id: Subject identifier

        Returns:
            List of completed module names in order
        """
        checkpoint = self.load(subject_id)
        if checkpoint is None:
            return []
        return checkpoint.get("completed_modules", [])

    def clear(self, subject_id: str) -> None:
        """
        Clear all checkpoints for a subject.

        Args:
            subject_id: Subject identifier
        """
        import shutil
        subject_dir = self._get_subject_dir(subject_id)
        if subject_dir.exists():
            shutil.rmtree(subject_dir)

    def clear_all(self) -> None:
        """Clear all checkpoints."""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _update_manifest(self, subject_id: str, module_name: str) -> None:
        """Update subject manifest with completed module."""
        manifest_path = self._get_subject_dir(subject_id) / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {
                "subject_id": subject_id,
                "completed_modules": [],
                "last_updated": None,
            }

        if module_name not in manifest["completed_modules"]:
            manifest["completed_modules"].append(module_name)

        manifest["last_updated"] = datetime.now().isoformat()

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)


def generate_reproducibility_hash(
    config: Dict[str, Any],
    code_version: str,
    data_checksums: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate a hash for reproducibility tracking.

    Args:
        config: Configuration dict
        code_version: Software version string
        data_checksums: Optional dict of file -> checksum

    Returns:
        SHA256 hash string
    """
    components = {
        "config": json.dumps(config, sort_keys=True),
        "code_version": code_version,
        "data_checksums": data_checksums or {},
    }

    content = json.dumps(components, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
