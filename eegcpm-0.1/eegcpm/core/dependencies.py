"""Dependency resolution for stage-first architecture.

This module provides functions to validate and resolve dependencies between
processing stages, ensuring that required upstream outputs exist before
downstream processing begins.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from .config import BaseStageConfig
from .paths import EEGCPMPaths


class DependencyError(Exception):
    """Raised when a dependency cannot be resolved."""
    pass


class MissingDependencyError(DependencyError):
    """Raised when a required upstream output does not exist."""
    pass


class CircularDependencyError(DependencyError):
    """Raised when circular dependencies are detected."""
    pass


@dataclass
class ResolvedDependency:
    """A resolved dependency path.

    Attributes
    ----------
    stage : str
        Stage name (e.g., "preprocessing")
    variant : str
        Variant name (e.g., "standard")
    path : Path
        Resolved path to dependency output
    exists : bool
        Whether the dependency exists on disk
    """
    stage: str
    variant: str
    path: Path
    exists: bool


def resolve_dependencies(
    config: BaseStageConfig,
    paths: EEGCPMPaths,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    validate_exists: bool = True
) -> Dict[str, ResolvedDependency]:
    """
    Resolve and validate dependencies for a stage configuration.

    Parameters
    ----------
    config : BaseStageConfig
        Stage configuration with dependencies
    paths : EEGCPMPaths
        Path manager instance
    subject : str, optional
        Subject ID (required for subject-level validation)
    session : str, optional
        Session ID
    task : str, optional
        Task name
    run : str, optional
        Run ID
    validate_exists : bool, default True
        If True, raise error if dependency doesn't exist

    Returns
    -------
    Dict[str, ResolvedDependency]
        Dictionary mapping dependency name to resolved path

    Raises
    ------
    MissingDependencyError
        If validate_exists=True and dependency doesn't exist
    DependencyError
        If dependency cannot be resolved

    Examples
    --------
    >>> paths = EEGCPMPaths(Path("/data/study"))
    >>> config = EpochsConfig(
    ...     stage="epochs",
    ...     name="standard",
    ...     depends_on={"preprocessing": "standard", "task": "rest"}
    ... )
    >>> deps = resolve_dependencies(
    ...     config, paths,
    ...     subject="sub-001",
    ...     session="01",
    ...     task="rest"
    ... )
    >>> print(deps["preprocessing"].path)
    /data/study/derivatives/preprocessing/standard/sub-001/ses-01/task-rest/
    """
    if not config.depends_on:
        return {}

    resolved = {}

    for dep_stage, dep_variant in config.depends_on.items():
        # Resolve path based on stage type
        if dep_stage == "preprocessing":
            if not all([subject, session, task, run]):
                raise DependencyError(
                    f"Preprocessing dependency requires subject, session, task, and run. "
                    f"Got: subject={subject}, session={session}, task={task}, run={run}"
                )
            dep_path = paths.get_preprocessing_dir(
                pipeline=dep_variant,
                subject=subject,
                session=session,
                task=task,
                run=run
            )

        elif dep_stage == "epochs":
            if not all([subject, session, task]):
                raise DependencyError(
                    f"Epochs dependency requires subject, session, and task. "
                    f"Got: subject={subject}, session={session}, task={task}"
                )
            # For epochs, we need the preprocessing variant from config
            preprocessing = config.depends_on.get("preprocessing")
            if not preprocessing:
                raise DependencyError(
                    "Epochs dependency requires preprocessing variant in depends_on"
                )
            dep_path = paths.get_epochs_dir(
                preprocessing=preprocessing,
                task=task,
                subject=subject,
                session=session
            )

        elif dep_stage == "source":
            if not all([subject, session, task]):
                raise DependencyError(
                    f"Source dependency requires subject, session, and task. "
                    f"Got: subject={subject}, session={session}, task={task}"
                )
            # For source, we need preprocessing variant
            preprocessing = config.depends_on.get("preprocessing")
            if not preprocessing:
                raise DependencyError(
                    "Source dependency requires preprocessing variant in depends_on"
                )
            dep_path = paths.get_source_dir(
                preprocessing=preprocessing,
                task=task,
                variant=dep_variant,
                subject=subject,
                session=session
            )

        elif dep_stage == "features":
            # Features are task/preprocessing/source-variant level
            if not task:
                raise DependencyError(
                    f"Features dependency requires task. Got: task={task}"
                )
            preprocessing = config.depends_on.get("preprocessing")
            source_variant = config.depends_on.get("source")
            if not all([preprocessing, source_variant]):
                raise DependencyError(
                    "Features dependency requires preprocessing and source variants in depends_on"
                )
            dep_path = paths.get_features_dir(
                preprocessing=preprocessing,
                task=task,
                source_variant=source_variant,
                feature_type=dep_variant
            )

        elif dep_stage == "task":
            # Task is a pseudo-dependency that doesn't resolve to a path
            # It's just used to specify which task this config applies to
            resolved[dep_stage] = ResolvedDependency(
                stage=dep_stage,
                variant=dep_variant,
                path=Path(),  # Empty path
                exists=True  # Always exists (it's just metadata)
            )
            continue

        else:
            raise DependencyError(f"Unknown dependency stage: {dep_stage}")

        # Check if dependency exists
        exists = dep_path.exists()

        if validate_exists and not exists:
            raise MissingDependencyError(
                f"Required {dep_stage} dependency not found: {dep_path}\n"
                f"Stage: {config.stage}, Variant: {config.name}\n"
                f"Dependency: {dep_stage}={dep_variant}"
            )

        resolved[dep_stage] = ResolvedDependency(
            stage=dep_stage,
            variant=dep_variant,
            path=dep_path,
            exists=exists
        )

    return resolved


def check_dependency_chain(
    config: BaseStageConfig,
    visited: Optional[List[str]] = None
) -> bool:
    """
    Check for circular dependencies in a configuration chain.

    Parameters
    ----------
    config : BaseStageConfig
        Configuration to check
    visited : list, optional
        Already visited stages (for recursion tracking)

    Returns
    -------
    bool
        True if no circular dependencies detected

    Raises
    ------
    CircularDependencyError
        If circular dependency is detected
    """
    if visited is None:
        visited = []

    stage_id = f"{config.stage}:{config.name}"

    if stage_id in visited:
        chain = " -> ".join(visited + [stage_id])
        raise CircularDependencyError(
            f"Circular dependency detected: {chain}"
        )

    visited.append(stage_id)

    # In practice, circular dependencies are unlikely in a stage-first
    # architecture since stages have a natural order, but we check anyway
    if config.depends_on:
        for dep_stage, dep_variant in config.depends_on.items():
            if dep_stage == "task":
                continue  # Skip pseudo-dependencies
            # Note: We would need to load the dependency config to recurse
            # For now, we just check for self-loops
            dep_id = f"{dep_stage}:{dep_variant}"
            if dep_id == stage_id:
                raise CircularDependencyError(
                    f"Self-loop detected: {stage_id} depends on itself"
                )

    return True


def lock_config(
    config: BaseStageConfig,
    output_dir: Path
) -> Path:
    """
    Lock configuration to an output directory.

    Saves the configuration as .config_locked.yaml in the output directory
    on first run. On subsequent runs, validates that the same config is used.

    Parameters
    ----------
    config : BaseStageConfig
        Configuration to lock
    output_dir : Path
        Output directory to lock config in

    Returns
    -------
    Path
        Path to locked config file

    Raises
    ------
    DependencyError
        If locked config exists but doesn't match current config

    Notes
    -----
    This prevents accidentally reusing an output directory with different
    preprocessing settings, which would lead to inconsistent results.
    """
    import yaml
    import hashlib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lock_file = output_dir / ".config_locked.yaml"

    # Compute hash of current config
    config_dict = config.model_dump()
    config_str = yaml.dump(config_dict, sort_keys=True)
    current_hash = hashlib.sha256(config_str.encode()).hexdigest()

    if lock_file.exists():
        # Load existing lock
        with open(lock_file, 'r') as f:
            locked = yaml.safe_load(f)

        locked_hash = locked.get('_config_hash')

        if locked_hash != current_hash:
            raise DependencyError(
                f"Config mismatch: Output directory {output_dir} was created "
                f"with a different configuration.\n"
                f"Either delete the directory or use the original config.\n"
                f"Locked config: {lock_file}"
            )
    else:
        # Create new lock
        config_dict['_config_hash'] = current_hash
        config_dict['_locked_at'] = str(Path.cwd())

        with open(lock_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    return lock_file


def get_dependency_summary(
    config: BaseStageConfig,
    paths: EEGCPMPaths,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None
) -> str:
    """
    Get a human-readable summary of dependencies.

    Parameters
    ----------
    config : BaseStageConfig
        Configuration with dependencies
    paths : EEGCPMPaths
        Path manager
    subject, session, task, run : str, optional
        Subject/session/task/run identifiers

    Returns
    -------
    str
        Formatted dependency summary
    """
    if not config.depends_on:
        return "No dependencies"

    try:
        resolved = resolve_dependencies(
            config, paths,
            subject=subject,
            session=session,
            task=task,
            run=run,
            validate_exists=False
        )

        lines = [f"Dependencies for {config.stage}:{config.name}"]
        lines.append("-" * 60)

        for dep_name, dep in resolved.items():
            status = "✓" if dep.exists else "✗"
            lines.append(f"  {status} {dep_name}: {dep.variant}")
            if dep.path != Path():
                lines.append(f"      Path: {dep.path}")
                if not dep.exists:
                    lines.append("      Status: MISSING")

        return "\n".join(lines)

    except DependencyError as e:
        return f"Error resolving dependencies: {e}"
