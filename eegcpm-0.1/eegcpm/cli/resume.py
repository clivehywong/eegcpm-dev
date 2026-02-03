"""Resume command for EEGCPM CLI."""

from pathlib import Path
from rich.console import Console

from ..workflow.state import WorkflowStateManager, ProcessingStatus


def resume_command(args):
    """
    Resume incomplete/failed workflows.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    console = Console()

    # Get state database path
    db_path = args.project / "derivatives" / ".eegcpm" / "state.db"

    if not db_path.exists():
        console.print(f"[yellow]No workflow state found at {db_path}[/yellow]")
        console.print("Run preprocessing to initialize workflow tracking.")
        return

    # Load state manager
    manager = WorkflowStateManager(db_path)

    # Find workflows to resume
    if args.subject:
        # Resume specific subject
        workflows = manager.get_all_states(
            subject_id=args.subject,
            task=args.task,
            pipeline=args.pipeline
        )
    else:
        # Resume all incomplete/failed
        incomplete = manager.get_all_states(status=ProcessingStatus.IN_PROGRESS)
        failed = manager.get_all_states(status=ProcessingStatus.FAILED)
        workflows = incomplete + failed

    if not workflows:
        console.print("[green]No workflows need resuming. All completed![/green]")
        return

    console.print(f"\n[bold]Resume Workflows[/bold]")
    console.print(f"Found {len(workflows)} workflows to resume:\n")

    for workflow in workflows:
        console.print(f"  [cyan]{workflow.subject_id}[/cyan] / [magenta]{workflow.task}[/magenta] / [blue]{workflow.pipeline}[/blue]")
        console.print(f"    Status: [{_get_status_color(workflow.status)}]{workflow.status.value}[/{_get_status_color(workflow.status)}]")

        completed = workflow.get_completed_steps()
        failed = workflow.get_failed_steps()

        if completed:
            console.print(f"    Completed steps: {', '.join(completed)}")
        if failed:
            console.print(f"    [red]Failed steps: {', '.join(failed)}[/red]")

        # Determine resume point
        if args.from_step:
            resume_from = args.from_step
        elif failed:
            resume_from = failed[0]
        elif completed:
            # Resume from next step after last completed
            all_steps = ["load_data", "preprocessing", "epochs", "source", "connectivity", "features"]
            last_completed = completed[-1]
            try:
                last_idx = all_steps.index(last_completed)
                if last_idx + 1 < len(all_steps):
                    resume_from = all_steps[last_idx + 1]
                else:
                    resume_from = None
            except ValueError:
                resume_from = None
        else:
            resume_from = "load_data"

        if resume_from:
            console.print(f"    [yellow]→ Would resume from: {resume_from}[/yellow]")
        else:
            console.print(f"    [green]→ Workflow complete[/green]")

        console.print()

    console.print("\n[yellow]⚠️  Note: Resume functionality is currently in preview mode.[/yellow]")
    console.print("This command shows workflows that can be resumed, but automatic resumption")
    console.print("is not yet implemented. To reprocess:")
    console.print()
    console.print("  • Use [bold]--force[/bold] flag with preprocessing/epochs/source/connectivity commands")
    console.print("  • Or manually delete failed outputs and re-run the stage")
    console.print()
    console.print("Full resume implementation is planned for v0.2.0")


def _get_status_color(status: ProcessingStatus) -> str:
    """Get color for status display."""
    return {
        ProcessingStatus.COMPLETED: 'green',
        ProcessingStatus.IN_PROGRESS: 'yellow',
        ProcessingStatus.FAILED: 'red',
        ProcessingStatus.PENDING: 'cyan',
        ProcessingStatus.SKIPPED: 'dim'
    }.get(status, 'white')
