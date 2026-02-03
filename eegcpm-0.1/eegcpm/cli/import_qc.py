"""Import QC command for EEGCPM CLI.

Imports QC metrics from JSON files into the workflow state database,
enabling the UI to display processing status and quality metrics.
"""

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from ..workflow.state import WorkflowStateManager
from ..workflow.import_qc import import_qc_metrics_to_state, get_import_summary


def import_qc_command(args):
    """
    Import QC metrics from JSON files to workflow state database.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments with:
            - project: Path to project root
            - derivatives: Optional path to derivatives (default: project/derivatives)
            - pipeline: Pipeline name (required)
            - task: Optional task filter
            - subject: Optional subject filter
            - force: Re-import existing entries
            - dry_run: Preview without importing
    """
    console = Console()

    # Resolve paths
    project_path = Path(args.project).resolve()
    derivatives_path = args.derivatives if args.derivatives else project_path / "derivatives"
    derivatives_path = Path(derivatives_path).resolve()

    # Validate paths
    if not project_path.exists():
        console.print(f"[red]Error: Project path not found: {project_path}[/red]")
        return 1

    if not derivatives_path.exists():
        console.print(f"[red]Error: Derivatives path not found: {derivatives_path}[/red]")
        return 1

    # Setup state manager
    db_path = derivatives_path / ".eegcpm" / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    manager = WorkflowStateManager(db_path)

    console.print(f"\n[bold]EEGCPM QC Import[/bold]")
    console.print(f"Project: {project_path}")
    console.print(f"Derivatives: {derivatives_path}")
    console.print(f"Pipeline: {args.pipeline}")
    console.print(f"State DB: {db_path}\n")

    # Filters
    filters = []
    if args.subject:
        filters.append(f"Subject: {args.subject}")
    if args.task:
        filters.append(f"Task: {args.task}")
    if filters:
        console.print(f"Filters: {', '.join(filters)}")

    # Dry run mode - just show summary
    if args.dry_run:
        console.print("[yellow]Dry run mode - no changes will be made[/yellow]\n")

        summary = get_import_summary(
            derivatives_path=derivatives_path,
            state_manager=manager,
            pipeline=args.pipeline
        )

        # Display summary table
        table = Table(
            title="Import Preview",
            box=box.ROUNDED,
            show_header=False
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total JSON files found", str(summary['total_json_files']))
        table.add_row("New to import", f"[green]{summary['new_to_import']}[/green]")
        table.add_row("Already in database", f"[yellow]{summary['already_in_db']}[/yellow]")
        table.add_row("Unique subjects", str(len(summary['subjects'])))
        table.add_row("Unique tasks", str(len(summary['tasks'])))

        console.print(table)

        if summary['subjects']:
            console.print(f"\nSubjects: {', '.join(summary['subjects'][:10])}")
            if len(summary['subjects']) > 10:
                console.print(f"  ... and {len(summary['subjects']) - 10} more")

        if summary['tasks']:
            console.print(f"Tasks: {', '.join(summary['tasks'])}")

        return 0

    # Perform import
    console.print("[bold]Importing QC metrics from JSON files...[/bold]\n")

    result = import_qc_metrics_to_state(
        derivatives_path=derivatives_path,
        state_manager=manager,
        pipeline=args.pipeline,
        task=args.task,
        subject_id=args.subject,
        force=args.force
    )

    # Display results
    console.print("[green]Import complete![/green]\n")

    # Results table
    table = Table(
        title="Import Results",
        box=box.ROUNDED,
        show_header=False
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Imported", f"[green]{result['imported']}[/green]")
    table.add_row("Skipped (already in DB)", f"[yellow]{result['skipped']}[/yellow]")
    table.add_row("Failed", f"[red]{result['failed']}[/red]" if result['failed'] > 0 else "0")
    table.add_row("Subjects processed", str(len(result['subjects'])))

    console.print(table)

    # Show subjects
    if result['subjects']:
        console.print(f"\nSubjects: {', '.join(result['subjects'][:10])}")
        if len(result['subjects']) > 10:
            console.print(f"  ... and {len(result['subjects']) - 10} more")

    # Show errors if any
    if result['failed'] > 0 and result.get('errors'):
        console.print("\n[red]Errors:[/red]")
        for error in result['errors'][:5]:
            console.print(f"  - {error}")
        if len(result['errors']) > 5:
            console.print(f"  ... and {len(result['errors']) - 5} more errors")

    # Hint for force mode
    if result['skipped'] > 0 and not args.force:
        console.print(
            f"\n[dim]Tip: Use --force to re-import {result['skipped']} "
            "existing entries[/dim]"
        )

    return 0
