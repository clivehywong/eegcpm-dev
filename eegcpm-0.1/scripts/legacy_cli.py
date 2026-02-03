#!/usr/bin/env python3
"""EEGCPM Preprocessing CLI"""

import argparse
import sys
import yaml
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from eegcpm.pipeline import RunProcessor


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    logging.basicConfig(level=level, handlers=handlers, force=True)


def load_config(config_path: Path) -> dict:
    """Load config from YAML."""
    if not config_path.exists():
        default_config = Path(__file__).parent / 'config' / 'preprocessing' / config_path.name
        if default_config.exists():
            config_path = default_config
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded config: {config_path}")
    return config


def load_subjects_list(subjects_file: Path) -> List[str]:
    """Load subject IDs from file."""
    if not subjects_file.exists():
        raise FileNotFoundError(f"File not found: {subjects_file}")
    
    with open(subjects_file) as f:
        subjects = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    logging.info(f"Loaded {len(subjects)} subjects")
    return subjects


def process_subject(
    subject_id: str,
    bids_root: Path,
    output_root: Path,
    config: dict,
    task: str,
    session: str = "01",
    pipeline: str = "cli-preprocessing"
) -> bool:
    """Process one subject/task."""
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Processing: {subject_id} - {task}")
    logging.info(f"{'=' * 80}\n")
    
    try:
        processor = RunProcessor(
            bids_root=bids_root,
            output_root=output_root,
            config=config,
            state_manager=None,
            verbose=True
        )
        
        results = processor.process_subject_task(
            subject_id=subject_id,
            task=task,
            session=session,
            pipeline=pipeline
        )
        
        n_success = sum(1 for r in results if r.success)
        n_failed = len(results) - n_success
        
        logging.info(f"\n{subject_id}/{task}: {n_success}/{len(results)} runs successful")
        
        return n_failed == 0
    
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='EEGCPM Preprocessing CLI')
    
    parser.add_argument('--bids-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument('--subject', type=str)
    subject_group.add_argument('--subjects-file', type=Path)
    
    parser.add_argument('--task', type=str, required=True, help='Task name (required)')
    parser.add_argument('--session', type=str, default='01')
    parser.add_argument('--pipeline', type=str, default='cli-preprocessing')
    parser.add_argument('--config', type=Path, default='standard.yaml')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--log-file', type=Path, default=None)
    parser.add_argument('--stop-on-error', action='store_true')
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    
    logging.info("EEGCPM Preprocessing CLI")
    logging.info(f"Started: {datetime.now().isoformat()}")
    
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Config load failed: {e}")
        return 3
    
    if args.subject:
        subjects = [args.subject.replace('sub-', '')]
    else:
        try:
            subjects = load_subjects_list(args.subjects_file)
        except Exception as e:
            logging.error(f"Subjects file failed: {e}")
            return 1
    
    n_success = 0
    n_failed = 0
    
    for i, subject_id in enumerate(subjects, 1):
        logging.info(f"\n[{i}/{len(subjects)}] {subject_id}")
        
        success = process_subject(
            subject_id=subject_id,
            bids_root=args.bids_root,
            output_root=args.output,
            config=config,
            task=args.task,
            session=args.session,
            pipeline=args.pipeline,
        )
        
        if success:
            n_success += 1
        else:
            n_failed += 1
            if args.stop_on_error:
                break
    
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Total: {len(subjects)}, Success: {n_success}, Failed: {n_failed}")
    
    return 0 if n_failed == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
