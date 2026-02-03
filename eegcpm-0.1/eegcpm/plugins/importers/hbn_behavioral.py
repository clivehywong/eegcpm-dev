"""
Behavioral data parsers for HBN paradigms.

Each paradigm has its own parser class that extracts trial-by-trial behavioral data
from MAT files and generates summary statistics.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrialData:
    """Single trial behavioral data."""

    trial_number: int
    onset_seconds: float
    duration_seconds: float = 0.0

    # Stimulus info
    stimulus_type: Optional[str] = None
    stimulus_side: Optional[str] = None  # "left" | "right"

    # Response info
    response: Optional[str] = None  # "left" | "right" | "none"
    response_time: Optional[float] = None  # RT in seconds
    correct: Optional[bool] = None

    # Task-specific extras
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        d = {
            "trial_number": self.trial_number,
            "onset": self.onset_seconds,
            "duration": self.duration_seconds,
        }
        if self.stimulus_type is not None:
            d["stimulus_type"] = self.stimulus_type
        if self.stimulus_side is not None:
            d["stimulus_side"] = self.stimulus_side
        if self.response is not None:
            d["response"] = self.response
        if self.response_time is not None:
            d["response_time"] = self.response_time
        if self.correct is not None:
            d["correct"] = int(self.correct)
        d.update(self.extras)
        return d


@dataclass
class ParadigmSummary:
    """Summary statistics for a paradigm."""

    subject_id: str
    paradigm: str
    block: Optional[str] = None

    n_trials: int = 0
    n_responses: int = 0
    n_correct: int = 0
    n_missed: int = 0

    accuracy: Optional[float] = None
    mean_rt: Optional[float] = None
    median_rt: Optional[float] = None
    std_rt: Optional[float] = None
    min_rt: Optional[float] = None
    max_rt: Optional[float] = None

    usable: bool = True
    exclusion_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "subject_id": self.subject_id,
            "paradigm": self.paradigm,
            "block": self.block,
            "n_trials": self.n_trials,
            "n_responses": self.n_responses,
            "n_correct": self.n_correct,
            "n_missed": self.n_missed,
            "accuracy": self.accuracy,
            "mean_rt": self.mean_rt,
            "median_rt": self.median_rt,
            "std_rt": self.std_rt,
            "min_rt": self.min_rt,
            "max_rt": self.max_rt,
            "usable": self.usable,
            "exclusion_reasons": "; ".join(self.exclusion_reasons) if self.exclusion_reasons else None,
        }


@dataclass
class BehavioralResult:
    """Result of parsing behavioral data."""

    subject_id: str
    paradigm: str
    success: bool
    trials: List[TrialData] = field(default_factory=list)
    summary: Optional[ParadigmSummary] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)


class BaseBehavioralParser(ABC):
    """Abstract base class for behavioral data parsers."""

    paradigm_name: str = "base"

    # QC thresholds
    MIN_ACCURACY = 0.5  # Chance level for 2AFC
    MAX_MISSED_RATE = 0.3
    MIN_RT = 0.1  # 100ms
    MAX_RT = 3.0  # 3000ms
    MIN_RT_STD = 0.05  # Detect button mashing

    def __init__(self, subject_id: str):
        self.subject_id = subject_id

    @abstractmethod
    def parse(self, mat_data: Dict[str, Any], block: Optional[str] = None) -> BehavioralResult:
        """Parse behavioral data from MAT file contents."""
        pass

    def compute_summary(
        self,
        trials: List[TrialData],
        paradigm: str,
        block: Optional[str] = None,
    ) -> ParadigmSummary:
        """Compute summary statistics from trials."""
        summary = ParadigmSummary(
            subject_id=self.subject_id,
            paradigm=paradigm,
            block=block,
            n_trials=len(trials),
        )

        if not trials:
            summary.usable = False
            summary.exclusion_reasons.append("No trials")
            return summary

        # Count responses and correct
        rts = []
        for trial in trials:
            if trial.response is not None and trial.response != "none":
                summary.n_responses += 1
                if trial.response_time is not None:
                    rts.append(trial.response_time)
            else:
                summary.n_missed += 1

            if trial.correct is True:
                summary.n_correct += 1

        # Compute accuracy
        if summary.n_responses > 0:
            summary.accuracy = summary.n_correct / summary.n_trials

        # Compute RT statistics
        if rts:
            rts = np.array(rts)
            summary.mean_rt = float(np.mean(rts))
            summary.median_rt = float(np.median(rts))
            summary.std_rt = float(np.std(rts))
            summary.min_rt = float(np.min(rts))
            summary.max_rt = float(np.max(rts))

        # QC checks
        self._apply_qc_checks(summary)

        return summary

    def _apply_qc_checks(self, summary: ParadigmSummary):
        """Apply QC criteria to determine usability."""
        # Check accuracy
        if summary.accuracy is not None and summary.accuracy < self.MIN_ACCURACY:
            summary.exclusion_reasons.append(f"Low accuracy: {summary.accuracy:.2%}")

        # Check missed rate
        if summary.n_trials > 0:
            missed_rate = summary.n_missed / summary.n_trials
            if missed_rate > self.MAX_MISSED_RATE:
                summary.exclusion_reasons.append(f"High miss rate: {missed_rate:.2%}")

        # Check RT validity
        if summary.mean_rt is not None:
            if summary.mean_rt < self.MIN_RT:
                summary.exclusion_reasons.append(f"RT too fast: {summary.mean_rt:.3f}s")
            if summary.mean_rt > self.MAX_RT:
                summary.exclusion_reasons.append(f"RT too slow: {summary.mean_rt:.3f}s")

        # Check RT variability (button mashing detection)
        if summary.std_rt is not None and summary.std_rt < self.MIN_RT_STD:
            summary.exclusion_reasons.append(f"Low RT variability: {summary.std_rt:.3f}s")

        summary.usable = len(summary.exclusion_reasons) == 0


class SAIIT2AFCParser(BaseBehavioralParser):
    """
    Parser for SAIIT_2AFC (Sound-induced Illusory Flash) paradigm.

    MAT file structure:
    - TargOnT: Target onset times (24 trials per block)
    - RespT: Response times (may have more entries than trials)
    - RespLR: Response side (1=left, 2=right)
    - trialLR: Correct answer (1=left, 2=right)
    - trialITI: Inter-trial intervals
    - ITIstartT: ITI start times

    Note: Behavioral timestamps are Unix epoch times. We convert them
    to be relative to the first stimulus onset (trial 1 = 0s).
    The actual alignment to EEG recording time requires the paradigm
    start trigger from the EEG data.
    """

    paradigm_name = "saiit2afc"

    def parse(
        self,
        mat_data: Dict[str, Any],
        block: Optional[str] = None,
        time_offset: float = 0.0,
    ) -> BehavioralResult:
        result = BehavioralResult(
            subject_id=self.subject_id,
            paradigm=self.paradigm_name,
            success=False,
        )

        try:
            # Extract arrays
            targ_on_t = np.atleast_1d(mat_data.get("TargOnT", []))
            resp_t = np.atleast_1d(mat_data.get("RespT", []))
            resp_lr = np.atleast_1d(mat_data.get("RespLR", []))
            trial_lr = np.atleast_1d(mat_data.get("trialLR", []))
            trial_iti = np.atleast_1d(mat_data.get("trialITI", []))

            if len(targ_on_t) == 0:
                result.errors.append("No target onset times found")
                return result

            n_trials = len(targ_on_t)

            # Convert Unix timestamps to relative times
            # Detect if timestamps are Unix epoch (> 1 billion seconds)
            first_onset = targ_on_t[0]
            is_unix_timestamp = first_onset > 1e9

            if is_unix_timestamp:
                # Convert to relative times (first trial = time_offset)
                targ_on_t = targ_on_t - first_onset + time_offset
                resp_t = resp_t - first_onset + time_offset
                result.raw_data["timestamp_conversion"] = "unix_to_relative"
                result.raw_data["original_first_onset"] = float(first_onset)

            result.raw_data.update({
                "n_targets": n_trials,
                "n_responses": len(resp_t),
                "n_trial_lr": len(trial_lr),
            })

            # Match responses to trials by time
            # Response should occur after target onset and before next trial
            for i, onset in enumerate(targ_on_t):
                trial = TrialData(
                    trial_number=i + 1,
                    onset_seconds=float(onset),
                )

                # Get correct answer for this trial
                if i < len(trial_lr):
                    trial.stimulus_side = "left" if trial_lr[i] == 1 else "right"

                # Get ITI
                if i < len(trial_iti):
                    trial.extras["iti"] = float(trial_iti[i])

                # Find response window
                # Response should be within 3s of target onset and before next trial
                next_onset = targ_on_t[i + 1] if i + 1 < n_trials else onset + 5.0
                window_end = min(onset + 3.0, next_onset)

                # Find responses in this window
                resp_mask = (resp_t > onset) & (resp_t <= window_end)
                resp_indices = np.where(resp_mask)[0]

                if len(resp_indices) > 0:
                    # Take first response in window
                    resp_idx = resp_indices[0]
                    trial.response_time = float(resp_t[resp_idx] - onset)

                    if resp_idx < len(resp_lr):
                        resp_side = resp_lr[resp_idx]
                        trial.response = "left" if resp_side == 1 else "right"

                        # Determine correctness
                        if trial.stimulus_side is not None:
                            trial.correct = (trial.response == trial.stimulus_side)
                else:
                    trial.response = "none"
                    trial.correct = False

                result.trials.append(trial)

            result.summary = self.compute_summary(result.trials, self.paradigm_name, block)
            result.success = True

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error parsing SAIIT_2AFC for {self.subject_id}: {e}")

        return result


class SurroundSuppParser(BaseBehavioralParser):
    """
    Parser for SurroundSupp (Surround Suppression) paradigm.

    MAT file structure:
    - TargOnT: Target onset times (64 trials per block)
    - BGcon: Background contrast conditions
    - CNTcon: Center contrast conditions
    - StimCond: Stimulus conditions
    - RespT: Single response time (last only)
    - RespLR: Single response (last only)
    """

    paradigm_name = "surrsuppblock"

    def parse(self, mat_data: Dict[str, Any], block: Optional[str] = None) -> BehavioralResult:
        result = BehavioralResult(
            subject_id=self.subject_id,
            paradigm=self.paradigm_name,
            success=False,
        )

        try:
            targ_on_t = np.atleast_1d(mat_data.get("TargOnT", []))
            bg_con = np.atleast_1d(mat_data.get("BGcon", []))
            cnt_con = np.atleast_1d(mat_data.get("CNTcon", []))
            stim_cond = np.atleast_1d(mat_data.get("StimCond", []))

            if len(targ_on_t) == 0:
                result.errors.append("No target onset times found")
                return result

            n_trials = len(targ_on_t)
            result.raw_data = {
                "n_trials": n_trials,
                "has_bg_con": len(bg_con) > 0,
                "has_cnt_con": len(cnt_con) > 0,
            }

            for i, onset in enumerate(targ_on_t):
                trial = TrialData(
                    trial_number=i + 1,
                    onset_seconds=float(onset),
                    stimulus_type="surround_suppression",
                )

                if i < len(bg_con):
                    trial.extras["bg_contrast"] = float(bg_con[i])
                if i < len(cnt_con):
                    trial.extras["center_contrast"] = float(cnt_con[i])
                if i < len(stim_cond):
                    trial.extras["stim_condition"] = int(stim_cond[i])

                # Note: RespT/RespLR only contain final response, not trial-by-trial
                # This paradigm doesn't have per-trial responses in the data
                result.trials.append(trial)

            # Summary without RT (no per-trial responses)
            result.summary = ParadigmSummary(
                subject_id=self.subject_id,
                paradigm=self.paradigm_name,
                block=block,
                n_trials=n_trials,
            )
            result.warnings.append("No per-trial responses available for this paradigm")
            result.success = True

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error parsing SurroundSupp for {self.subject_id}: {e}")

        return result


class VisLearnParser(BaseBehavioralParser):
    """
    Parser for vis_learn (Visual Learning) paradigm.

    MAT file structure:
    - TGonT: Target onset times (10 trials)
    - ITIstartT: ITI start times
    - par: Parameters struct
    """

    paradigm_name = "vislearn"

    def parse(self, mat_data: Dict[str, Any], block: Optional[str] = None) -> BehavioralResult:
        result = BehavioralResult(
            subject_id=self.subject_id,
            paradigm=self.paradigm_name,
            success=False,
        )

        try:
            tg_on_t = np.atleast_1d(mat_data.get("TGonT", []))
            iti_start_t = np.atleast_1d(mat_data.get("ITIstartT", []))

            if len(tg_on_t) == 0:
                result.errors.append("No target onset times found")
                return result

            n_trials = len(tg_on_t)
            result.raw_data = {"n_trials": n_trials}

            for i, onset in enumerate(tg_on_t):
                trial = TrialData(
                    trial_number=i + 1,
                    onset_seconds=float(onset),
                    stimulus_type="visual_learning",
                )

                # Compute duration if ITI start available
                if i < len(iti_start_t):
                    trial.duration_seconds = float(iti_start_t[i] - onset)

                result.trials.append(trial)

            # Summary without behavioral responses
            result.summary = ParadigmSummary(
                subject_id=self.subject_id,
                paradigm=self.paradigm_name,
                block=block,
                n_trials=n_trials,
            )
            result.warnings.append("Passive viewing paradigm - no behavioral responses")
            result.success = True

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error parsing vis_learn for {self.subject_id}: {e}")

        return result


class WISCProcSpeedParser(BaseBehavioralParser):
    """
    Parser for WISC_ProcSpeed (Processing Speed) paradigm.

    MAT file structure:
    - x, y: Position arrays
    - whichButton: Button press array
    - perf: Performance score (int)
    - trial_duration: Duration matrix
    """

    paradigm_name = "wiscprocspeed"

    def parse(self, mat_data: Dict[str, Any], block: Optional[str] = None) -> BehavioralResult:
        result = BehavioralResult(
            subject_id=self.subject_id,
            paradigm=self.paradigm_name,
            success=False,
        )

        try:
            x = np.atleast_1d(mat_data.get("x", []))
            y = np.atleast_1d(mat_data.get("y", []))
            which_button = np.atleast_1d(mat_data.get("whichButton", []))
            perf = mat_data.get("perf", None)
            trial_duration = mat_data.get("trial_duration", None)

            n_positions = len(x)
            n_buttons = len(which_button)

            result.raw_data = {
                "n_positions": n_positions,
                "n_buttons": n_buttons,
                "performance": int(perf) if perf is not None else None,
            }

            # Create trial entries for each button press
            for i in range(n_buttons):
                trial = TrialData(
                    trial_number=i + 1,
                    onset_seconds=0.0,  # No timing info available
                    stimulus_type="wisc_procspeed",
                )

                if i < len(which_button):
                    trial.extras["button"] = int(which_button[i])

                result.trials.append(trial)

            # Summary with performance score
            result.summary = ParadigmSummary(
                subject_id=self.subject_id,
                paradigm=self.paradigm_name,
                block=block,
                n_trials=n_buttons,
                n_responses=n_buttons,
            )
            if perf is not None:
                result.summary.extras = {"performance_score": int(perf)}

            result.warnings.append("Limited trial-level timing available")
            result.success = True

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error parsing WISC_ProcSpeed for {self.subject_id}: {e}")

        return result


class VideoParser(BaseBehavioralParser):
    """
    Parser for Video paradigm.

    MAT file structure:
    - movie_presentation_order: Order of movies shown
    - block_perm: Block permutation
    - exclude: Exclusion flags
    """

    paradigm_name = "video"

    def parse(self, mat_data: Dict[str, Any], block: Optional[str] = None) -> BehavioralResult:
        result = BehavioralResult(
            subject_id=self.subject_id,
            paradigm=self.paradigm_name,
            success=False,
        )

        try:
            movie_order = np.atleast_1d(mat_data.get("movie_presentation_order", []))
            block_perm = np.atleast_1d(mat_data.get("block_perm", []))
            exclude = np.atleast_1d(mat_data.get("exclude", []))

            result.raw_data = {
                "n_movies": len(movie_order),
                "movie_order": movie_order.tolist() if len(movie_order) > 0 else [],
                "block_perm": block_perm.tolist() if len(block_perm) > 0 else [],
            }

            # Create entries for each movie segment
            for i, movie_id in enumerate(movie_order):
                trial = TrialData(
                    trial_number=i + 1,
                    onset_seconds=0.0,  # No timing available
                    stimulus_type="video",
                )
                trial.extras["movie_id"] = int(movie_id)
                result.trials.append(trial)

            # Summary (passive viewing)
            result.summary = ParadigmSummary(
                subject_id=self.subject_id,
                paradigm=self.paradigm_name,
                block=block,
                n_trials=len(movie_order),
            )
            result.warnings.append("Passive viewing paradigm - no behavioral responses")
            result.success = True

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error parsing Video for {self.subject_id}: {e}")

        return result


# Parser registry
PARADIGM_PARSERS = {
    "SAIIT_2AFC": SAIIT2AFCParser,
    "SAIIT": SAIIT2AFCParser,
    "SurrSupp_Block": SurroundSuppParser,
    "SurroundSupp": SurroundSuppParser,
    "SurrSupp": SurroundSuppParser,
    "vis_learn": VisLearnParser,
    "VisLearn": VisLearnParser,
    "vislearn": VisLearnParser,
    "WISC_ProcSpeed": WISCProcSpeedParser,
    "WISC": WISCProcSpeedParser,
    "Video": VideoParser,
}


def get_parser(paradigm: str, subject_id: str) -> Optional[BaseBehavioralParser]:
    """Get appropriate parser for a paradigm."""
    parser_class = PARADIGM_PARSERS.get(paradigm)
    if parser_class:
        return parser_class(subject_id)
    return None


def parse_behavioral_file(
    mat_file: Path,
    subject_id: str,
    paradigm: str,
    block: Optional[str] = None,
) -> BehavioralResult:
    """
    Parse a behavioral MAT file.

    Args:
        mat_file: Path to MAT file
        subject_id: Subject ID
        paradigm: Paradigm name
        block: Optional block identifier

    Returns:
        BehavioralResult with parsed data
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        return BehavioralResult(
            subject_id=subject_id,
            paradigm=paradigm,
            success=False,
            errors=["scipy required for MAT file loading"],
        )

    parser = get_parser(paradigm, subject_id)
    if parser is None:
        return BehavioralResult(
            subject_id=subject_id,
            paradigm=paradigm,
            success=False,
            errors=[f"No parser available for paradigm: {paradigm}"],
        )

    try:
        mat_data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
        # Remove MATLAB meta keys
        mat_data = {k: v for k, v in mat_data.items() if not k.startswith("_")}
        return parser.parse(mat_data, block)
    except Exception as e:
        return BehavioralResult(
            subject_id=subject_id,
            paradigm=paradigm,
            success=False,
            errors=[f"Error loading MAT file: {e}"],
        )


def discover_behavioral_files(
    subject_dir: Path,
    subject_id: str,
) -> List[Tuple[Path, str, Optional[str]]]:
    """
    Discover behavioral MAT files for a subject.

    Returns:
        List of (file_path, paradigm, block) tuples
    """
    files = []
    beh_dir = subject_dir / "Behavioral" / "mat_format"

    if not beh_dir.exists():
        return files

    # Map file patterns to paradigms
    paradigm_patterns = [
        ("SAIIT_2AFC", "SAIIT_2AFC"),
        ("SurroundSupp", "SurrSupp_Block"),
        ("SurrSupp", "SurrSupp_Block"),
        ("vis_learn", "vis_learn"),
        ("WISC_ProcSpeed", "WISC_ProcSpeed"),
        ("Video", "Video"),
    ]

    for mat_file in sorted(beh_dir.glob("*.mat")):
        fname = mat_file.stem
        # Remove subject ID prefix
        fname_clean = fname.replace(f"{subject_id}_", "")

        for pattern, paradigm in paradigm_patterns:
            if pattern in fname_clean:
                # Extract block info (e.g., "Block1", "Block2")
                block = None
                for suffix in ["_Block1", "_Block2", "_Block3", "_Block4", "1", "2", "3", "4"]:
                    if suffix in fname_clean:
                        block = suffix.strip("_")
                        break
                files.append((mat_file, paradigm, block))
                break

    return files
