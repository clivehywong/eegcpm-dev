"""Trial filtering functions for epoch QC and selection.

Implements filtering based on:
- Reaction time (RT) bounds
- Response accuracy
- Missing responses
- Sequential position (practice/fatigue trials)
"""

from typing import Optional

import numpy as np
import pandas as pd


def filter_by_rt(
    events_df: pd.DataFrame,
    rt_column: str = "response_time",
    rt_min: Optional[float] = None,
    rt_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter trials by reaction time bounds.

    Args:
        events_df: Events dataframe with RT column
        rt_column: Name of RT column (default: "response_time")
        rt_min: Minimum valid RT in seconds (reject faster)
        rt_max: Maximum valid RT in seconds (reject slower)

    Returns:
        Filtered dataframe
    """
    df = events_df.copy()

    if rt_column not in df.columns:
        raise ValueError(f"RT column '{rt_column}' not found in events")

    mask = np.ones(len(df), dtype=bool)

    if rt_min is not None:
        mask &= df[rt_column] >= rt_min

    if rt_max is not None:
        mask &= df[rt_column] <= rt_max

    return df[mask]


def filter_by_accuracy(
    events_df: pd.DataFrame,
    accuracy_column: str = "accuracy",
    keep: str = "correct",
) -> pd.DataFrame:
    """
    Filter trials by response accuracy.

    Args:
        events_df: Events dataframe with accuracy column
        accuracy_column: Name of accuracy column (default: "accuracy")
        keep: Which trials to keep - "correct" or "incorrect"

    Returns:
        Filtered dataframe
    """
    df = events_df.copy()

    if accuracy_column not in df.columns:
        raise ValueError(f"Accuracy column '{accuracy_column}' not found in events")

    if keep == "correct":
        mask = df[accuracy_column] == 1
    elif keep == "incorrect":
        mask = df[accuracy_column] == 0
    else:
        raise ValueError(f"Invalid keep value: {keep}. Use 'correct' or 'incorrect'")

    return df[mask]


def drop_missing_responses(
    events_df: pd.DataFrame,
    rt_column: str = "response_time",
) -> pd.DataFrame:
    """
    Drop trials with missing responses.

    Args:
        events_df: Events dataframe
        rt_column: Name of RT column (trials with NaN/0 RT are missing)

    Returns:
        Filtered dataframe
    """
    df = events_df.copy()

    if rt_column not in df.columns:
        return df  # No RT column, can't filter

    # Drop trials with NaN or 0 RT
    mask = df[rt_column].notna() & (df[rt_column] > 0)

    return df[mask]


def drop_sequential_positions(
    events_df: pd.DataFrame,
    drop_first_n: int = 0,
    drop_last_n: int = 0,
) -> pd.DataFrame:
    """
    Drop trials from beginning or end of run.

    Useful for removing practice trials (beginning) or fatigue trials (end).

    Args:
        events_df: Events dataframe
        drop_first_n: Number of trials to drop from beginning
        drop_last_n: Number of trials to drop from end

    Returns:
        Filtered dataframe
    """
    df = events_df.copy()

    n_trials = len(df)
    if drop_first_n + drop_last_n >= n_trials:
        raise ValueError(
            f"Cannot drop {drop_first_n + drop_last_n} trials from {n_trials} total"
        )

    # Slice to keep middle trials
    start_idx = drop_first_n
    end_idx = n_trials - drop_last_n if drop_last_n > 0 else n_trials

    return df.iloc[start_idx:end_idx].reset_index(drop=True)


def apply_trial_filters(
    events_df: pd.DataFrame,
    filters_config: "TrialFilterConfig",
) -> pd.DataFrame:
    """
    Apply all trial filters from configuration.

    Args:
        events_df: Events dataframe
        filters_config: TrialFilterConfig instance

    Returns:
        Filtered dataframe
    """
    df = events_df.copy()

    # 1. Sequential position first (practice/fatigue trials)
    if filters_config.drop_first_n > 0 or filters_config.drop_last_n > 0:
        df = drop_sequential_positions(
            df,
            drop_first_n=filters_config.drop_first_n,
            drop_last_n=filters_config.drop_last_n,
        )

    # 2. Missing responses
    if filters_config.drop_missing_responses:
        df = drop_missing_responses(df, rt_column=filters_config.rt_field)

    # 3. RT bounds
    if filters_config.rt_min is not None or filters_config.rt_max is not None:
        df = filter_by_rt(
            df,
            rt_column=filters_config.rt_field,
            rt_min=filters_config.rt_min,
            rt_max=filters_config.rt_max,
        )

    # 4. Accuracy filter
    if filters_config.accuracy_filter is not None:
        df = filter_by_accuracy(
            df,
            accuracy_column=filters_config.accuracy_field,
            keep=filters_config.accuracy_filter,
        )

    return df
