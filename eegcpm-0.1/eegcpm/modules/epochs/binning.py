"""Epoch binning strategies for trial sorting.

Implements binning by:
- Stimulus × Accuracy (2×2 design)
- Reaction time quartiles/percentiles
- Sequential position (early/middle/late trials)
- Custom metadata fields
- Stimulus × RT interaction
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def bin_stimulus_x_accuracy(
    events_df: pd.DataFrame,
    task_config: "TaskConfig",
    binning_config: "BinningConfig",
) -> Dict[str, pd.DataFrame]:
    """
    Bin trials by stimulus condition × response accuracy.

    Creates bins like: target_left_correct, target_left_incorrect, etc.

    Args:
        events_df: Events dataframe
        task_config: TaskConfig with conditions defined
        binning_config: BinningConfig with conditions and responses

    Returns:
        Dict mapping bin names to dataframes
    """
    bins = {}

    # Get condition and response names
    conditions = binning_config.conditions
    responses = binning_config.responses

    if not conditions or not responses:
        raise ValueError("stimulus_x_accuracy requires conditions and responses")

    # Assume events_df has "trial_type" and "accuracy" columns
    for condition in conditions:
        for response in responses:
            # Filter trials matching condition and response
            # This assumes trial_type column has condition names
            # and accuracy column has 1=correct, 0=incorrect
            bin_name = f"{condition}_{response}"

            if response == "correct":
                mask = (events_df["trial_type"] == condition) & (events_df["accuracy"] == 1)
            elif response == "incorrect":
                mask = (events_df["trial_type"] == condition) & (events_df["accuracy"] == 0)
            elif response == "missing":
                mask = (events_df["trial_type"] == condition) & (
                    events_df["response_time"].isna() | (events_df["response_time"] == 0)
                )
            else:
                # Custom response mapping - look up in task_config
                continue

            bins[bin_name] = events_df[mask].copy()

    return bins


def bin_rt_quartiles(
    events_df: pd.DataFrame,
    binning_config: "BinningConfig",
) -> Dict[str, pd.DataFrame]:
    """
    Bin trials by reaction time quartiles/percentiles.

    Args:
        events_df: Events dataframe
        binning_config: BinningConfig with rt_field and n_bins

    Returns:
        Dict mapping bin names to dataframes
    """
    rt_column = binning_config.rt_field
    n_bins = binning_config.n_bins
    labels = binning_config.labels

    if rt_column not in events_df.columns:
        raise ValueError(f"RT column '{rt_column}' not found")

    # Remove missing RTs
    df = events_df[events_df[rt_column].notna() & (events_df[rt_column] > 0)].copy()

    # Compute quartiles
    rt_bins = pd.qcut(
        df[rt_column],
        q=n_bins,
        labels=labels or [f"Q{i+1}" for i in range(n_bins)],
        duplicates="drop",
    )

    # Create bins
    bins = {}
    for label in rt_bins.unique():
        if pd.notna(label):
            bins[str(label)] = df[rt_bins == label].copy()

    return bins


def bin_sequential_position(
    events_df: pd.DataFrame,
    binning_config: "BinningConfig",
) -> Dict[str, pd.DataFrame]:
    """
    Bin trials by sequential position in run.

    Divides trials into equal-sized blocks (e.g., early, middle, late).

    Args:
        events_df: Events dataframe
        binning_config: BinningConfig with bins and position_labels

    Returns:
        Dict mapping bin names to dataframes
    """
    n_bins = binning_config.bins
    labels = binning_config.position_labels or [
        f"position_{i+1}" for i in range(n_bins)
    ]

    n_trials = len(events_df)
    trials_per_bin = n_trials // n_bins

    bins = {}
    for i, label in enumerate(labels):
        start_idx = i * trials_per_bin
        end_idx = (i + 1) * trials_per_bin if i < n_bins - 1 else n_trials

        bins[label] = events_df.iloc[start_idx:end_idx].copy()

    return bins


def bin_by_metadata_field(
    events_df: pd.DataFrame,
    binning_config: "BinningConfig",
) -> Dict[str, pd.DataFrame]:
    """
    Bin trials by arbitrary metadata field.

    Args:
        events_df: Events dataframe
        binning_config: BinningConfig with field and values

    Returns:
        Dict mapping bin names to dataframes
    """
    field = binning_config.field
    values = binning_config.values

    if not field or not values:
        raise ValueError("metadata_field binning requires field and values")

    if field not in events_df.columns:
        raise ValueError(f"Field '{field}' not found in events")

    bins = {}
    for value in values:
        bin_name = f"{field}_{value}"
        bins[bin_name] = events_df[events_df[field] == value].copy()

    return bins


def bin_stimulus_x_rt(
    events_df: pd.DataFrame,
    task_config: "TaskConfig",
    binning_config: "BinningConfig",
) -> Dict[str, pd.DataFrame]:
    """
    Bin trials by stimulus condition × RT quartiles.

    Creates bins like: target_left_fast, target_left_slow, etc.

    Args:
        events_df: Events dataframe
        task_config: TaskConfig with conditions
        binning_config: BinningConfig with conditions, rt_field, n_bins

    Returns:
        Dict mapping bin names to dataframes
    """
    conditions = binning_config.conditions
    rt_column = binning_config.rt_field
    n_bins = binning_config.n_bins
    labels = binning_config.labels or [f"rt_{i+1}" for i in range(n_bins)]

    if not conditions:
        raise ValueError("stimulus_x_rt requires conditions")

    bins = {}

    for condition in conditions:
        # Get trials for this condition
        condition_df = events_df[events_df["trial_type"] == condition].copy()

        if len(condition_df) == 0:
            continue

        # Bin by RT within condition
        rt_bins = pd.qcut(
            condition_df[rt_column],
            q=n_bins,
            labels=labels,
            duplicates="drop",
        )

        # Create bins
        for label in rt_bins.unique():
            if pd.notna(label):
                bin_name = f"{condition}_{label}"
                bins[bin_name] = condition_df[rt_bins == label].copy()

    return bins


def apply_binning(
    events_df: pd.DataFrame,
    binning_configs: List["BinningConfig"],
    task_config: "TaskConfig",
) -> Dict[str, pd.DataFrame]:
    """
    Apply all binning strategies from configuration.

    Args:
        events_df: Events dataframe
        binning_configs: List of BinningConfig instances
        task_config: TaskConfig instance

    Returns:
        Dict mapping bin names to dataframes
    """
    if not binning_configs:
        return {"all": events_df.copy()}

    all_bins = {}

    for binning_config in binning_configs:
        if binning_config.type == "stimulus_x_accuracy":
            bins = bin_stimulus_x_accuracy(events_df, task_config, binning_config)
        elif binning_config.type == "rt_quartiles":
            bins = bin_rt_quartiles(events_df, binning_config)
        elif binning_config.type == "sequential_position":
            bins = bin_sequential_position(events_df, binning_config)
        elif binning_config.type == "metadata_field":
            bins = bin_by_metadata_field(events_df, binning_config)
        elif binning_config.type == "stimulus_x_rt":
            bins = bin_stimulus_x_rt(events_df, task_config, binning_config)
        else:
            raise ValueError(f"Unknown binning type: {binning_config.type}")

        all_bins.update(bins)

    return all_bins
