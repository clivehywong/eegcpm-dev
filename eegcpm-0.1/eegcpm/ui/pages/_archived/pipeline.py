"""Pipeline builder page."""

import streamlit as st
from pathlib import Path


def show_pipeline_page():
    """Pipeline configuration page."""
    st.title("⚙️ Pipeline Builder")

    if "project" not in st.session_state:
        st.warning("Please load a project first.")
        return

    # Pipeline steps
    st.subheader("Configure Pipeline Steps")

    # Preprocessing
    with st.expander("1. Preprocessing", expanded=True):
        show_preprocessing_config()

    # Epochs
    with st.expander("2. Epoch Extraction"):
        show_epochs_config()

    # Source reconstruction
    with st.expander("3. Source Reconstruction"):
        show_source_config()

    # Connectivity
    with st.expander("4. Connectivity Analysis"):
        show_connectivity_config()

    # Prediction
    with st.expander("5. Prediction"):
        show_prediction_config()

    # Save/Run
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save Configuration"):
            save_config()

    with col2:
        if st.button("Run Locally"):
            run_pipeline_local()

    with col3:
        if st.button("Submit to HPC"):
            submit_to_hpc()


def show_preprocessing_config():
    """Preprocessing configuration."""
    col1, col2 = st.columns(2)

    with col1:
        st.number_input("High-pass filter (Hz)", value=0.5, key="l_freq")
        st.number_input("Low-pass filter (Hz)", value=40.0, key="h_freq")
        st.number_input("Notch filter (Hz)", value=0.0, key="notch_freq",
                       help="Set to 0 to disable")

    with col2:
        st.selectbox(
            "ICA Method",
            ["infomax", "extended_infomax", "picard", "fastica"],
            key="ica_method",
        )
        st.number_input("ICA Components", value=20, key="ica_n_components")
        st.checkbox("Use ASR", key="use_asr", help="Artifact Subspace Reconstruction")


def show_epochs_config():
    """Epochs configuration."""
    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Epoch start (s)", value=-0.5, key="tmin")
        st.number_input("Epoch end (s)", value=1.0, key="tmax")

    with col2:
        st.text_input("Baseline window", value="-0.2, 0.0", key="baseline")
        st.checkbox("Use autoreject", key="use_autoreject")

    st.text_area(
        "Event IDs (JSON)",
        value='{"target": 1, "nontarget": 2}',
        key="event_id",
        help="JSON mapping of event names to codes",
    )


def show_source_config():
    """Source reconstruction configuration."""
    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Inverse Method",
            ["sLORETA", "dSPM", "eLORETA", "MNE"],
            key="source_method",
        )
        st.selectbox(
            "Parcellation",
            ["conn_networks (32 ROIs)", "aparc (68 ROIs)", "schaefer100"],
            key="parcellation",
        )

    with col2:
        st.number_input("SNR", value=3.0, key="snr")
        st.slider("Depth weighting", 0.0, 1.0, 0.8, key="depth")


def show_connectivity_config():
    """Connectivity configuration."""
    st.multiselect(
        "Connectivity Methods",
        ["correlation", "partial_correlation", "plv", "pli", "wpli", "coherence"],
        default=["correlation", "plv"],
        key="conn_methods",
    )

    st.markdown("**Frequency Bands**")
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Theta (4-8 Hz)", value=True, key="band_theta")
        st.checkbox("Alpha (8-13 Hz)", value=True, key="band_alpha")
    with col2:
        st.checkbox("Beta (13-30 Hz)", value=True, key="band_beta")
        st.checkbox("Gamma (30-45 Hz)", value=False, key="band_gamma")

    st.markdown("**Time Windows**")
    st.text_area(
        "Time Windows (JSON)",
        value='[{"name": "prestim", "tmin": -0.5, "tmax": 0.0}, {"name": "poststim", "tmin": 0.0, "tmax": 0.5}]',
        key="time_windows",
    )


def show_prediction_config():
    """Prediction configuration."""
    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Prediction Type",
            ["between_subject", "within_subject", "between_group"],
            key="prediction_type",
        )
        st.text_input("Target Variable", value="behavioral_score", key="target_variable")

    with col2:
        st.selectbox(
            "CV Strategy",
            ["kfold", "leave_one_out", "stratified_kfold"],
            key="cv_strategy",
        )
        st.number_input("CV Folds", value=5, key="n_folds")

    st.multiselect(
        "Models to Compare",
        ["ridge", "lasso", "svr", "svm", "random_forest", "cnn", "lstm"],
        default=["ridge", "svr"],
        key="models",
    )


def save_config():
    """Save current configuration."""
    st.info("Configuration saved!")
    # TODO: Implement config saving


def run_pipeline_local():
    """Run pipeline locally."""
    st.info("Starting local pipeline execution...")
    # TODO: Implement local execution


def submit_to_hpc():
    """Submit to HPC."""
    st.info("Submitting jobs to HPC...")
    # TODO: Implement HPC submission
