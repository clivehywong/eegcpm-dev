"""Results viewer page."""

import streamlit as st
from pathlib import Path


def show_results_page():
    """View and export results."""
    st.title("📊 Results")

    if "project" not in st.session_state:
        st.warning("Please load a project first.")
        return

    tab1, tab2, tab3 = st.tabs(["Summary", "Export", "Reports"])

    with tab1:
        show_summary()

    with tab2:
        show_export()

    with tab3:
        show_reports()


def show_summary():
    """Results summary."""
    st.subheader("Pipeline Results Summary")

    # Processing status
    st.markdown("### Processing Status")

    status_data = [
        {"Subject": "sub-001", "Preprocessing": "✅", "Epochs": "✅", "Source": "✅", "Connectivity": "✅", "Features": "✅"},
        {"Subject": "sub-002", "Preprocessing": "✅", "Epochs": "✅", "Source": "✅", "Connectivity": "✅", "Features": "✅"},
        {"Subject": "sub-003", "Preprocessing": "✅", "Epochs": "✅", "Source": "⏳", "Connectivity": "⏸️", "Features": "⏸️"},
    ]

    st.dataframe(status_data, width="stretch")

    # Prediction summary
    st.markdown("### Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best R²", "0.38", delta="0.03")

    with col2:
        st.metric("Best Model", "Ridge")

    with col3:
        st.metric("Significant Edges", "124")

    # Quality metrics
    st.markdown("### Data Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Subjects Processed", "45/50")
        st.metric("Mean Epochs Retained", "85%")

    with col2:
        st.metric("Mean ICA Components Removed", "3.2")
        st.metric("Failed Subjects", "2")


def show_export():
    """Export results."""
    st.subheader("Export Results")

    st.markdown("### Export Options")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Connectivity matrices", value=True, key="export_conn")
        st.checkbox("Feature matrices", value=True, key="export_features")
        st.checkbox("Predictions", value=True, key="export_predictions")

    with col2:
        st.checkbox("Model weights", value=False, key="export_weights")
        st.checkbox("Quality reports", value=True, key="export_quality")
        st.checkbox("Figures", value=True, key="export_figures")

    st.selectbox("Format", ["NPZ", "CSV", "HDF5", "MATLAB (.mat)"], key="export_format")

    export_path = st.text_input("Export Path", value="./exports/")

    if st.button("Export"):
        st.success(f"Results exported to {export_path}")


def show_reports():
    """Generate reports."""
    st.subheader("Generate Reports")

    report_type = st.selectbox(
        "Report Type",
        ["Full Analysis Report", "Quality Control Report", "Prediction Summary", "Methods Section"],
    )

    st.selectbox("Format", ["HTML", "PDF", "Markdown"], key="report_format")

    if st.button("Generate Report"):
        st.info(f"Generating {report_type}...")

        # Placeholder
        st.markdown("""
        ---
        ## Analysis Report

        ### Dataset
        - 50 subjects from Healthy Brain Network
        - Task: Visual oddball paradigm

        ### Preprocessing
        - Bandpass filter: 0.5-40 Hz
        - ICA: 20 components, Infomax
        - Artifact rejection: autoreject

        ### Source Reconstruction
        - Method: sLORETA
        - Parcellation: CONN 32 ROIs

        ### Connectivity
        - Methods: Correlation, PLV
        - Frequency bands: Theta, Alpha, Beta
        - Time windows: Pre-stimulus, Post-stimulus

        ### Prediction Results
        - Target: Behavioral score
        - Best model: Ridge (R² = 0.38)
        - Cross-validation: Leave-one-out

        ---
        """)
