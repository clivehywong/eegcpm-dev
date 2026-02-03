"""Analysis page."""

import streamlit as st


def show_analysis_page():
    """Analysis and exploration page."""
    st.title("🔬 Analysis")

    if "project" not in st.session_state:
        st.warning("Please load a project first.")
        return

    tab1, tab2, tab3 = st.tabs(["ERP", "Connectivity", "Prediction"])

    with tab1:
        show_erp_analysis()

    with tab2:
        show_connectivity_analysis()

    with tab3:
        show_prediction_analysis()


def show_erp_analysis():
    """ERP visualization and analysis."""
    st.subheader("ERP Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox("Subject", ["sub-001", "sub-002"], key="erp_subject")
        st.selectbox("Condition", ["target", "nontarget"], key="erp_condition")

    with col2:
        st.multiselect("Channels", ["Fz", "Cz", "Pz"], key="erp_channels")
        st.checkbox("Show grand average", key="erp_grand_avg")

    # Placeholder for ERP plot
    st.info("ERP plot will be displayed here")

    # ERP statistics
    st.subheader("ERP Statistics")
    st.markdown("""
    | Component | Latency (ms) | Amplitude (µV) |
    |-----------|--------------|----------------|
    | P1        | 100          | 2.5            |
    | N1        | 170          | -3.2           |
    | P3        | 350          | 8.1            |
    """)


def show_connectivity_analysis():
    """Connectivity visualization."""
    st.subheader("Connectivity Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.selectbox("Subject", ["sub-001", "sub-002", "Grand Average"], key="conn_subject")

    with col2:
        st.selectbox("Method", ["correlation", "plv", "wpli"], key="conn_method")

    with col3:
        st.selectbox("Frequency Band", ["theta", "alpha", "beta"], key="conn_band")

    st.selectbox("Time Window", ["prestim", "poststim"], key="conn_window")

    # Placeholder for connectivity matrix
    st.info("Connectivity matrix heatmap will be displayed here")

    # Network summary
    st.subheader("Network Summary")
    st.markdown("""
    | Network | Within-Network | Global Connectivity |
    |---------|----------------|---------------------|
    | DMN     | 0.45           | 0.32                |
    | Salience| 0.52           | 0.38                |
    | FPN     | 0.48           | 0.35                |
    """)


def show_prediction_analysis():
    """Prediction results and model comparison."""
    st.subheader("Prediction Analysis")

    # Model comparison
    st.markdown("### Model Comparison")

    st.markdown("""
    | Model | R² | MSE | Pearson r | p-value |
    |-------|-----|-----|-----------|---------|
    | Ridge | 0.35 | 12.5 | 0.59 | 0.001 |
    | SVR   | 0.32 | 13.2 | 0.57 | 0.002 |
    | CNN   | 0.38 | 11.8 | 0.62 | <0.001 |
    """)

    # Scatter plot placeholder
    st.info("Predicted vs. Actual scatter plot will be displayed here")

    # Feature importance
    st.subheader("Feature Importance")
    st.info("Top contributing edges/features will be displayed here")
