"""HPC job management page."""

import streamlit as st


def show_hpc_page():
    """HPC job management."""
    st.title("🖥️ HPC Jobs")

    tab1, tab2, tab3 = st.tabs(["Submit Jobs", "Monitor", "Configuration"])

    with tab1:
        show_submit_jobs()

    with tab2:
        show_monitor()

    with tab3:
        show_hpc_config()


def show_submit_jobs():
    """Submit jobs to HPC."""
    st.subheader("Submit Jobs")

    if "project" not in st.session_state:
        st.warning("Please load a project first.")
        return

    # Subject selection
    st.markdown("### Select Subjects")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("All subjects", value=True, key="all_subjects")

    with col2:
        if not st.session_state.get("all_subjects", True):
            st.multiselect("Select subjects", ["sub-001", "sub-002", "sub-003"], key="selected_subjects")

    # Job configuration
    st.markdown("### Job Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Job Name Prefix", value="eegcpm", key="job_prefix")
        st.selectbox("Partition", ["normal", "gpu", "long"], key="partition")

    with col2:
        st.text_input("Time Limit", value="04:00:00", key="time_limit")
        st.text_input("Memory", value="32G", key="memory")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("CPUs per task", value=8, key="cpus")

    with col2:
        st.number_input("GPUs (0 for none)", value=0, key="gpus")

    # Submit
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Scripts (Dry Run)"):
            st.info("Scripts generated in ./slurm_scripts/")
            st.code("""#!/bin/bash
#SBATCH --job-name=eegcpm_sub-001
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load python/3.11
source activate eegcpm

python -m eegcpm.cli run --subject sub-001
""")

    with col2:
        if st.button("Submit All Jobs"):
            st.success("Submitted 50 jobs to SLURM")


def show_monitor():
    """Monitor job status."""
    st.subheader("Job Monitor")

    # Refresh button
    if st.button("🔄 Refresh"):
        st.info("Refreshing job status...")

    # Job status table
    st.markdown("### Active Jobs")

    job_data = [
        {"Job ID": "12345", "Name": "eegcpm_sub-001", "Status": "RUNNING", "Time": "01:23:45", "Node": "node01"},
        {"Job ID": "12346", "Name": "eegcpm_sub-002", "Status": "RUNNING", "Time": "01:20:12", "Node": "node02"},
        {"Job ID": "12347", "Name": "eegcpm_sub-003", "Status": "PENDING", "Time": "-", "Node": "-"},
        {"Job ID": "12348", "Name": "eegcpm_sub-004", "Status": "PENDING", "Time": "-", "Node": "-"},
    ]

    st.dataframe(job_data, width="stretch")

    # Completed jobs
    st.markdown("### Completed Jobs")

    completed_data = [
        {"Job ID": "12340", "Name": "eegcpm_sub-005", "Status": "COMPLETED", "Runtime": "02:15:30", "Exit": "0"},
        {"Job ID": "12341", "Name": "eegcpm_sub-006", "Status": "COMPLETED", "Runtime": "02:18:45", "Exit": "0"},
        {"Job ID": "12342", "Name": "eegcpm_sub-007", "Status": "FAILED", "Runtime": "00:45:12", "Exit": "1"},
    ]

    st.dataframe(completed_data, width="stretch")

    # Actions
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Cancel All Pending"):
            st.warning("Cancelled 2 pending jobs")

    with col2:
        if st.button("Resubmit Failed"):
            st.info("Resubmitted 1 failed job")


def show_hpc_config():
    """HPC configuration."""
    st.subheader("HPC Configuration")

    st.markdown("### SLURM Settings")

    st.text_input("Default Partition", value="normal", key="default_partition")
    st.text_input("Default Time", value="04:00:00", key="default_time")
    st.text_input("Default Memory", value="32G", key="default_memory")
    st.number_input("Default CPUs", value=8, key="default_cpus")

    st.markdown("### Modules to Load")

    st.text_area(
        "Module List",
        value="python/3.11\ncuda/11.8\nmne/1.6",
        key="modules",
        help="One module per line",
    )

    st.markdown("### Environment")

    st.text_input("Conda Environment", value="eegcpm", key="conda_env")
    st.text_input("Log Directory", value="./slurm_logs/", key="log_dir")

    if st.button("Save Configuration"):
        st.success("Configuration saved!")
