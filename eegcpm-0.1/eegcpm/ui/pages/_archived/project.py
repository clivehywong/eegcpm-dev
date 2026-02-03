"""Project setup page."""

import streamlit as st
from pathlib import Path


def show_project_page():
    """Project setup and management."""
    st.title("📁 Project Setup")

    tab1, tab2, tab3 = st.tabs(["Create New", "Load Existing", "Project Info"])

    with tab1:
        show_create_project()

    with tab2:
        show_load_project()

    with tab3:
        show_project_info()


def show_create_project():
    """Create new project form."""
    st.subheader("Create New Project")

    with st.form("create_project"):
        name = st.text_input("Project Name", placeholder="HBN_CPM_Study")
        description = st.text_area("Description", placeholder="Describe your study...")
        root_path = st.text_input("Data Directory", placeholder="/path/to/data")
        sampling_rate = st.number_input("Sampling Rate (Hz)", value=500.0, min_value=1.0)

        submitted = st.form_submit_button("Create Project")

        if submitted:
            if not name or not root_path:
                st.error("Please provide project name and data directory")
            else:
                try:
                    from eegcpm.core.project import create_project, save_project

                    project = create_project(
                        name=name,
                        root_path=Path(root_path),
                        description=description,
                        sampling_rate_hz=sampling_rate,
                    )
                    save_project(project)
                    st.session_state["project"] = project
                    st.success(f"Project '{name}' created successfully!")

                except Exception as e:
                    st.error(f"Error creating project: {e}")


def show_load_project():
    """Load existing project."""
    st.subheader("Load Existing Project")

    project_file = st.text_input(
        "Project File",
        placeholder="/path/to/project/project.json"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Project"):
            if project_file:
                try:
                    from eegcpm.core.project import load_project

                    project = load_project(Path(project_file))
                    st.session_state["project"] = project
                    st.success(f"Loaded project: {project.name}")
                except Exception as e:
                    st.error(f"Error loading project: {e}")
            else:
                st.warning("Please enter project file path")

    with col2:
        if st.button("Scan BIDS Directory"):
            if project_file:
                try:
                    from eegcpm.core.project import scan_bids_directory

                    project = scan_bids_directory(Path(project_file))
                    st.session_state["project"] = project
                    st.success(f"Found {len(project.subjects)} subjects")
                except Exception as e:
                    st.error(f"Error scanning directory: {e}")


def show_project_info():
    """Display current project info."""
    st.subheader("Current Project")

    if "project" not in st.session_state:
        st.info("No project loaded. Create or load a project first.")
        return

    project = st.session_state["project"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Name:** {project.name}")
        st.markdown(f"**Path:** {project.root_path}")
        st.markdown(f"**Subjects:** {len(project.subjects)}")

    with col2:
        st.markdown(f"**Sampling Rate:** {project.sampling_rate_hz} Hz")
        st.markdown(f"**Description:** {project.description}")

    # Subject list
    if project.subjects:
        st.subheader("Subjects")

        subject_data = []
        for subject in project.subjects:
            n_sessions = len(subject.sessions)
            n_runs = sum(len(s.runs) for s in subject.sessions)
            subject_data.append({
                "Subject ID": subject.id,
                "Sessions": n_sessions,
                "Runs": n_runs,
            })

        st.dataframe(subject_data, width="stretch")
