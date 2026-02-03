#!/usr/bin/env python3
"""Launch the run selection Streamlit UI."""

import subprocess
import sys
from pathlib import Path

# Get the UI page path
ui_page = Path(__file__).parent.parent / "eegcpm" / "ui" / "pages" / "run_selection.py"

if not ui_page.exists():
    print(f"Error: UI page not found at {ui_page}")
    sys.exit(1)

print("=" * 80)
print("Launching Run Selection UI")
print("=" * 80)
print(f"\nUI Page: {ui_page}")
print("\nStarting Streamlit server...")
print("Press Ctrl+C to stop\n")
print("=" * 80)

# Launch streamlit
subprocess.run([
    "streamlit", "run",
    str(ui_page),
    "--server.port", "8503",
    "--server.headless", "false"
])
