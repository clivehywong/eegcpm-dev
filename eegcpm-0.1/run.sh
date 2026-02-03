find /Users/clive/eegcpm/eegcpm-0.1 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find /Users/clive/eegcpm/eegcpm-0.1 -name "*.pyc" -delete 2>/dev/null

streamlit run eegcpm/ui/app.py --server.port 8502
