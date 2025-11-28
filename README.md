# Delivery Route Planner
Simple Flask app that loads daily store demand, runs an OR-Tools vehicle routing solver, and shows routes on a web map.

## What you need
- Python 3.10+ recommended
- Data files in `data/` (demand CSVs per weekday, `store_metadata.csv`, and distance/time matrices)
- Dependencies from `requirements.txt`

## Quick start
1) Create and activate a virtualenv (optional but recommended).
2) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3) Run the app:
   ```bash
   python app.py
   ```
4) Open `http://localhost:1234` in your browser. Pick a weekday, set vehicle settings, and generate routes. You can clear or update routes via the UI controls.