"""Streamlit Cloud entry point for the complete learning platform."""
from pathlib import Path
import runpy
import sys


APP_DIR = Path(__file__).resolve().parent / "AI POWERED PERSONALISED LEARNING PATH GENERATOR"
sys.path.insert(0, str(APP_DIR))
runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")