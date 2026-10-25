@echo off
cd /d "%~dp0"
if not exist venv (
    echo Setting up for the first time, please wait...
    python -m venv venv
)
call venv\Scripts\activate
pip install -q -r requirements.txt
streamlit run app.py
pause