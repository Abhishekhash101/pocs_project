@echo off
echo Checking and installing required libraries from requirements.txt...
echo.
pip install -r requirements.txt
echo.
echo ==========================================================
echo All libraries are installed. Starting the application...
echo ==========================================================
echo.
streamlit run main.py
pause