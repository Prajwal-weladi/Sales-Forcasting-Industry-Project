@echo off
echo Starting Advanced Sales Analytics Dashboard...
echo.
echo Installing required packages...
cd backend
pip install -r requirements_advanced.txt
echo.
echo Starting Streamlit dashboard...
streamlit run advanced_analytics_dashboard.py
pause 