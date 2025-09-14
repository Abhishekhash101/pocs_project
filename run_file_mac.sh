#!/bin/bash
echo "Checking and installing required libraries..."
pip install -r requirements.txt

echo "-----------------------------------------"
echo "Starting the application..."
echo "-----------------------------------------"
streamlit run main.py