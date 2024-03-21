#!/bin/bash

# Check if python3.9.10 is installed
if ! command -v python3.9.10 &> /dev/null
then
    echo "python3.9.10 could not be found"
    exit
fi

# Create a virtual environment
python3.9.10 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Check if requirements.txt exists in the current directory
if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found in the current directory."
    exit 1
fi

# Install packages from requirements.txt
pip install -r requirements.txt