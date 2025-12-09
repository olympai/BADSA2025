#!/bin/bash

# Script to set up virtual environment and install dependencies

echo "Setting up virtual environment..."

# Create virtual environment
python3 -m venv venv

# Check if virtual environment was created successfully
if [ ! -d "venv" ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created successfully!"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Setup complete! Virtual environment is ready."
    echo "To activate the virtual environment in the future, run:"
    echo "  source venv/bin/activate"
    echo "To deactivate, run:"
    echo "  deactivate"
    echo ""
    
else
    echo "Error: Failed to install dependencies"
    exit 1
fi
