#!/bin/bash

# Setup script for LaBraM LoRA project
echo "Setting up LaBraM LoRA environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "labram-lora"; then
    echo "Environment 'labram-lora' already exists. Updating..."
    conda env update -f environment.yml
else
    echo "Creating new environment 'labram-lora'..."
    conda env create -f environment.yml
fi

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate labram-lora"
echo ""
echo "To install the project in development mode, run:"
echo "pip install -e ."
echo ""
echo "To run the model, execute:"
echo "python lora.py"
echo ""
echo "Alternative installation methods:"
echo "1. Development install: pip install -e ."
echo "2. Regular install: pip install ."
echo "3. With dev dependencies: pip install -e .[dev]"
echo "4. With jupyter support: pip install -e .[jupyter]"