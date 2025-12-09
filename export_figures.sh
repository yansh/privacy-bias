#!/bin/bash

set -e

ENV_NAME="plots_env"
PYTHON_VERSION="3.11"

# ---------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------
EXPORT_FLAG=false
EXPORT_DIR="figures"

while [[ $# -gt 0 ]]; do
    case $1 in
        --export)
            EXPORT_FLAG=true
            if [[ -n "$2" && "$2" != --* ]]; then
                EXPORT_DIR="$2"
                shift
            fi
            ;;
    esac
    shift
done

# ---------------------------------------------------------
# Check if environment already exists
# ---------------------------------------------------------
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Activating it..."
else
    echo "Environment '$ENV_NAME' does not exist. Creating it..."
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y

    # Install necessary packages only when environment is newly created
    echo "Installing required packages..."
    conda install -c conda-forge \
        numpy \
        jupyter \
        ipywidgets \
        seaborn \
        matplotlib \
        fontconfig \
        ipykernel \
        -y

    # Force reinstall ipykernel to fix VS Code kernel issues
    conda install -n $ENV_NAME ipykernel --update-deps --force-reinstall -y
fi

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# ---------------------------------------------------------
# Register the kernel if environment is newly created
# ---------------------------------------------------------
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Registering the kernel for $ENV_NAME..."
    python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"
fi

# ---------------------------------------------------------
# If export flag: run notebook non-interactively
# ---------------------------------------------------------
if $EXPORT_FLAG ; then
    echo "Export enabled. Output folder: $EXPORT_DIR"

    # create folder if needed
    mkdir -p "$EXPORT_DIR"
    export EXPORT_DIR="$EXPORT_DIR"

    echo "Running plots.ipynb with export parameter..."
    jupyter nbconvert \
        --to notebook \
        --execute plots/plots.ipynb \
        --output output_executed.ipynb \
        --ExecutePreprocessor.kernel_name=$ENV_NAME \
        --ExecutePreprocessor.timeout=600 \
        --Application.log_level=ERROR 

    echo "Notebook executed. Figures saved in '$EXPORT_DIR/'."
    # Remove the converted file (output_executed.ipynb) after execution    
    rm -f ./plots/output_executed.ipynb
else
    echo ""
    echo "============================================================"
    echo " Environment is ready!"
    echo ""
    echo " You can now run:"
    echo "     conda activate $ENV_NAME"
    echo "     jupyter notebook"
    echo ""
    echo " Notebook will have a selectable kernel named:"
    echo "     Python ($ENV_NAME)"
    echo "============================================================"
    echo ""
    echo "Script completed successfully."
fi
