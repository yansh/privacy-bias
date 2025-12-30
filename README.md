# Privacy Bias

**Paper title**: Privacy Bias in Language Models: A Contextual Integrity-based Auditing Metric

**Requested Badge(s):**
  - [x] Available
  - [x] Functional
  - [x] Reproduced

## Table of Contents
1. [Description](#description)
2. [Important Files](#important-files)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Requirements](#software-requirements)
5. [Setup Instructions](#setup-instructions)
   - [Step 1: Build the Docker Image](#step-1-build-the-docker-image)
   - [Step 2: Run the Docker Container](#step-2-run-the-docker-container)
   - [Step 3: Run the Experiment](#step-3-run-the-experiment)
5. [Prompting OpenAI](#prompting-openai)
6. [Exporting figures](#exporting-figures)
## Description

```bibtex
@Article{PoPETS:PrivacyBias26,
  author    =   "Yan Shvartzshnaider and Vasisht Duddu",
  title     =   "{Privacy Bias in Language Models: A Contextual Integrity-based Auditing Metric}",
  year      =   2026,
  volume    =   2026,
  journal   =   "{Proceedings on Privacy Enhancing Technologies}",
}
```

## Summary of results and takaways

For summary of the results and main takaways, please visit the [website](https://yansh.github.io/privacy-bias/website/)

## Important Files

*  `data` - datasets of all the generated vignettes
*  `data/openAI` -  openAI prompt batches
* `plots/plots.ipynb` — Jupyter notebook to generate paper figures.
* `plots/dataframes` - Results dataframes.
* `raw_results.7z` — Archive containing raw CSV results.
  * To unpack: `7z x raw_results.7z`
* `run_experiments.sh`  - bash script to run prompts agains the LLMs
* `export_figures.sh`  - bash script to export all the paper figures

### Notes:

- **Raw Data Storage**:  
  The repository contains raw CSV which requires around 3.1 GB of storage to unarchive.
  This includes results_PETS (1.6GB), results_temp/ (201MB) and results_paraphrasing/ (1.4GB).
  For loading models from disk, each models considered in our work take up atmost 10GB each.

- **Plot Reproducibility**:  
  Plots can be reproduced without a GPU, using dataframes containing processed raw data.

- **GPU & API Keys**:  
  Running new models requires a GPU and Hugging Face API keys.


## Hardware Requirements

- NVIDIA GPU (tested on RTX 4090)
- VRAM: 24 GB
- Driver: 550.127.05
- CUDA: 12.4

## Software Requirements

- Docker Engine 28.3.3+
- git 2.39.5+
- NVIDIA Drivers (tested with 550.127.05)
- NVIDIA Container Toolkit

## Instructions

### Using a locally built Docker image

1. Build the Docker image:

```bash
docker build -t privacy_bias:latest .
```

2. Run:

```bash
docker run --gpus all --runtime=nvidia -it \
    -v $(pwd):/home/ubuntu/privacy-bias \
    privacy_bias:latest /bin/bash -c "cd /home/ubuntu/privacy-bias && exec bash"
```

### Using VS Code

1. Install the Dev Containers extension in VS Code.
2. Open the repository in VS Code.
3. Press F1 → Dev Containers: Open Folder in Container… → select project.


### Run the Experiment

In the docker run: 

To run the script with the default configurations (i.e., using all models, temperature 0, and dataset "iot"):

```bash
bash run_experiments.sh
```


#### Running with selected arguments:

You can specify specific models, temperatures, datasets, and paraphrasing methods using the command-line arguments:

- **`--models=<model1> <model2> ...`**: Space-separated list of models to run experiments with (e.g., `--models="allenai/tulu-2-7b meta-llama/Meta-Llama-3.1-8B-Instruct"`).  
- **`--temps=<temp1> <temp2> ...`**: Space-separated list of temperatures to use (e.g., `--temps="0 0.5"`). 
- **`--datasets=<dataset1> <dataset2> ...`**: Space-separated list of datasets to use (e.g., `--datasets="iot confaide"`). 
- **`--paraphrasing=<method1> <method2> ...`**: Space-separated list of paraphrasing methods (e.g., `--paraphrasing="gpt gemini"`).

*Example usage:*

`bash run_experiments.sh --models="allenai/tulu-2-7b" --temps="0.5" --datasets="iot confaide"`




### Prompting OpenAI

The `data` folder contains batches of data specifically prepared for use with the `gpt-4o-mini` model.  

Upload the batches directly to the [OpenAI platform](https://platform.openai.com/).


## Exporting figures

This script requires **Python >= 3.10**.

Use this command to export the paper figures into the `figs` folder
```bash
bash export_figures.sh --export figs
```


