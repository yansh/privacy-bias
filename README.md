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

## Important Files

*  `data` - Datasets of all the generated vignettes
*  `data/openAI` - includes openAI prompt batches
* `plots/plots.ipynb` — Jupyter notebook to generate paper figures.
* `plots/dataframes` - Processes results dataframes
* `raw_results.7z` — Archive of raw CSV results.
  * To unpack: `7z x raw_results.7z`
* `run_experiments.sh`  - bash script to run prompts agains the LLMs


## Hardware Requirements

- NVIDIA GPU (tested on RTX 4090)
- VRAM: 24 GB
- Driver: 550.127.05
- CUDA: 12.4

Note: The repositary has the raw data which requires large storage. 
The plots can be reproduced without requiring access to GPU. 
GPU and API keys to HuggingFace are required for running new models.

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
docker run -p 8888:8888 \
    -v ./plots/objects:/home/auditor/./plots/objects \
    -v ./plots:/home/auditor/plots \
    privacy_bias:latest
```

```bash
docker run --gpus all --runtime=nvidia -it \
    -v $(pwd):/home/ubuntu/privacy-bias \
    privacy_bias:latest /bin/bash -c "cd /home/ubuntu/privacy-bias && exec bash"
```


3. Open your browser and go to: http://localhost:8888

### Using VS Code

1. Install the Dev Containers extension in VS Code.
2. Open the repository in VS Code.
3. Press F1 → Dev Containers: Open Folder in Container… → select project.


### Run the Experiment

```bash
bash run_experiments.sh
```

## Prompting OpenAI

The `data` folder contains batches of data specifically prepared for use with the `gpt-4o-mini` model.  

Upload the batches directly to the [OpenAI platform](https://platform.openai.com/).

