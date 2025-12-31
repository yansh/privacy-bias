# Privacy Bias

**Paper title**: Privacy Bias in Language Models: A Contextual Integrity-based Auditing Metric

**Requested Badge(s):**
  - [x] Available
  - [x] Functional
  - [x] Reproduced

## Table of Contents
1. [Description](#description)
2. [Important Files](#important-files)
3. [Minimum Hardware Requirements](#minimum-hardware-requirements)
4. [Minimum Software Requirements](#minimum-software-requirements)
5. [Estimated Time and Storage Consumption](#estimated-time-and-storage-consumption)
6. [Setup Instructions](#setup-instructions)
   - [Step 1: Build the Docker Image](#step-1-build-the-docker-image)
   - [Step 2: Run the Docker Container](#step-2-run-the-docker-container)
   - [Step 3: Run the Experiment](#step-3-run-the-experiment)
7. [Prompting the `gpt-4o-mini` Model on OpenAI Platform](#prompting-the-gpt-4o-mini-model-on-openai-platform)
8. [Exporting figures](#exporting-figures)
9. [Summary of Main Results and Takaways](#summary-of-main-results-and-takaways)

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

*  `data` - datasets of all the generated vignettes
*  `data/openAI` -  openAI prompt batches
* `plots/plots.ipynb` — Jupyter notebook to generate paper figures.
* `plots/dataframes` - Results dataframes.
* `raw_results.7z` — Archive containing raw CSV results.
  * To unpack: `7z x raw_results.7z`
* `run_experiments.sh`  - bash script to run prompts agains the LLMs
* `export_figures.sh`  - bash script to export all the paper figures

### Notes:



- **Plot Reproducibility**:  
  Plots can be reproduced without a GPU, using dataframes containing processed raw data.

- **GPU & API Keys**:  
  Running new models requires a GPU and Hugging Face API keys.


## Minimum Hardware Requirements

- NVIDIA GPU (tested on RTX 4090)
- VRAM: 24 GB
  - Note allenai/tulu-2-13b and allenai/tulu-2-dpo-13b require 80 GB 
- Driver: 550.127.05
- CUDA: 12.4

## Minimum Software Requirements

- Docker Engine 28.3.3+
- git 2.39.5+
- NVIDIA Drivers (tested with 550.127.05)
- NVIDIA Container Toolkit


## Estimated Time and Storage Consumption


Model                       | Temperature | Dataset | Paraphrasing | Experiment Time
----------------------------|-------------|---------|--------------|----------------
Tulu-2-7B-AWQ               | 0           | iot     | gpt          | 00:21:05
Meta-Llama-3.1-8B-Instruct  | 0           | iot     | gpt          | 00:18:42
Tulu-2-dpo-7b               | 0           | iot     | gpt          | 00:33:00
Tulu-2-13B-AWQ              | 0           | iot     | gpt          | 00:41:59
Tulu-2-7b                   | 0           | iot     | gpt          | 00:25:21
Tulu-2-dpo-13b<sup>*</sup>  | 0           | iot     | gpt          | 00:49:57
Tulu-2-13b<sup>*</sup>      | 0           | iot     | gpt          | 00:18:48
----------------------------|-------------|---------|--------------|----------------
Total                       |             |   -    |  -            | 3:28:52

* Experiments marked with an asterisk (*) were run on a server with 80GB VRAM.

**Note:** As discussed below in the [‘gpt-4o-mini’ model](#prompting-openai) experiments are run using the OpenAI platform. The experiment times would depend on the the OpenAI platform. While most batches complete in under 1 hour, they may sometimes take much longer.

### Storage

**Raw Data Storage**:    The repository contains raw CSV which requires around 3.1 GB of storage to unarchive.
  This includes results_PETS (1.6GB), results_temp/ (201MB) and results_paraphrasing/ (1.4GB).
  For loading models from disk, each models considered in our work take up atmost 10GB each.


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


#### Running with Selected Arguments:

You can specify specific models, temperatures, datasets, and paraphrasing methods using the command-line arguments:

- **`--models=<model1> <model2> ...`**: Space-separated list of models to run experiments with (e.g., `--models="allenai/tulu-2-7b meta-llama/Meta-Llama-3.1-8B-Instruct"`).  
- **`--temps=<temp1> <temp2> ...`**: Space-separated list of temperatures to use (e.g., `--temps="0 0.5"`). 
- **`--datasets=<dataset1> <dataset2> ...`**: Space-separated list of datasets to use (e.g., `--datasets="iot confaide"`). 
- **`--paraphrasing=<method1> <method2> ...`**: Space-separated list of paraphrasing methods (e.g., `--paraphrasing="gpt gemini"`).

*Example usage:*

`bash run_experiments.sh --models="allenai/tulu-2-7b" --temps="0.5" --datasets="iot confaide"`




### Prompting the `gpt-4o-mini` Model on OpenAI Platform

This part of the experiment requires using the OpenAI platform and an OpenAI secret token linked to a created project. It allows you to run prepared prompt batches on the `gpt-4o-mini` model and collect the results for analysis.

**Requirements:**  
- An OpenAI account.  
- An OpenAI secret token associated with a created project.  
- The `data` folder containing batches of prompts prepared for `gpt-4o-mini`.

**Steps to Reproduce:**  
1. Sign in to the [OpenAI platform](https://platform.openai.com/) and create a new project.  
2. Ensure your OpenAI secret token is linked to the project.  
3. Prepare the prompt batches located in the `data` folder.  
4. Upload the batches directly through [OpenAI's platform batch interface](https://platform.openai.com/batches/) to run the prompts.  
5. After processing, download the results, under Files section.
6. The raw results are saved in the `raw_results` folder, ready for further analysis and processing.

## Exporting Figures

This script requires **Python >= 3.10**.

Use this command to export the paper figures into the `figs` folder
```bash
bash export_figures.sh --export figs
```

## Summary of Main Results and Takeaways

For a summary of the results and main takeaways, please visit the [website](https://yansh.github.io/privacy-bias/website/).

*The results can be reproduced using the Jupyter notebook in [`plots/plots.ipynb`](https://github.com/yansh/privacy-bias/blob/main/plots/plots.ipynb) along with the provided dataframes.*

### Demonstrating Prompt Sensitivity

Figure 3 shows significant variance in responses due to paraphrasing and changing the Likert scale order, which hinders the reliable evaluation of privacy biases. Figure 5 further illustrates variance caused by prompt variation, with three random Likert scale orders per prompt.

### Identifying Privacy Biases

`gpt-4o-mini` and `llama-3.1-8B` exhibit several notable privacy biases. Across all senders, information types, and recipients, for fixed transmission principles (except *stored indefinitely* and *used for advertising*), `gpt-4o-mini` is less conservative, with privacy biases ranging from *strongly acceptable* to *somewhat acceptable*. In contrast, `llama-3.1-8B` is more conservative, generally ranking information flows as *somewhat unacceptable*.

### Demonstrating Impact of LLM Configuration

Figures 8 (Base LLMs with different capacities), Figure 9 (Base vs. Aligned LLMs), and Figure 10 (Base vs. Quantized LLMs) show that privacy biases vary across different capacities and optimizations, even when the training dataset is similar.

## Security/Privacy Issues and Ethical Concerns  

The use of LLM has societal implications. In our evaluation, we use latest LLMs that require a large amount of energy and resources to maintain. While our work has relatively little environmental impact because we are not training or fine-tuning the models, we acknowledge that through the use of tools like OpenAI in our research we contribute to the overall negative effect of these systems on the environment. Our work carries also a social implication: using the theory of Contextual Integrity abrings additional layer of normative rigor in evaluating LLM-based system in understanding how they contribute to the purpose, values and functions in the contexts they operate. 

We do not demonstrate any novel attacks and hence, there is no direct potential for misuse. Further, our work does not have any user studies and we do not require ethics approval form our institute's IRB.