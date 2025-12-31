#!/bin/bash

# Set your Hugging Face API token here (required for meta-llama model).
export HF_TOKEN="YOUR TOKEN HERE"

# Define available models and their templates.

declare -A models=(
    ["allenai/tulu-2-7b"]="tulu"
    ["TheBloke/tulu-2-7B-AWQ"]="tulu"
    ["allenai/tulu-2-dpo-7b"]="tulu"
    ["allenai/tulu-2-13b"]="tulu"
    ["TheBloke/tulu-2-13B-AWQ"]="tulu"
    ["allenai/tulu-2-dpo-13b"]="tulu"
    ["meta-llama/Meta-Llama-3.1-8B-Instruct"]="llama3"
)

# Default parameters
default_temps=(0)  # temperatures: 0, 0.5, 1
default_datasets=("iot")  # also available "confaide"
default_paraphrasing=("gpt")  # Default paraphrasing, other are: "gemini", "pegasus"

# If no arguments are passed, use the defaults
if [ "$#" -eq 0 ]; then
    models_list=("${!models[@]}")  # List of model names
    temps=("${default_temps[@]}")
    datasets=("${default_datasets[@]}")
    paraphrasing=("${default_paraphrasing[@]}")
else
    # Parsing command line arguments
    for i in "$@"; do
        case $i in
            --models=*)
                models_list=(${i#*=})
                ;;
            --temps=*)
                temps=(${i#*=})
                ;;
            --datasets=*)
                datasets=(${i#*=})
                ;;
            --paraphrasing=*)
                paraphrasing=(${i#*=})
                ;;
            *)
                echo "Invalid argument: $i"
                exit 1
                ;;
        esac
    done
fi

# If no models were passed, use the default list of models
if [ -z "$models_list" ]; then
    models_list=("${!models[@]}")
fi

# If no temperatures were passed, use the default temperature
if [ -z "$temps" ]; then
    temps=("${default_temps[@]}")
fi

# If no paraphrasing methods were passed, use the default paraphrasing method
if [ -z "$paraphrasing" ]; then
    paraphrasing=("${default_paraphrasing[@]}")
fi

# Display what will be used
echo "Using models: ${models_list[@]}"
echo "Using temperatures: ${temps[@]}"
echo "Using datasets: ${datasets[@]}"
echo "Using paraphrasing methods: ${paraphrasing[@]}"

# Record the start time of the whole script using builtin bash timer
SECONDS=0

# Run the experiment for all combinations of models, temperatures, datasets, and paraphrasing methods
for temp in "${temps[@]}"; do
    for model in "${models_list[@]}"; do
        template="${models[$model]}" 
        for dataset in "${datasets[@]}"; do
            for para in "${paraphrasing[@]}"; do                
                echo "Running experiment with model: $model, template: $template, temperature: $temp, dataset: $dataset, paraphrasing: $para"

                # Track time for each experiment
                exp_start=$SECONDS
                time python run_experiment.py --model "$model" --template "$template" --dataset "$dataset" --temperature "$temp" --paraphrasing "$para"

                # Track time for each experiment
                exp_duration=$((SECONDS - exp_start))

                printf "Experiment time: %02d:%02d:%02d\n\n" \
                    $((exp_duration/3600)) \
                    $((exp_duration%3600/60)) \
                    $((exp_duration%60))

            done
        done
    done
done
# Total time
total_time=$SECONDS

echo "=========================================="
printf "Total execution time: %02d:%02d:%02d\n" \
    $((total_time/3600)) \
    $((total_time%3600/60)) \
    $((total_time%60))
