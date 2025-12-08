#!/bin/bash


declare -A models

# meta-llama requires hugging face permissions. Export your HF token here.

export HF_TOKEN="YOUR TOKEN HERE"  


models=(    
    ["allenai/tulu-2-7b"]="tulu"
    ["TheBloke/tulu-2-7B-AWQ"]="tulu"
    ["allenai/tulu-2-dpo-7b"]="tulu"
    
    ["allenai/tulu-2-13b"]="tulu"
    ["TheBloke/tulu-2-13B-AWQ"]="tulu"
    ["allenai/tulu-2-dpo-13b"]="tulu"
    ["meta-llama/Meta-Llama-3.1-8B-Instruct"]="llama3"
)

datasets=("iot")  # "confaide" "coppa"
temps=(0 ) #0.5 1)
paraphrasing=("gpt") # "gemini" "pegasus")

for temp in "${temps[@]}"; do
    for model in "${!models[@]}"; do
        template="${models[$model]}"
        for dataset in "${datasets[@]}"; do
            for para in "${paraphrasing[@]}"; do
                echo "Running experiment with model: $model, template: $template, dataset: $dataset, temperature: $temp, paraphrasing: $para"
                python run_experiment.py --model "$model" --template "$template" --dataset "$dataset" --temperature "$temp" --paraphrasing "$para"
            done
        done
    done
done
