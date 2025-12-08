"""
This code relate to the paper:
"Privacy Bias in Language Models: A Contextual Integrity-based Auditing Metric"

To appear in the Proceedings of the  Privacy Enhancing Technologies Symposium (PETS), 2026.
"""

from pathlib import Path
import locale
from typing import Optional, Any
import logging, sys
import logging
import os
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from helper.prompts import k_shot_prompts_by_gpt, k_shot_prompts_by_gemini, k_shot_prompts_by_pegasus, prompt_wrap, k_shot_prompts_confainde
from tqdm import tqdm
import torch
import random
import re
import gc
import itertools
from helper.data import *
from huggingface_hub import login

os.environ ['CUDA_LAUNCH_BLOCKING'] ='1'
os.environ['CURL_CA_BUNDLE'] = ''

# Retrieve the token from the environment variable
token = os.getenv("HF_TOKEN")

# Now use the token where needed, for example in a login function
if token:
    print(f"Logging in with token: {token}")
    login(
         token=token, 
            add_to_git_credential=True
        )
else:
    print("Token not found. Please set the token environment variable.")




def create_dir_if_doesnt_exist(path_to_dir: Path, log: logging.Logger) -> Optional[Path]:
    """Create directory using provided path.
    No explicit checks after creation because pathlib verifies it itself.
    Check pathlib source if errors happen.

    Args:
        path_to_dir (Path): Directory to be created
        log (logging.Logger): Logging facility

    Returns:
        Optional[Path]: Maybe Path to the created directory
    """
    resolved_path: Path = path_to_dir.resolve()

    if not resolved_path.exists():
        log.info("{} does not exist. Creating...")

        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.error("Error occurred while creating directory {}. Exception: {}".format(resolved_path, e))
            return None

        log.info("{} created.".format(resolved_path))
    else:
        log.info("{} already exists.".format(resolved_path))

    return resolved_path



def process_and_save_results(args, prompts, request_outputs, scenarios, shot_ids, likert_order, likert_order_ids, gt=None):
    """
    Process the generated outputs and save them to a CSV file.
    """
    answers = []
    for output in request_outputs:        
        generated_answer = output.outputs[0].text.replace("\n","")             
        answers.append(generated_answer)
            
    # Create DataFrame based on prompt type
    if args.prompt_type == "kshot":
        data = {
            "Scenario": scenarios,
            "Prompt": prompts,
            "Response": answers,
            "prompt_id": shot_ids,
            'likert': likert_order,
            'likert_order_id': likert_order_ids
        }
        if args.dataset == 'confaide':
            data["gt"] = gt
    elif args.prompt_type == "std":
        data = {
            "Scenario": scenarios,
            "prompt_id": shot_ids,
            "Prompt": prompts,
            "Response": answers,
            'likert': likert_order,
            'likert_order_id': likert_order_ids
        }
    else:
        data = {}

    # Create DataFrame for response
    response_df = pd.DataFrame(data)

    # Prepare result folder and filename
    results_folder = 'results/'
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(results_folder + args.model.split("/")[0] + "/", exist_ok=True)

    # Depending on the mode, add "pvar" to the dataset folder
    dataset_folder = args.dataset + ("pvar" if False else "")  # Change flag for pvar mode
    os.makedirs(results_folder + args.model.split("/")[0] + "/" + dataset_folder, exist_ok=True)

    # Format temperature string
    temp = str(args.temperature).replace(".", "").replace("00", "0").replace("10", "1")
    filename = f"{args.model.split('/')[1]}_{dataset_folder}_t{temp}_{args.prompt_type}.csv"
    file_path = os.path.join(results_folder, args.model.split("/")[0], dataset_folder, filename)

    # Export the DataFrame to CSV
    response_df.to_csv(file_path)

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"


def process_prompts(args, log):    
        ## disable INFO logging messages
        locale.getpreferredencoding = getpreferredencoding     
        logging.disable(sys.maxsize)
        data_path, output_path = Path(args.data_path), Path(args.output_path)
        create_dir_if_doesnt_exist(data_path, log)
        create_dir_if_doesnt_exist(output_path, log)
        sampling_params = SamplingParams(max_tokens = 100, temperature=args.temperature) #, top_p=0.95)
        model = args.model
        template = args.template
        print("Processing[: "+model+ "]")    

        scenarios =[]
        
        if args.dataset != "confaide":
            # List of Likert scale options
            likert_scale = [
                "strongly unacceptable",
                "somewhat unacceptable",
                "neutral",
                "somewhat acceptable",
                "strongly acceptable"
            ]

            # Generate all possible permutations of the Likert scale
            permutations = list(itertools.permutations(likert_scale))

            # Create the dictionary with permutations as keys and their index as values
            likert_str = {f"ordering_{i+1}": f"[{', '.join(perm)}]" for i, perm in enumerate(permutations)}
            
            # use Gemini paraphrased prompts
            if args.paraphrasing == 'gpt':
                k_shot_prompts =  k_shot_prompts_by_gpt
            if args.paraphrasing == 'gemini':
                k_shot_prompts = k_shot_prompts_by_gemini
            elif args.paraphrasing == 'pegasus': 
                k_shot_prompts = k_shot_prompts_by_pegasus #k_shot_prompts_by_gemini
            
            # returns prompts, ground_truth, dataset df
            prompt_list, _, df= get_prompts(data_path, args)
        
        elif args.dataset=="confaide":
                # List of Likert scale options
            likert_scale = [
                "strongly disagree", 
                "somewhat disagree",
                "neutral",
                "somewhat agree",
                "strongly agree"
            ]

            # Generate all possible permutations of the Likert scale
            permutations = list(itertools.permutations(likert_scale))

            # Create the dictionary with permutations as keys and their index as values
            likert_str = {f"ordering_{i+1}": f"[{', '.join(perm)}]" for i, perm in enumerate(permutations)}
        
            
            k_shot_prompts = k_shot_prompts_confainde


            # returns prompts, ground_truth, dataset df
            prompt_list, ground_truth, df= get_prompts(data_path, args)
            
        
        
        sample_size = len(prompt_list[:1])
        
        ## generate prompts to query the model
        prompts =[]
        shot_ids = []
        prompt_ids = []
        likert_order = []
        likert_order_ids=[]
        gt = []          
        for i, scenario  in tqdm(zip(range(len(prompt_list[:sample_size])), 
                                prompt_list[:sample_size]), 
                                bar_format='{l_bar}{bar:20}{r_bar}', 
                                desc ="Composing prompts"):    
            if args.prompt_type == "std":
                # For std, for each information flow (in prompt list) we generate 3 prompts with different likert scales. We don't use paraphrasing as in kshot method
                for j, order in zip(range(len(k_shot_prompts)), random.sample(list(likert_str.keys()), len(k_shot_prompts))):
                    prompt_full = f"{prompt_wrap[template]['start']} {k_shot_prompts[0].format(scenario=scenario, likert_str=likert_str[order])} {prompt_wrap[template]['end']}"                        
                    if args.dataset=="confaide":
                        gt.append(ground_truth[i])
                    scenarios.append(scenario)                
                    # if you want to randomize Likert for each prompt use: likert_str[order]
                    likert_order.append(f"[{', '.join(likert_scale)}]") 
                    likert_order_ids.append("ordering_0") #order.replace("ordering_",""))
                    prompts.append(prompt_full)
                    prompt_ids.append("p#"+str(j))
            #
            #  For Cofaide. For every prompt variation. We run 3 differet likert scales. 98 x 9 (3 in case of openAI) prompt variations x 3 random likert scales.
            #         
            likert_random = False
            if args.prompt_type == "kshot":
                for i in range(len(k_shot_prompts)):
                    if likert_random:
                        for j, order in zip(range(3), random.sample(list(likert_str.keys()), 3)):
                            prompt_full = f"{prompt_wrap[template]['start']} {k_shot_prompts[i].format(scenario=scenario, likert_str=likert_str[order])} {prompt_wrap[template]['end']}"
                            if args.dataset=="confaide":
                                gt.append(ground_truth[i])                        
                            scenarios.append(scenario)                
                            prompts.append(prompt_full)
                            likert_order.append(likert_str[order])
                            likert_order_ids.append(order.replace("ordering_",""))
                            shot_ids.append("p#"+str(i)+"_"+str(j))
                    else:
                        
                        # for prompt variation (pvar)
                        
                        prompt_full = f"{prompt_wrap[template]['start']} {k_shot_prompts[i].format(scenario=scenario, likert_str='[' + ', '.join(likert_scale) + ']')} {prompt_wrap[template]['end']}"
                        if args.dataset=="confaide":
                            gt.append(ground_truth[i])                        
                        scenarios.append(scenario)                
                        prompts.append(prompt_full)
                        likert_order.append(f"[{', '.join(likert_scale)}]") 
                        likert_order_ids.append("0")
                        shot_ids.append("p#"+str(i))

        
        llm = LLM(model, 
                trust_remote_code=True,
                max_model_len = 1042,
                dtype='bfloat16',
                gpu_memory_utilization=0.9,
                enforce_eager=True
                )

        request_outputs = llm.generate(prompts, sampling_params)

        process_and_save_results(args, prompts, request_outputs, scenarios, shot_ids, likert_order, likert_order_ids, gt)

        # Cleanup
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()