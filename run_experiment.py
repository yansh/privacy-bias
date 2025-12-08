"""
This code relate to the paper:
"Privacy Bias in Language Models: A Contextual Integrity-based Auditing Metric"

To appear in the Proceedings of the  Privacy Enhancing Technologies Symposium (PETS), 2026.
"""

import logging
from helper.utils import *
import argparse

def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="data/", help='Root directory of the dataset')
    parser.add_argument('--output_path', type=str, default="output/", help='Root directory of the dataset')
    parser.add_argument('--dataset', type=str, default="coppa", help='Options: pubrec, coppa, confaide')
    parser.add_argument('--template', type=str, default="tulu", help='Options: llama, llama2, llama3, tulu')
    parser.add_argument('--model', type=str)
    parser.add_argument('--size', type=str, default="13b", help='Options: 7b, 13b, 70b')
    parser.add_argument('--temperature', type=float, default=0, help='Options: values from 0 to 1 ')
    parser.add_argument('--model_type', type=str, default="align", help='Options: align, nonalign, quantized')
    parser.add_argument('--prompt_type', type=str, default="kshot", help='Options: std, cot, kshot')
    parser.add_argument('--paraphrasing', type=str, default="gpt", help='Options: gpt, gemini, pegasus')


    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="LLM-CI.log", filemode="w")
    log: logging.Logger = logging.getLogger("LLM-CI")
    args = handle_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process_prompts(args, log)

