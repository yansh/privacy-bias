
import pandas as pd
import urllib

def load_confaide():
    tier_2a = urllib.request.urlopen("https://raw.githubusercontent.com/skywalker023/confaide/main/benchmark/tier_2a.txt")
    tier_2a_prompts = []
    for line in tier_2a:
        line=str(line.decode('utf-8'))
        line = line.split("\\n")
        tier_2a_prompts.append(line[1].strip()+" Answer: ")

    tier_2_gt = urllib.request.urlopen("https://raw.githubusercontent.com/skywalker023/confaide/main/benchmark/tier_2_labels.txt")
    tier_2_gt_list = []
    for line in tier_2_gt:
        tier_2_gt_list.append(float(line.decode('utf-8')))

    return tier_2a_prompts, tier_2_gt_list


def get_prompts(data_path, args):
    df = pd.DataFrame()
    if args.dataset == "confaide":
        prompt_list, ground_truth_list = load_confaide()
    else:
        df = pd.read_csv(data_path / '{}.csv'.format(args.dataset))
        prompt_list = df['vignette'].to_list()
        ground_truth_list = None # to be replaced

    return prompt_list, ground_truth_list, df
   
def get_dataset_df(data_path, args):
   df = pd.read_csv(data_path / '{}.csv'.format(args.dataset))
   return df
