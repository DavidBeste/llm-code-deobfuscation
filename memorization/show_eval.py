import argparse
import os
import warnings

from datasets import load_dataset
import numpy as np

from utils import extract_function_name, load_jsonl_dataset
from utils_eval import check_correctness2, count_elements, load_metrics, load_metrics_new, get_code

# Sample usage: python3 show_eval.py --eval_dataset_path datasets/obfuscation_dataset_encode_arithmetic_eval.json --data_suffix _encode_arithmetic --original_path datasets/original --obfuscated_path datasets/obfuscated --deobfuscated_path datasets/deobfuscated

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Repo card metadata block was not found. Setting CardData to empty.")
# define parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--eval_dataset_path', required=True)
parser.add_argument('--original_path')
parser.add_argument('--obfuscated_path')
parser.add_argument('--deobfuscated_path')
parser.add_argument('--orig_data_suffix', default='', type=lambda s: [item for item in s.split(',')])
parser.add_argument('--obfs_data_suffix')
parser.add_argument('--data_suffix', default='', type=lambda s: [item for item in s.split(',')])
args = parser.parse_args()

eval_dataset = load_jsonl_dataset(args.eval_dataset_path)
random_seed = 42
data = load_dataset("jordiae/exebench", split='test_real')
data = data.shuffle(random_seed)
#train_functions = [list(a.keys())[0].split("__name__")[1] for a in train_set]
eval_index = [list(a.keys())[0].split("__name__")[0] for a in eval_dataset]
eval_function = [list(a.keys())[0].split("__name__")[1] for a in eval_dataset]

#print(eval_index)
#print(eval_function)

successful_samples = {}

orig_to_deobfs_ratio = {}
orig_to_deobfs_ratios = {}
original_program_lengths = {}

correct_names = []
incorrect_names = {}

compiled = 0
semantically_correct = 0

original_metrics_all = []
obfuscated_metrics_all = []
deobfuscated_metrics_all = []

metrics_usable = 0
print(args.data_suffix)

for index, sample in zip(eval_index, eval_function):
#    print(f"{args.deobfuscated_path}_io_test/{sample}{args.data_suffix[0]}.exe")
    if os.path.exists(f"{args.deobfuscated_path}_io_test/{sample}{args.data_suffix[0]}.exe"):
        compiled += 1
                


#    print(len(args.data_suffix))

    correctnesses = []

    sample_usable = 1

    for suffix in args.data_suffix:
#        print("checking correctness " , suffix)
        correctness_result = check_correctness2(f"{sample}", f"{sample}{suffix}", f"{args.original_path}_io_test/", f"{args.deobfuscated_path}_io_test/", no_run=True)

#        print(correctness_result)
        if not type(correctness_result) == int or correctness_result != 1:
            sample_usable = 0
            print(sample)
#            print(f"nano {args.original_path}_eval/{sample}_tigress_out.c {args.deobfuscated_path}_eval/{sample}{suffix}.c ../FineTuning-6144-eval/datasets/original_eval/{sample}.c")
            #print(f"{args.deobfuscated_path}_eval/{sample}{suffix}.c")
            #print(f"../FineTuning-6144-eval/datasets/original_eval/{sample}.c")
            break

    if sample_usable == 0:
        continue

    #if type(correctness_result) == int and correctness_result == 1:
    semantically_correct += 1
 #   print(args.original_path + "_eval/" + sample + args.orig_data_suffix + "_metrics.json")
    original_metrics = load_metrics_new(args.original_path + "_eval/" + sample + args.orig_data_suffix[0] + "_metrics.json")
 #   print(args.obfuscated_path + "_eval/" + sample + args.obfs_data_suffix + "_metrics.json")
    obfuscated_metrics = load_metrics(args.obfuscated_path + "_eval/" + sample + args.obfs_data_suffix + "_metrics.json")
 #   print(args.deobfuscated_path + "_eval/" + sample + args.data_suffix[0] + "_metrics.json")
    deobfuscated_metrics = load_metrics(args.deobfuscated_path + "_eval/" + sample + args.data_suffix[0] + "_metrics.json")
    #print(original_metrics)
    #print(obfuscated_metrics)
    #print(deobfuscated_metrics)

    if len(original_metrics) != 3 or len(obfuscated_metrics) != 3 or len(deobfuscated_metrics) != 3:
        continue

    if -1 in original_metrics or -1 in obfuscated_metrics or -1 in deobfuscated_metrics:
        continue

#    print(original_metrics)
#    print(obfuscated_metrics)
#    print(deobfuscated_metrics)

    metrics_usable += 1
    original_metrics_all.append(original_metrics)
    obfuscated_metrics_all.append(obfuscated_metrics)
    deobfuscated_metrics_all.append(deobfuscated_metrics)


print(f"Compiled: {compiled} / {len(eval_index)}")
print(f"Semantically Correct: {semantically_correct} / {len(eval_index)}")
print(f"Samples remaining with usable metrics: {metrics_usable} / {len(eval_index)}")

print(np.mean(original_metrics_all, axis=0))
print(np.mean(obfuscated_metrics_all, axis=0))
print(np.mean(deobfuscated_metrics_all, axis=0))

print(f"Ratio for bar plots: {np.mean(original_metrics_all, axis=0)/np.mean(deobfuscated_metrics_all, axis=0)}")
ratios = np.mean(original_metrics_all, axis=0)/np.mean(deobfuscated_metrics_all, axis=0)


# uncomment this for the ratio of or and deobf program lenght
#print(f"{ratios[1]}, ", end='')
