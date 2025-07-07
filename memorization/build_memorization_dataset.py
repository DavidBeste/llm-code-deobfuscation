import argparse
from collections import Counter
import json
import os
import random
import re
import shutil

from datasets import load_dataset

from utils import build_bogus_parameters, build_program, check_if_compilable2, count_function_parameters, extract_function, extract_parameter_permutations, load_jsonl_dataset, save_jsonl_dataset
from utils_eval import check_correctness2, obfuscate_simple, run_semantical_tests
from utils_memorization import check_constant_minimum, randomize_constants_from_c_code

# define parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_dataset', required=True)
parser.add_argument('--output_dataset', required=True)
parser.add_argument('--original_path', required=True)
parser.add_argument('--obfuscated_path', required=True)
parser.add_argument('--deobfuscated_path', required=True)
parser.add_argument('--data_suffix', required=True, default='', type=lambda s: [item for item in s.split(',')])

args = parser.parse_args()
# Sample: python3 build_memorization_dataset.py --input_dataset ../FineTuning/datasets/obfuscation_dataset_encode_arithmetic_eval.json --output_dataset dummy_output
# Sample: python3 build_memorization_dataset.py --input_dataset ../FineTuning/datasets/obfuscation_dataset_encode_arithmetic_eval.json --output_dataset dummy_output --original_path ../FineTuning/datasets/original --deobfuscated_path ../FineTuning/datasets/deobfuscated --data_suffix _encode_arithmetic
# Sample: python3 build_memorization_dataset.py --input_dataset ../FineTuning/datasets/obfuscation_dataset_encode_arithmetic_eval.json --output_dataset dummpy_output --original_path ../FineTuning/datasets/original --obfuscated_path ../FineTuning/datasets/obfuscated --deobfuscated_path ../FineTuning/datasets/deobfuscated --data_suffix _encode_arithmetic,_encode_branches,_flatten,_opaque,_randomize_arguments
dataset = load_jsonl_dataset(args.input_dataset)

random_seed = 42
data = load_dataset("jordiae/exebench", split='test_real')
data = data.shuffle(random_seed)
random.seed(random_seed)

errors_at_deobfuscation = []

def is_well_formatted_json(file_path):
    """
    Check if the file contains well-formatted JSON.
    
    Args:
    file_path (str): Path to the file to be checked.
    
    Returns:
    bool: True if the file contains well-formatted JSON, False otherwise.
    """
    try:
        with open(file_path, 'r') as file:
            json.load(file)

        return True

    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        return False

def check_output_samples_format(path, name):
    for i in range(10):
        print(f"{path}/{name}/output_{i}.json")
        if not is_well_formatted_json(f"{path}/{name}/output_{i}.json"):
            return -1

    return 0

def check_output_similarity(path1, name1, path2, name2):
    score = 0
    for i in range(10):
        with open(f"{path1}/{name1}/output_{i}.json", "r") as f:
            out1 = json.load(f)

        with open(f"{path2}/{name2}/output_{i}.json", "r") as f:
            out2 = json.load(f)
    if out1 == out2:
        score += 1
    print(jsondiff(out1, out2))
    return score

def delete_outputs(path):
    for i in range(10):
        if os.path.exists(f"{path}/output_{i}.json"):
            os.remove(f"{path}/output_{i}.json")

transformations = ["encode_arithmetic", "encode_branches", "flatten", "opaque", "randomize_arguments"]
memorization_datasets = {}

for t in transformations:
    memorization_datasets[t] = []

for sample in dataset:
    sample_id = list(sample.keys())[0].split("__name__")[0]
    sample_name = list(sample.keys())[0].split("__name__")[1]
    original_sample = list(sample.values())[0].split("<|ORIG|>")[1]
    sample_usable = 1

    # ToDo: Also check semantical correctness
    if not check_constant_minimum(original_sample):
        sample_usable = 0

    for suffix in args.data_suffix:
        correctness_result = check_correctness2(f"{sample_name}", f"{sample_name}{suffix}", f"{args.original_path}_io_test/", f"{args.deobfuscated_path}_io_test/", no_run=True)

        if not type(correctness_result) == int or correctness_result != 1:
            sample_usable = 0
            break

    if sample_usable == 0:
        continue

    print(f"{args.original_path}_eval/{sample_name}")
    with open(f"{args.original_path}_eval/{sample_name}.c", "r") as f:
        original_content = f.read()

    print(original_content)
    print(check_if_compilable2(original_content))

    randomized_code = randomize_constants_from_c_code(original_content)
    i = 0
    run_result = -1

    while i < 100:
        sample_usable = 1
        print(randomized_code)
        randomized_code = randomize_constants_from_c_code(original_content)
        i += 1

        if not check_if_compilable2(randomized_code):
            sample_usable = 0
            continue

        with open(f"MemTest/original_modified_eval/{sample_name}.c", "w") as f:
            f.write(randomized_code)

        program = data[int(list(sample.keys())[0].split("__name__")[0])]
        print(randomized_code)
        obfuscated_program = build_program(sample=program, empty_main=False, is_main=False, func_def_is_external=True, func_def=randomized_code)

        with open(f"MemTest/original_modified_io_test/{sample_name}.cpp", "w") as f:
            f.write(obfuscated_program)

        for j in range(1):
            delete_outputs(f"MemTest/original_modified_io_test/{sample_name}")
            run_result = run_semantical_tests("MemTest/original_modified_io_test", "datasets/input_samples", sample_name, "")
            if run_result != 0 or check_output_samples_format("MemTest/original_modified_io_test", sample_name) != 0:
                sample_usable = 0
                break

        if check_output_similarity(f"{args.original_path}_io_test/", sample_name, "MemTest/original_modified_io_test/", sample_name) > 6:
            sample_usable = 0 

        if sample_usable == 1:
            break

        print(check_output_samples_format("MemTest/original_modified_io_test/", sample_name))

    if sample_usable == 1:
        errors = 0

        param_count = count_function_parameters(program['func_def'], program['fname'])
        obfuscation_status = obfuscate_simple(f"{sample_name}.c", sample_name, param_count, f"MemTest/original_modified_eval/", f"MemTest/obfuscated_modified_eval/", is_eval=True)

        for t in transformations:
            test = extract_function(f"MemTest/obfuscated_modified_eval/{sample_name}_{t}.c", sample_name, "", False, t == "opaque", t == "encode_branches")
            print(test)

            if type(test) == str:
                errors += 1
                errors_at_deobfuscation.append(f"transformation {t}: Error during the extraction")
                break

            # erroneous extractions are skipped
            if test[1] == "":
                print(f"transformation {t}: Error during the extraction")
                errors += 1
                errors_at_deobfuscation.append(f"transformation {t}: Error during the extraction")
                break

            obfuscated_function_name, obfuscated_function = extract_function(f"MemTest/obfuscated_modified_eval/{sample_name}_{t}.c", sample_name, "", False, t == "opaque", t == "encode_branches", extract_helpers=False)
            obfuscated_function_name, obfuscated_code = extract_function(f"MemTest/obfuscated_modified_eval/{sample_name}_{t}.c", program['fname'], "", False, t == "opaque", t == "encode_branches")
            print(obfuscated_function)

            if t == "randomize_arguments":
                if not program['fname'] in program['real_exe_wrapper']: # obfuscated_program before
                    errors += 1
                    print("Function name not found")
                    errors_at_deobfuscation.append("Function name not found in wrapper")
                    continue

                wrapper = program['real_exe_wrapper']

                if not re.search(program['fname'] + r"\s*\([^)]*\)", program['real_exe_wrapper']):
                    errors += 1
                    errors_at_deobfuscation.append("Function head not found with regex in wrapper")
                    continue

                function_arguments = re.search(program['fname'] + r"\([^)]*\)", program['real_exe_wrapper']).group().removeprefix(program['fname'] + "(")[:-1].split(",")
                param_perm = extract_parameter_permutations(program['func_def'], extract_function(f"MemTest/obfuscated_modified_eval/{sample_name}_randomize_arguments_intermediate.c", program['fname'])[1], program['fname'] )               
                new_params, additional_vars = build_bogus_parameters(function_arguments, param_perm)
                new_function_call = program['fname'] + "(" + ",".join(new_params) + ")"

                wrapper = re.sub(program['fname'] + r"\s*\([^)]*\)", new_function_call, wrapper)

                if additional_vars != "":
                    main_head = re.search("main\s*\([^)]*\)\s*{", wrapper).group()
                    wrapper = re.sub("main\s*\([^)]*\)\s*{", main_head + "\n" + additional_vars, wrapper)

                obfuscated_program = build_program(sample=program, empty_main=False, is_main=False, func_def_is_external=True, func_def=obfuscated_code, modified_wrapper=wrapper)

            else:
                obfuscated_program = build_program(sample=program, empty_main=False, is_main=False, func_def_is_external=True, func_def=obfuscated_code)

            obfuscated_program = re.sub(program['fname'] + r"\s*\(", obfuscated_function_name + "(", obfuscated_program)

            with open("MemTest/obfuscated_modified_io_test/" + program['fname'] + "_" + t + ".cpp", "w") as f:
                f.write(obfuscated_program)

            mem_sample = f"<|OBFS|>{obfuscated_function}<|ORIG|>{randomized_code}"
            delete_outputs(f"MemTest/original_modified_io_test/{sample_name}")
            delete_outputs(f"MemTest/obfuscated_modified_io_test/{sample_name}_{t}")
            correctness_result = check_correctness2(f"{sample_name}", f"{sample_name}_{t}", f"MemTest/original_modified_io_test/", f"MemTest/obfuscated_modified_io_test/", no_run=False)

            if type(correctness_result) == int and correctness_result == 1:
                # copy the tigress file to the target dir so that we can use eval_deobf to extract the obfuscated name when computing the metrics
                memorization_datasets[t].append({list(sample.keys())[0] : mem_sample})

            else:
                errors_at_deobfuscation.append(f"Apparently, tigress messed things up: {correctness_result}")

for t in transformations:
    save_jsonl_dataset(memorization_datasets[t], f"{args.output_dataset}_{t}")

print(Counter(errors_at_deobfuscation))