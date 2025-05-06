import argparse
from collections import Counter
import json
import logging
import os
import re
import random

import clang.cindex as cindex
from datasets import load_dataset
from transformers import AutoTokenizer

from utils import (
    build_bogus_parameters,
    build_program,
    check_sample_preconditions,
    check_token_length_limits,
    count_arithmetic_operations,
    count_branches,
    count_function_parameters,
    create_folder_if_not_exists,
    extract_function2 as extract_function,
    extract_parameter_permutations,
    find_unresolved_symbols_function,
    get_random_chain,
    normalize_data_types,
    remove_comments,
    save_dataset_raw,
    save_dataset_json,
    save_input_samples,
    timeit
)
from utils_eval import check_correctness2, get_fake_function_parameter_types, obfuscate

#chain_length = 1

error_log = []    
detailed_error_log = {}

exebench_test_path = "data_0_time1678114487_default.jsonl" # Is the real test part of the dataset
cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-14.so.1")  # Set the path to libclang.so

# define parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', help='The tokenizer to use for the sample filtering')
parser.add_argument('--max_tokens', type=int)
parser.add_argument('--chain_length', type=int)
parser.add_argument('--number_of_samples', type=int)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

chain_length = args.chain_length

# Configure the logging system
logging.basicConfig(filename=f'datasets/eval_data_{chain_length}_eval_err.log', level=logging.ERROR,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
                    
@timeit
def main():
    number_of_samples = args.number_of_samples
    data = load_dataset("jordiae/exebench", split='test_real')
    data = data.shuffle(42)
#    print(data['fname'][:40])
    
    total_samples = 0
    raw_deobf_dataset = ""
    json_deobf_dataset = [] # use instruction and output
    json_deobf_dataset2 = []

    raw_deobf_datasets = {}
    json_deobf_datasets = {}
    json_deobf2_datasets = {}
    random.seed(42)

    training_samples = number_of_samples
    #a = load_jsonl_dataset("obfuscation_dataset_chain2.json")
    #n1 = [list(c.keys())[0].split("__name__")[1] for c in a]

   # control = input("Delete existing metadata (Y/N)?")

   # if control == "Y":
   #     os.system("rm datasets/original_eval_io_test/*.exe datasets/obfuscated_io_test/*.exe datasets/original_eval_io_test/*/*.json datasets/obfuscated_io_test/*/*.json")

    names = []
    mapping_table = {}

    iteration_timeout = 200
    iterations = 0

    # second condition prevents infinite loop
    while total_samples < number_of_samples and iterations < iteration_timeout:
        iterations += 1
        for i, sample in enumerate(data):
 #           if i >= training_samples:
 #               break
            
            if total_samples >= number_of_samples:
                break
                    

            status_array = check_sample_preconditions(sample, names)

            if status_array != [0, 0, 0, 0, 0, 0, 0]:
                training_samples += 1
                continue

            # Wouldn't need to be placed so high but we do it for error logging purposes
            t_chain = get_random_chain(chain_length, replacement=True)

            #if sample['fname'] in n1:
            #    training_samples += 1
            #    continue
            func = sample['func_def']

            try:
                num_ops = count_arithmetic_operations(func)
                num_brns = count_branches(func)

            except:
                error_log.append("Failed to extract number of arithmetic operations and / or branches")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                
                training_samples += 1
                continue

            if num_ops < 1 or num_brns < 1:
                error_log.append(f"Code is too simple: Too little arithmetic operations: {num_ops < 1}, Too little branches: {num_brns < 1}")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                print("Code is too simple")
                training_samples += 1
                continue

        #    t_chain = []

        #    while not check_transformation_order(t_chain) or t_chain == []:
        #        transformations = ["encode_arithmetic", "encode_branches", "flatten", "opaque", "randomize_arguments"]
        #        transformation_count = random.randint(0,5)
        #        t_chain = []

        #        for j in range(transformation_count):
        #            random_transformation = random.choice(transformations)
        #            t_chain.append(random_transformation)
        #            transformations.remove(random_transformation)
            

            print(t_chain)
            print(sample['fname'])
            function_name_counter = names.count(sample['fname'])

            if function_name_counter > 0: # where is names appended
                error_log.append("Duplicate function name")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                print("duplicate")
                training_samples += 1
                continue

            else:
                duplicate_string = ""

            # skip if the function name is too long
            if len(sample['fname']) > 256-len(os.getcwd()):
                error_log.append("Function name is too long")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                training_samples += 1
                print("Function name is too long")
                continue

            # dont include main functions as well as this messes up the current pipeline
            if sample['fname'] == "main":
                error_log.append("Main function")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                training_samples += 1
                print("Invalid function name")
                continue

            #if sample['fname'] in n1:
            #    continue

            with open("datasets/original_eval/" + sample['fname'] + "_" + str(chain_length) + duplicate_string + ".c", "w") as f:
                f.write(normalize_data_types(build_program(sample=sample)))

            param_count = count_function_parameters(sample['func_def'], sample['fname'])
            obfuscation_status = obfuscate(sample['fname'] + duplicate_string + ".c", sample, sample['fname'], param_count, t_chain)
            
            if obfuscation_status != 0:
                error_log.append(f"Obfuscation error: {['Process Timeout', 'Failed to extract intermediate program'][-obfuscation_status-1]}")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                print("Obfuscation error")
                training_samples += 1
                continue

            suffix = f"_{chain_length}_chain.c"

            if not os.path.exists("datasets/obfuscated_eval/" + sample['fname'] + duplicate_string + suffix):
                error_log.append("Obfuscated tigress file does not exist")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                training_samples += 1
                continue

            # we could randomize the identifier names even further since tigress uses a fixed naming convention after CleanUp for id names but we could also argue that any program that is to be deobfuscated could be conver>
            
            test = extract_function("datasets/obfuscated_eval/" + sample['fname'] + duplicate_string + suffix, sample['fname'], "", False, "opaque" in t_chain, "encode_branches" in t_chain)

            if type(test) == str:
                error_log.append("Failed to extract obfuscated function from tigress code")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                errors += 1
                continue

            # erroneous extractions are skipped
            if test[1] == "":
                error_log.append("Failed to extract obfuscated function from tigress code, however tuple was returned")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                print(f"Error during the extraction")
                training_samples += 1
                continue
            
            print("datasets/obfuscated_eval/" + sample['fname'] + duplicate_string + suffix)
            target_function = extract_function("datasets/obfuscated_eval/" + sample['fname'] + duplicate_string + suffix, sample['fname'], "", False, "opaque" in t_chain, "encode_branches" in t_chain)
            obfuscated_function_name, obfuscated_function = extract_function("datasets/obfuscated_eval/" + sample['fname'] + duplicate_string + suffix, sample['fname'], "", False, "opaque" in t_chain, "encode_branches" in t_chain, extract_helpers=False)
            obfuscated_function_name, obfuscated_code = extract_function("datasets/obfuscated_eval/" + sample['fname'] + duplicate_string + suffix, sample['fname'], "", False, "opaque" in t_chain, "encode_branches" in t_chain)
            original_function = extract_function("datasets/original_eval/" + sample['fname'] + "_" + str(chain_length) + duplicate_string + "_tigress_canonicalized.c", sample['fname'], extract_helpers=False)[1]
            code = "// Obfuscated code\n" + remove_comments(obfuscated_function) + "\n// Deobfuscated code\n" + remove_comments(original_function) + "<|end|>" # <|end|> is a makeshift EOS token, will be replaced by the EOS token depending on the chosen tokenizer later

            obfs, orig = code.split("// Obfuscated code\n")[1].split("<|end|>")[0].split("\n// Deobfuscated code\n")

            # also obfuscated-original_eval-pairs that are too large for the model are left out, continue sampling until the desired sample count is achieved without the special cases here
#            if len(tokenizer(code)['input_ids']) > args.max_tokens:
            if not check_token_length_limits(obfs, orig, [("deepseek-coder-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct"), ("codellama", "codellama/CodeLlama-7b-hf"), ("gpt-4", "gpt-4")], args.max_tokens):
                error_log.append(f"Sample exceeds {args.max_tokens} tokens")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                print("code is too long")
                training_samples += 1
                continue

            if obfuscated_function_name == "" or obfuscated_function_name == None:
                print("Empty obfuscated function name")
                error_log.append("Failed to extract obfuscated function name from tigress code")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                training_samples += 1
                continue

            original_eval_function = build_program(sample=sample, no_deps=True)
            obfuscated_program = build_program(sample=sample, empty_main=False, func_def_is_external=True, func_def=obfuscated_code)

            with open("datasets/original_eval_io_test/" + sample['fname'] + "_" + str(chain_length) + ".cpp",  "w") as f:
                f.write(build_program(sample=sample, empty_main=False, func_def_is_external=False)) 

            create_folder_if_not_exists("datasets/input_samples/" + sample['fname'] + "_" + str(chain_length))
            save_input_samples(sample, f"_{chain_length}")
            original_eval_function = build_program(sample=sample, no_deps=True)
            obfuscated_program = build_program(sample=sample, empty_main=False, func_def_is_external=True, func_def=obfuscated_code)

            if "randomize_arguments" in t_chain:
                try:
                    function_arguments = re.search(sample['fname'] + r"\s*\([^)]*\)", sample['real_exe_wrapper']).group().removeprefix(sample['fname'] + "(").removesuffix(")").split(",")
                except:
                    error_log.append("Failed to find the call to the target function in the real_exe_wrapper")
                    detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                    logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                    print("Regular expression search fail")
                    training_samples += 1
                    continue

                param_perm = extract_parameter_permutations(sample['func_def'], extract_function("datasets/obfuscated_eval/" + sample['fname'] + "_" + str(chain_length) + "_chain_randomize_arguments_intermediate.c", sample['fname'])[1], sample['fname'])
                print(param_perm)
                new_params, additional_vars = build_bogus_parameters(function_arguments, param_perm)
                new_function_call = sample['fname'] + "(" + ",".join(new_params) + ")"

                modified_exe_wrapper = sample['real_exe_wrapper']
                modified_exe_wrapper = re.sub(sample['fname'] + r"\s*\([^)]*\)", new_function_call, modified_exe_wrapper)

                if additional_vars != "":
                    main_head = re.search("main\s*\([^)]*\)\s*{", modified_exe_wrapper).group()
                    modified_exe_wrapper = re.sub("main\s*\([^)]*\)\s*{", main_head + "\n" + additional_vars, modified_exe_wrapper)
                    
                obfuscated_program = build_program(sample=sample, empty_main=False, func_def_is_external=True, func_def=obfuscated_code, modified_wrapper=modified_exe_wrapper)

            else:
                obfuscated_program = build_program(sample=sample, empty_main=False, func_def_is_external=True, func_def=obfuscated_code)


            if sample['fname'] + "__bench(" in obfuscated_program:
                obfuscated_program = re.sub(sample['fname'] + r"(?!__bench)\(", sample['fname'] + "__bench(", obfuscated_program)

            obfuscated_program = re.sub(sample['fname'] + r"\s*\(", obfuscated_function_name + "(", obfuscated_program)

            with open("datasets/obfuscated_io_test/" + sample['fname'] + suffix.removesuffix(".c") + ".cpp", "w") as f:
                f.write(obfuscated_program)

            
            fake_calls = find_unresolved_symbols_function(obfuscated_code)
            fake_parameters = [get_fake_function_parameter_types(obfuscated_code, fake_call) for fake_call in fake_calls]
            fake_func_defs = "\n".join(["unsigned int " + fake_call + "(" + ",".join(fake_parameter) + "){return 0;}\n" for fake_call, fake_parameter in zip(fake_calls, fake_parameters)])

            print(fake_func_defs)

            print(fake_parameters)

            with open("datasets/obfuscated_eval/" + sample['fname'] + suffix + "_function.c", "w") as f:
                f.write(obfuscated_code)

            is_correct = check_correctness2(sample['fname'] + "_" + str(chain_length), sample['fname'] + suffix.removesuffix(".c"), "datasets/original_eval_io_test/", "datasets/obfuscated_io_test/")
            print(is_correct)

            if not type(is_correct) == int or is_correct == 0:
                error_log.append(f"Sample broke during evaluation, {[f'Evaluation failed before actual comparison (e.g.) crash: {is_correct}', 'Obfuscation is semantically incorrect'][type(is_correct) == int]}")
                detailed_error_log[f"{i}__{sample['fname']}__{t_chain}"] = error_log[-1]
                logging.error(f"An error occurred: " + f"{i}__{sample['fname']}__{t_chain}__{error_log[-1]}", exc_info=True)
                print("Sample broke during obfuscation evaluation", is_correct)
                training_samples += 1
                continue

            code = "<|OBFS|>\n" + obfs + "<|ORIG|>\n" + orig

            names.append(sample['fname'])
            mapping_table[sample['fname']] = "__".join(t_chain)

            raw_deobf_dataset += code
            json_deobf_dataset.append({'instruction' : "// Obfuscated code\n" + remove_comments(obfuscated_function) + "\n Deobfuscated code\n", 'input' : '', 'output' : remove_comments(original_function)})
            json_deobf_dataset2.append({str(i) + "__name__" + sample['fname'] : code})
            total_samples += 1
            print(f"Total no. of samples: {total_samples}")

   

    save_dataset_raw(raw_deobf_dataset, "_" + str(chain_length) + "_chain_eval_l.c")
    save_dataset_json(json_deobf_dataset, "_" + str(chain_length) + "_chain_eval_l.c")
    save_dataset_json(json_deobf_dataset2, "_" + str(chain_length) + "_chain_eval2_l.c")

    with open("datasets/transformations_" + str(chain_length) + "_chain_eval.json", "w") as f:
        json.dump(mapping_table, f)

    print(f"i = {i}\ntraining_samples = {training_samples}\ntotal_samples = {total_samples}")

    error_dict = Counter(error_log)

    with open(f"datasets/eval_data_{chain_length}_eval_err.json", "w") as f:
        json.dump(error_dict, f)

    with open(f"datasets/eval_data_{chain_length}_eval_err_detailed.json", "w") as f:
        json.dump(detailed_error_log, f)

if __name__ == "__main__":
    main()
