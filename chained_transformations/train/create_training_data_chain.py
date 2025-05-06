import argparse
import json
import os
import random
from shlex import split
import subprocess
import sys
import time

import clang.cindex as cindex
from datasets import load_dataset
from transformers import AutoTokenizer

from randomize_idns import (
    get_identifier_names,
    post_process,
    randomize_identifiers2
)
from utils import (
    backup_directory,
    build_program,
    check_sample_preconditions,
    check_token_length_limits,
    count_arithmetic_operations,
    count_branches,
    count_function_parameters,
    extract_function2 as extract_function,
    get_random_chain,
    normalize_data_types,
    remove_comments,
    save_dataset_raw,
    save_dataset_json,
    timeit
)

exebench_test_path = "data_0_time1678114487_default.jsonl" # Is the real test part of the dataset
cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-14.so.1")  # Set the path to libclang.so
# define parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', help='The tokenizer to use for the sample filtering')
parser.add_argument('--max_tokens', type=int)
parser.add_argument('--chain_length', type=int)
parser.add_argument('--number_of_samples', type=int)
args = parser.parse_args()
too_longs = 0
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

def obfuscate(filename, target_function, parameter_count, t_chain):
    tigress_header = "#include \"/usr/local/bin/tigresspkg/3.3.3/tigress.h\"\n"

    opaque_choice = ",".join(random.sample(["list", "array", "env"], random.randint(1,3)))
    opaque_choice2 = random.sample(["call", "bug", "true", "junk", "fake_call", "question"], 1)

    opaque_chain = ""

    for choice in opaque_choice2:
        opaque_chain += " --Transform=AddOpaque --Functions=" + target_function + " --AddOpaqueStructs=" + opaque_choice + " --AddOpaqueKinds=" + choice

    if "question" in opaque_choice2:
        opaque_chain += " --Transform=Inline --Functions=/.*QUESTION.*/"

    command_templates = {
        "basic" : " --Transform=CleanUp --CleanUpKinds=annotations",
        "encode_arithmetic" : " --Transform=EncodeArithmetic --Functions=" + target_function,
        "encode_branches" : " --Transform=AntiBranchAnalysis --Functions=" + target_function + " --AntiBranchAnalysisKinds=branchFuns --AntiBranchAnalysisObfuscateBranchFunCall=false --AntiBranchAnalysisBranchFunFlatten=true",
#        "encode_data" : " --Transform=EncodeData --LocalVariables=" + target_function + ":" + target_local_variables + " --EncodeDataCodecs=poly1,
        "flatten" : " --Transform=Flatten --Functions=" + target_function + " --FlattenRandomizeBlocks=" + ["true", "false"][random.randint(0,1)] + " --FlattenSplitBasicBlocks=" + ["true", "false"][1] + " --FlattenDispatch=" + ["switch", "goto", "indirect"][random.randint(0,2)] + " --FlattenConditionalKinds=" + ["branch", "compute", "flag"][random.randint(0,2)],
#        "merge" : " --Transform=Merge --Functions=" + first_function + "," + second_function,
        "opaque" : opaque_chain,  # maybe for more complex opaque predicates additional helper functions and structures are needed?
        "randomize_arguments" : " --Transform=RndArgs  --RndArgsBogusNo=" + str([random.randint(1,5), random.randint(int(parameter_count*0.5), parameter_count)][parameter_count > 0]) + " --Functions=" + target_function,
#        "split" : " --Transform=Split --Functions=" + target_function,
#        "encode_literals" : " --Transform=InitOpaque --Functions=" + target_function + " --Transform=EncodeLiterals --Functions=" + target_function, # needs opaque expressions for computing function addresses bz default, even if all additional settings are set to false tigress still requires InitOpaque
#        "anti_alias_analysis" : " --Transform=InitOpaque --Functions=" + target_function + " --Transform=AntiAliasAnalysis --Functions=" + target_function,
#        "inline" : " --Transform=Inline --Functions=" + target_function # Inlining two custom functions into each other is not possible? Only inlining one or more custom functions into main?
#        "virtualize" : " --Transform=Virtualize --Functions=" + target_function # later also region level obfuscation for the training dataset to increase diversity? What about self modifying virtualization?
    }
    
    seed = 0 # zero means tigress randomizes the output
    cmd = "tigress --Seed=0 --Statistics=0 --Verbosity=0 --Environment=x86_64:Linux:Clang:14.0.0 --Transform=InitEntropy --Functions=init_tigress --InitEntropyKinds=vars"

    if "encode_branches" in t_chain:
        cmd += " --Transform=InitBranchFuns --InitBranchFunsCount=1"

    if "opaque" in t_chain:
        cmd += " --Transform=InitOpaque --InitOpaqueStructs=" + opaque_choice + " --Functions=init_tigress"

    full_cmd = cmd + " datasets/original/" + filename + " --out=" + "datasets/original/" + filename.removesuffix(".c") + "_tigress_canonicalized.c"


    #os.system(full_cmd)
    try:
        tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=10)
        print(tigress_out.stdout)
        print(tigress_out.stderr)

    except subprocess.TimeoutExpired:
        return -1

    
    if not "randomize_arguments" in t_chain:
        chain = ""

        for t in t_chain:
            chain += command_templates[t] + " "

        chain += command_templates["basic"]

        full_cmd = cmd + chain
        # add basic at the end of every sample
#       full_cmd = cmd + command_templates[k] + " original/" + filename + " --out=" + "obfuscated/" + filename.removesuffix(".c") + "_" + k + "_cleanup.c"
        if not "encode_branches" in t_chain:
            full_cmd += " --Transform=SoftwareMetrics --SoftwareMetricsKind=* --Functions=* --SoftwareMetricsJsonFileName=" + "datasets/obfuscated/" + filename.removesuffix(".c") + "_chain" + "_metrics.json"
        full_cmd += " datasets/original/" + filename + " --out=" + "datasets/obfuscated/" + filename.removesuffix(".c") + "_chain.c"
        print(full_cmd)
        #  os.system(full_cmd)
        try:
            tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=10)
            print(tigress_out.stdout)
            print(tigress_out.stderr)

        except subprocess.TimeoutExpired:
            return -1
    else:
        pivot = t_chain.index('randomize_arguments') + 1
#            pre_rnd, post_rnd = t_chain[:pivot], t_chain[pivot:]
        pre_rnd = t_chain
        post_rnd = []

#            print(pre_rnd, post_rnd)
#            exit()

        chain = ""
        for t in pre_rnd:
            chain += command_templates[t] + " "

        chain += command_templates["basic"]
        full_cmd = cmd + chain
#       full_cmd = cmd + command_templates[k] + " original/" + filename + " --out=" + "obfuscated/" + filename.removesuffix(".c") + "_" + k + "_cleanup.c"
        
        #full_cmd += " datasets/original/" + filename + " --out=" + "datasets/obfuscated/" + filename.removesuffix(".c") + "_chain_randomize_arguments_intermediate.c"
        full_cmd += " datasets/original/" + filename + " --out=" + "datasets/obfuscated/" + filename.removesuffix(".c") + "_chain.c"
        print(full_cmd)
        #  os.system(full_cmd)
        try:
            tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=10)
            print(tigress_out.stdout)
            print(tigress_out.stderr)

        except subprocess.TimeoutExpired:
            return -1

        """chain = ""

        function_name, intermediate_program = extract_function("datasets/obfuscated/" + filename.removesuffix(".c") + "_chain_randomize_arguments_intermediate.c", target_function, "", False, "opaque" in t_chain, "encode_branches" in t_chain)

        with open("datasets/obfuscated/" + filename.removesuffix(".c") + "_chain_randomize_arguments_intermediate_function.c", "w") as f:
            f.write(tigress_header + sample['real_deps'].replace("# 1", "") + "\n" + intermediate_program + "\nint main(){}\n")

        for t in post_rnd:
            chain += command_templates[t] + " "

#          chain += command_templates["basic"]
        cmd2 = cmd + chain

        if not "encode_branches" in t_chain:
            cmd2 += " --Transform=SoftwareMetrics --SoftwareMetricsKind=* --Functions=* --SoftwareMetricsJsonFileName=" + "datasets/obfuscated/" + filename.removesuffix(".c") + "_chain_metrics.json"

        cmd2 += " datasets/obfuscated/" + filename.removesuffix(".c") + "_chain_randomize_arguments_intermediate_function.c" + " --out=" + "datasets/obfuscated/" + filename.removesuffix(".c") + "_chain.c"
        print(cmd2)
        os.system(cmd2)"""

    return 0




@timeit
def main():
    data = load_dataset("jordiae/exebench", split='train_real_compilable')
    data = data.shuffle(42)
    max_file_name_length = 256-len(os.getcwd())-35 # -35 to account for long appendices like _chain_partially_rnd_with_helpers.c
    raw_deobf_dataset = ""
    json_deobf_dataset = [] # use instruction and output
    json_deobf_dataset2 = []
    current_dataset_length = 0
    raw_deobf_datasets = {}
    json_deobf_datasets = {}
    json_deobf2_datasets = {}
    too_longs = 0
    random.seed(42)

    training_samples = args.number_of_samples
    

    names = []
    mapping_table = {}
    choice = "N"
    total_samples = 0

    """if os.path.exists("datasets/training_progress.json"):
        choice = input("Training data in progress found. Resume (Y/N)?")

        if choice == "Y":
            with open("datasets/training_progress.json", "r") as f:
                training_progress = json.load(f)

            with open("datasets/obfuscation_dataset_chain_training_temp.txt", "r") as f:
                raw_deobf_dataset = f.read()

            json_deobf_dataset2 = load_jsonl_dataset("datasets/obfuscation_dataset_chain2_training_temp.json")
            names = training_progress['names']
            training_samples = training_progress['training_samples']
            temp_i = training_progress['i']
            total_samples = training_progress['total_samples']"""

    start_time = time.time()
    backup_directory("original")
    backup_directory("obfuscated")
    duplicate_string = ""

    for i, sample in enumerate(data):
    #    if choice == "Y" and i < temp_i:
    #        continue

        if total_samples >= args.number_of_samples:
            break

        status_array = check_sample_preconditions(sample, names)

        if status_array != [0, 0, 0, 0, 0, 0, 0]:
            training_samples += 1
            continue

        with open("datasets/original/" + sample['fname'][:max_file_name_length] + "_" + str(args.chain_length) + duplicate_string + ".c", "w") as f:
            f.write(normalize_data_types(build_program(sample=sample, is_main=sample['fname'] == "main")))

#        print(training_samples)
#        print(i)
#        print(sample['fname'])
        param_count = count_function_parameters(sample['func_def'], sample['fname'])
        t_chain = get_random_chain(args.chain_length, replacement=True)
        obfuscation_success = obfuscate(sample['fname'][:max_file_name_length] + "_" + str(args.chain_length) + duplicate_string + ".c", sample['fname'], param_count, t_chain)

        if obfuscation_success != 0:
            print("Tigress obfuscation timeout exceeded")
            training_samples += 1
            continue

        if not os.path.exists("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + "_" + str(args.chain_length) + duplicate_string + "_chain" + ".c"):
            training_samples += 1
            continue

        # we could randomize the identifier names even further since tigress uses a fixed naming convention after CleanUp for id names but we could also argue that any program that is to be deobfuscated could be conver>
        
        suffix = "_chain.c"
        test = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + "_" + str(args.chain_length) + duplicate_string + suffix, sample['fname'], "", False, "opaque" in t_chain, "encode_branches" in t_chain)

        if type(test) == str:
            #errors += 1
            training_samples += 1
            continue

        # erroneous extractions are skipped
        if test[1] == "":
            print(f"Error during the extraction")
#            errors += 1
#                print(sample['real_deps'])
#                print(sample['func_def'])
#            sample_info[str(i) + "__" + sample['fname']] = "Error while extracting obfuscated sample"
            training_samples += 1
            continue

        target_function = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + "_" + str(args.chain_length) + suffix, sample['fname'], "", False, "opaque" in t_chain, "encode_branches" in t_chain)

    #    if target_function == "":
    #        training_samples += 1
    #        continue

        obfuscated_function_name, obfuscated_function = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + "_" + str(args.chain_length) + duplicate_string + suffix, sample['fname'], "", False, "opaque" in t_chain, "encode_branches" in t_chain, extract_helpers=False)
        obfuscated_function_name, obfuscated_code = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + "_" + str(args.chain_length) + duplicate_string + suffix, sample['fname'], "", False, "opaque" in t_chain, "encode_branches" in t_chain)

#        print("obfuscated/" + sample['fname'] + duplicate_string + suffix)

        original_function = extract_function("datasets/original/" + sample['fname'][:max_file_name_length] + "_" + str(args.chain_length) + duplicate_string + "_tigress_canonicalized.c", sample['fname'], extract_helpers=False)[1]
        
        with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}_{args.chain_length}{duplicate_string}_chain_partially_rnd_with_helpers.c", "w") as f:
            f.write(obfuscated_code)

        # ToDo use exotic placeholder for // Obfuscatedcode // Deobfuscatedcode for both training scripts
        code = "// Obfuscatedcode\n" + remove_comments(obfuscated_function) + "\n// Deobfuscatedcode\n" + remove_comments(original_function) # <|end|> is a makeshift EOS token, will be replaced by the EOS token depending on the chosen tokenizer later
        with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}_{args.chain_length}{duplicate_string}_chain_sample.c", "w") as f:
            f.write(code)

        obfs_ids, labels = get_identifier_names(obfuscated_code, ignore_function_declarations=False)
        obfs_ids_corrected = []

        for obfs_id in obfs_ids:
                #print(not obfs_id in code_without_helper_defs)
                if obfs_id.split("::")[-1] in obfuscated_code: # careful, heuristic might break if struct has the same attr name as e. g., a global var, although id randomization should minimize risk
                    obfs_ids_corrected.append(obfs_id)

        randomized_code_obfs = randomize_identifiers2(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}_{args.chain_length}{duplicate_string}_chain_partially_rnd_with_helpers.c", f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}_{args.chain_length}{duplicate_string}_chain_sample.c", identifier_names=obfs_ids_corrected, labels=labels, ignore_func_decls=False, function_name=sample['fname'])
        randomized_code_obfs = post_process(randomized_code_obfs).replace("__RND__", "") + "<|end|>"
        randomized_code_obfs = randomized_code_obfs.replace("// Obfuscatedcode\n", "// Obfuscated code\n").replace("// Deobfuscatedcode\n", "// Deobfuscated code\n")

        #code = "// Obfuscated code\n" + obfuscated_function + "\n// Deobfuscated code\n" + original_function + "<|end|>" # <|end|> is a makeshift EOS token, will be replaced by the EOS token depending on the chosen tokenizer later

        # also obfuscated-original-pairs that are too large for the model are left out, continue sampling until the desired sample count is achieved without the special cases here
       # if len(tokenizer(randomized_code_obfs)['input_ids']) > args.max_tokens:
       #     print("code is too long")
       #     training_samples += 1
       #     continue

        obfs, orig = randomized_code_obfs.split("// Obfuscated code\n")[1].split("<|end|>")[0].split("\n// Deobfuscated code\n")

        if not check_token_length_limits(obfs, orig, [("deepseek-coder-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct"), ("codellama", "codellama/CodeLlama-7b-hf"), ("gpt-4", "gpt-4")], args.max_tokens):
            print("code is too long")

          #  errors += 1
            too_longs += 1
            continue

        randomized_code_obfs = "<|OBFS|>\n" + obfs + "<|ORIG|>\n" + orig

        if obfuscated_function_name == "":
            training_samples += 1
            continue

        

        total_samples += 1
        print(f"Total samples: {total_samples}")


        mapping_table[sample['fname']] = "__".join(t_chain)
        raw_deobf_dataset += randomized_code_obfs
#        json_deobf_dataset.append({'instruction' : "// Obfuscated code\n" + extract_function("datasets/obfuscated/" + sample['fname'] + duplicate_string + suffix, sample['fname'], "", False, "opaque" in t_chain, "encode_branches" in t_chain) + "\n// Deobfuscated code\n", 'input' : '', 'output' : extract_function("datasets/original/" + sample['fname'] + duplicate_string + "_tigress_canonicalized.c", sample['fname'])[:-1]})
       # json_deobf_dataset.append({'instruction' : "// Obfuscated code\n" + obfuscated_function + "\n Deobfuscated code\n", 'input' : '', 'output' : original_function})
        json_deobf_dataset2.append({str(i) + "__name__" + sample['fname'] : randomized_code_obfs})

        if total_samples % 100 == 0:
            save_dataset_raw(raw_deobf_dataset, f"_{args.chain_length}_chain_{args.max_tokens}_training_temp.c")
          #  save_dataset_json(json_deobf_dataset, "_chain_training.c")
            save_dataset_json(json_deobf_dataset2, f"_{args.chain_length}_chain2_{args.max_tokens}_training_temp.c")

            with open(f"datasets/training_progress_{args.chain_length}_{args.max_tokens}.json", "w") as f:
                json.dump({"i" : i, "training_samples" : training_samples, "mapping_table" : mapping_table, "names" : names, "total_samples" : total_samples}, f)

    save_dataset_raw(raw_deobf_dataset, f"_{args.chain_length}_chain_{args.max_tokens}_training.c")
  #  save_dataset_json(json_deobf_dataset, "_chain_training.c")
    save_dataset_json(json_deobf_dataset2, f"_{args.chain_length}_chain2_{args.max_tokens}_training.c")

    with open(f"datasets/transformations_{args.chain_length}_chain_{args.max_tokens}_training.json", "w") as f:
        json.dump(mapping_table, f)

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time}")

    with open(f"datasets/transformations_{args.chain_length}_chain_{args.max_tokens}_too_long.txt", "w") as f:
        f.write(str(too_longs))

if __name__ == "__main__":
    main()
