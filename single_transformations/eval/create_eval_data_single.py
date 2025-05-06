import argparse
import json
import logging
import os
import random
import re
from shlex import split
import subprocess
import sys
import time

import clang.cindex as cindex
from datasets import load_dataset
from transformers import AutoTokenizer

from utils import (
    backup_directory,
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
    normalize_data_types,
    save_input_samples,
    save_dataset_raw,
    save_dataset_json,
    timeit
)
from utils_eval import check_correctness2

# define parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', help='The tokenizer to use for the sample filtering')
parser.add_argument('--max_tokens', type=int)
parser.add_argument('--number_of_samples', type=int)
args = parser.parse_args()

#if len(sys.argv) < 2:
#    suffix = "eval_ext"
#else:
#    suffix = sys.argv[1]

# Sample: python3 create_eval_data_single.py --tokenizer deepseek-ai/deepseek-coder-6.7b-instruct --max_tokens 2048 --number_of_samples 200
suffix = "eval"

cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-14.so.1")  # Set the path to libclang.so
#tokenizer = AutoTokenizer.from_pretrained("codegen25-7b-multi-full", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

syntax_broken = []
semantics_broken = []
runtime_broken = []
output_broken = []
output_json_broken = []


# Configure the logging system
logging.basicConfig(filename=f'datasets/eval_data_poly_{suffix}_eval_err.log', level=logging.ERROR,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
        
def obfuscate(filename, target_function, target_function2, parameter_count, sample):
    tigress_header = "#include \"/usr/local/bin/tigresspkg/3.3.3/tigress.h\"\n"

    opaque_choice = ",".join(random.sample(["list", "array", "env"], random.randint(1,3)))
    opaque_choice2 = random.sample(["call", "bug", "true", "junk", "fake_call", "question"], 1)

    opaque_chain = ""

    for choice in opaque_choice2:
        opaque_chain += " --Transform=AddOpaque --Functions=" + target_function + " --AddOpaqueStructs=" + opaque_choice + " --AddOpaqueKinds=" + choice
        
#        if choice == "question":
#            opaque_chain += " --Transform=Inline --Functions=/.*QUESTION.*/"

    if "question" in opaque_choice2:
        opaque_chain += " --Transform=Inline --Functions=/.*QUESTION.*/"

    command_templates = {
        "basic" : " --Transform=CleanUp --CleanUpKinds=names,annotations",
        "encode_arithmetic" : " --Transform=EncodeArithmetic --Functions=" + target_function, #+ " --Transform=CleanUp --CleanUpKinds=names,annotations",
#        "encode_data" : " --Transform=EncodeData --LocalVariables=" + target_function + ":" + target_local_variables + " --EncodeDataCodecs=poly1 --Transform=CleanUp --CleanUpKinds=names,annotations",
        "flatten" : " --Transform=Flatten --Functions=" + target_function + " --FlattenRandomizeBlocks=" + ["true", "false"][random.randint(0,1)] + " --FlattenSplitBasicBlocks=" + ["true", "false"][1] +  " --FlattenDispatch=" + ["switch", "goto", "indirect"][random.randint(0,2)] + " --FlattenConditionalKinds=" + ["branch", "compute", "flag"][random.randint(0,2)], #+ " --Transform=CleanUp  --CleanUpKinds=names,annotations",
#        "merge" : " --Transform=Merge --Functions=" + first_function + "," + second_function + " --Transform=CleanUp --CleanUpKinds=names,annotations",
        "opaque" : " --Transform=InitOpaque --InitOpaqueStructs=" + opaque_choice + " --Functions=init_tigress" + opaque_chain , #+ " --Transform=CleanUp --CleanUpKinds=names,annotations",  # maybe for more complex opaque predicates additional helper functions and structures are needed?
        "randomize_arguments" : " --Transform=RndArgs  --RndArgsBogusNo=" + str([random.randint(1,5), random.randint(int(parameter_count*0.5), parameter_count)][parameter_count > 0])  + " --Functions=" + target_function, #+ " --Transform=CleanUp --CleanUpKinds=names,annotations",
#        "split" : " --Transform=Split --Functions=" + target_function + " --Transform=CleanUp --CleanUpKinds=names,annotations",
#        "encode_literals" : " --Transform=InitOpaque --Functions=" + target_function + " --Transform=EncodeLiterals --Functions=" + target_function + " --Transform=CleanUp --CleanUpKinds=names,annotations", # needs opaque expressions for computing function addresses bz default, even if all additional settings are set to false tigress still requires InitOpaque
        "encode_branches" : " --Transform=InitBranchFuns --InitBranchFunsCount=1 --Transform=AntiBranchAnalysis --Functions=" + target_function + " --AntiBranchAnalysisKinds=branchFuns --AntiBranchAnalysisObfuscateBranchFunCall=false --AntiBranchAnalysisBranchFunFlatten=true" #+ " --Transform=CleanUp --CleanUpKinds=names,annotations",
#        "anti_alias_analysis" : " --Transform=InitOpaque --Functions=" + target_function + " --Transform=AntiAliasAnalysis --Functions=" + target_function + " --Transform=CleanUp --CleanUpKinds=names,annotations",
#        "inline" : " --Transform=Inline --Functions=" + target_function + " --Transform=CleanUp --CleanUpKinds=names,annotations" # Inlining two custom functions into each other is not possible? Only inlining one or more custom functions into main?
#        "virtualize" : " --Transform=Virtualize --Functions=" + target_function + " --Transform=CleanUp --CleanUpKinds=names,annotations"# later also region level obfuscation for the training dataset to increase diversity? What about self modifying virtualization?
    }
    
    seed = 0 # zero means tigress randomizes the output
    cmd = "tigress --Seed=0 --Statistics=0 --Verbosity=0 --Environment=x86_64:Linux:Clang:14.0.0 --Transform=InitEntropy --Functions=init_tigress --InitEntropyKinds=vars"
    

    full_cmd = cmd + " datasets/original_eval/" + filename + " --out=" + "datasets/original_eval/" + filename.removesuffix(".c") + "_tigress_canonicalized.c"
    #  os.system(full_cmd)
    try:
        tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=100)

    except subprocess.TimeoutExpired:
        return -1

    try:
        function_name, canonicalized_program = extract_function("datasets/original_eval/" + filename.removesuffix(".c") + "_tigress_canonicalized.c", target_function, "", False, False, False)

    except ValueError:
        print("Failed to extract canonicalized program")
        return -1

#        with open("original_eval/" + filename.removesuffix(".c") + "_tigress_canonicalized_function.c", "w") as f:
#            f.write(normalize_data_types(build_program(sample, True, False, True, extract_function("original_eval/" + filename.removesuffix(".c") + "_tigress_canonicalized.c", filename.removesuffix(".c"), extract_helpers=False)[1])))
#            f.write(normalize_data_types(build_program(sample=sample, is_main=sample['fname'] == "main")))
    
    with open("datasets/original_eval/" + filename.removesuffix(".c") + "_tigress_canonicalized_function.c", "w") as f:
        f.write(tigress_header + sample['real_deps'].replace("# 1", "") + "\n" + normalize_data_types(canonicalized_program) + "\nint main(){}\n")


    for k in command_templates.keys():
#            full_cmd = cmd + command_templates[k] + " original/" + filename + " --out=" + "obfuscated/" + filename.removesuffix(".c") + "_" + k + "_cleanup.c"
        if k != "randomize_arguments":
            full_cmd = cmd + command_templates[k]
            
#            if k != "encode_branches":
#                full_cmd += " --Transform=SoftwareMetrics --SoftwareMetricsKind=* --Functions=* --SoftwareMetricsJsonFileName=" + filename.removesuffix(".c") + "_metrics.json"

            if k != "basic":
                full_cmd += command_templates['basic']

            full_cmd += " datasets/original_eval/" + filename + " --out=" + "datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + k + ".c"
#                print(full_cmd)
#                full_cmd += " original_eval/" + filename.removesuffix(".c") + "_tigress_canonicalized_function.c" + " --out=" + "obfuscated_eval/" + filename.removesuffix(".c") + "_" + k + ".c"
#                os.system(full_cmd)
            
            try:
                tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=100)

            except subprocess.TimeoutExpired:
                return -1

        else:
            full_cmd = cmd + command_templates[k]
            full_cmd += " datasets/original_eval/" + filename + " --out=" + "datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + k + "_intermediate.c"
#                full_cmd += " original_eval/" + filename.removesuffix(".c") + "_tigress_canonicalized_function.c" + " --out=" + "obfuscated_eval/" + filename.removesuffix(".c") + "_" + k + "_intermediate.c"

#                print(full_cmd)
#                os.system(full_cmd)
            try:
                tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=100)

            except subprocess.TimeoutExpired:
                return -1

            full_cmd = cmd + command_templates['basic'] + " datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + k + "_intermediate.c" + " --out=" + "datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + k + ".c"
#                os.system(full_cmd)
            try:
                tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=100)

            except subprocess.TimeoutExpired:
                return -1

    return 0
#            print(full_cmd)
#            os.system(full_cmd)

@timeit
def main():
    os.system("echo \"\" > gcc.log")
    os.system("echo \"\" > gcc_io.log")
    random_seed = 42
    data = load_dataset("jordiae/exebench", split='test_real')
    data = data.shuffle(random_seed)
    training_samples = args.number_of_samples
    random.seed(random_seed)
    create_folder_if_not_exists("datasets/original_eval")
    create_folder_if_not_exists("datasets/obfuscated_eval")
    create_folder_if_not_exists("datasets/input_samples")
    create_folder_if_not_exists("datasets/original_io_test")
    create_folder_if_not_exists("datasets/obfuscated_io_test")
    sample_info = {}
#    data = load_from_disk("/media/superuser/Lokaler Datenträger/Datasets/ExebenchTrainReal")
    
    raw_deobf_dataset = ""
    json_deobf_dataset = [] # use instruction and output
    raw_deobf_datasets = {}
    json_deobf_datasets = {}
    json_deobf2_datasets = {}
    backup_directory("original_eval")
    backup_directory("obfuscated_eval")
    duplicate_string = ""
    transformations = ["encode_arithmetic", "encode_branches", "flatten", "opaque", "randomize_arguments"]
    too_longs = 0
#    transformations = ["opaque"]
    names = []
    for t in transformations:
        raw_deobf_datasets[t] = ""
        json_deobf_datasets[t] = []
        json_deobf2_datasets[t] = []

    start_time = time.time()
    for i, sample in enumerate(data):
        if i >= training_samples:
            break

        status_array = check_sample_preconditions(sample, names)

        if status_array != [0, 0, 0, 0, 0, 0, 0]:
            training_samples += 1
            continue

        max_file_name_length = 256-len(os.getcwd())-2 # the -2 is because of the .c ending, theoretically it could break later when obfuscating due to the transformation suffix but when the obfuscate fails the sample will be skipped anyways so this case is covered too
        print(max_file_name_length)
        # we can identify unique long names but on disk it might be overwritten, but since we only need the dataset this is ok
        with open("datasets/original_eval/" + sample['fname'][:max_file_name_length] + duplicate_string + ".c", "w") as f:
            f.write(normalize_data_types(build_program(sample=sample, is_main=sample['fname'] == "main")))

#        os.system("g++ -I lib original_io_test/" + sample['fname'][:max_file_name_length] + ".cpp -o " + "original_io_test/" + sample['fname'][:max_file_name_length] + ".exe")

#        if not os.path.exists("original_io_test/" + sample['fname'][:max_file_name_length] + ".exe"):
#            training_samples += 1
#            continue

#        is_correct = check_correctness(sample['fname'][:max_file_name_length], sample['fname'][:max_file_name_length], "original_io_test/")

#        if not type(is_correct) == int or is_correct == 0:
#            training_samples += 1
#            continue

        print(training_samples)
        print(i)
        print("current length " + str(len(raw_deobf_datasets[t].split("<|end|>"))))
#        input()

#        print(sample['fname'])
        param_count = count_function_parameters(sample['func_def'], sample['fname'])
        obfuscation_status = obfuscate(sample['fname'][:max_file_name_length] + duplicate_string + ".c", sample['fname'], "", param_count, sample)

        if obfuscation_status != 0:
            print("Obfuscation error")
            training_samples += 1
            continue

        errors = 0
        raw_codes = {}
        json_codes = {}
        json_codes2 = {}
        print("Sample name: ", sample['fname'])

        with open("datasets/original_io_test/" + sample['fname'][:max_file_name_length] + ".cpp", "w") as f:
            f.write(build_program(sample=sample, empty_main=False))

        save_input_samples(sample, "", max_file_name_length)

        #is_correct = check_correctness(sample['fname'][:max_file_name_length], sample['fname'][:max_file_name_length], "original_io_test/")
        #print(is_correct)

        #if not type(is_correct) == int or is_correct == 0:
        #    print("Sample broke during original evaluation", is_correct)
        #    training_samples += 1
        #    continue
        func_canonicalized = extract_function(sample['fname'] + "_tigress_canonicalized", sample['fname'])
        io_wrapper_canonicalized = build_program(sample=sample, empty_main=False, is_main=False, func_def_is_external=True, func_def=func_canonicalized)

        with open("datasets/original_io_test/" + sample['fname'][:max_file_name_length] + "_tigress_canonicalized.cpp", "w") as f:
            f.write(io_wrapper_canonicalized)

        for t in transformations:
            # we could randomize the identifier names even further since tigress uses a fixed naming convention after CleanUp for id names but we could also argue that any program that is to be deobfuscated could be converted into this form and therefore model generalization with this respect might not be as important
            suffix = "_" + t + ".c"
            test = extract_function("datasets/obfuscated_eval/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'], "", False, t == "opaque", t == "encode_branches")

            if type(test) == str:
                errors += 1
                continue

            # erroneous extractions are skipped
            if test[1] == "":
                print(f"transformation {t}: Error during the extraction")
                errors += 1
                sample_info[str(i) + "__" + sample['fname']] = "Error while extracting obfuscated sample"
                continue

            obfuscated_function_name, obfuscated_function = extract_function("datasets/obfuscated_eval/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'], "", False, t == "opaque", t == "encode_branches", extract_helpers=False)
            obfuscated_function_name, obfuscated_code = extract_function("datasets/obfuscated_eval/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'], "", False, t == "opaque", t == "encode_branches")
            code = "// Obfuscated code\n" + obfuscated_function + "\n// Deobfuscated code\n" + extract_function("datasets/original_eval/" + sample['fname'][:max_file_name_length] + duplicate_string + "_tigress_canonicalized.c", sample['fname'], extract_helpers=False)[1] + "<|end|>" # <|end|> is a makeshift EOS token, will be replaced by the EOS token depending on the chosen tokenizer later

            obfs, orig = code.split("// Obfuscated code\n")[1].split("<|end|>")[0].split("\n// Deobfuscated code\n")

            # also obfuscated-original-pairs that are too large for the model are left out, continue sampling until the desired sample count is achieved without the special cases here
            #if len(tokenizer(code)['input_ids']) > args.max_tokens:
            #    print("code is too long")
            #    errors += 1
            #    sample_info[str(i) + "__" + sample['fname']] = "Code is too long"
            #    continue

            if not check_token_length_limits(obfs, orig, [("deepseek-coder-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct"), ("codellama", "codellama/CodeLlama-7b-hf"), ("gpt-4", "gpt-4")], args.max_tokens):
#            if len(randomized_code_obfs_tokenized) > args.max_tokens:
                print("code is too long")
                errors += 1
                sample_info[str(i) + "__" + sample['fname']] = "Code is too long"
                too_longs += 1
                continue

            if obfuscated_function_name == "":
                errors += 1
                continue

            if t == "randomize_arguments":
                if not sample['fname'] in sample['real_exe_wrapper']: # obfuscated_program before
                    errors += 1
                    print("Function name not found")
                    sample_info[str(i) + "__" + sample['fname']] = "Function name not found"
                    continue

                wrapper = sample['real_exe_wrapper']

                if not re.search(sample['fname'] + r"\s*\([^)]*\)", sample['real_exe_wrapper']):
                    errors += 1
                    sample_info[str(i) + "__" + sample['fname']] = "Function head not found"
                    continue

                function_arguments = re.search(sample['fname'] + r"\([^)]*\)", sample['real_exe_wrapper']).group().removeprefix(sample['fname'] + "(")[:-1].split(",")
                param_perm = extract_parameter_permutations(sample['func_def'], extract_function("datasets/obfuscated_eval/" + sample['fname'] + "_randomize_arguments_intermediate.c", sample['fname'])[1], sample['fname'])
                new_params, additional_vars = build_bogus_parameters(function_arguments, param_perm)
                new_function_call = sample['fname'] + "(" + ",".join(new_params) + ")"
                
                wrapper = re.sub(sample['fname'] + r"\s*\([^)]*\)", new_function_call, wrapper)

                if additional_vars != "":
                    main_head = re.search("main\s*\([^)]*\)\s*{", wrapper).group()
                    wrapper = re.sub("main\s*\([^)]*\)\s*{", main_head + "\n" + additional_vars, wrapper)

                obfuscated_program = build_program(sample=sample, empty_main=False, is_main=False, func_def_is_external=True, func_def=obfuscated_code, modified_wrapper=wrapper)

            else:
                obfuscated_program = build_program(sample=sample, empty_main=False, is_main=False, func_def_is_external=True, func_def=obfuscated_code)
#
            obfuscated_program = re.sub(sample['fname'] + r"\s*\(", obfuscated_function_name + "(", obfuscated_program)

            with open("datasets/obfuscated_eval/" + sample['fname'][:max_file_name_length] + duplicate_string + "_" + t + "_function.c", "w") as f:
                f.write(obfuscated_code)

            with open("datasets/obfuscated_io_test/" + sample['fname'][:max_file_name_length] + "_" + t + ".cpp", "w") as f:
                f.write(obfuscated_program)

            # use tigress to canonicalize the original code and use it as the deobfuscated version instead of the original one to ease the mapping between deobfuscated and obfuscated code, maybe a dataset version with the original functions is also interesting to see if the LLMs are still able to learn the mapping or if this becomes significantly harder
            
            code = "<|OBFS|>\n" + obfs + "<|ORIG|>\n" + orig
            raw_codes[t] = code
            json_codes[t] = {'instruction' : "// Obfuscated code\n" + extract_function("datasets/obfuscated_eval/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'], "", False, t == "opaque", t == "encode_branches", extract_helpers=False)[1] + "\n// Deobfuscated code\n", 'input' : '', 'output' : extract_function("datasets/original_eval/" + sample['fname'][:max_file_name_length] + duplicate_string + "_tigress_canonicalized.c", sample['fname'], extract_helpers=False)[1][:-1]}
            json_codes2[t] = {str(i) + "__name__" + sample['fname'] : code}

        # throw the sample away if for any transformation any error occurred to make sure that the difference between the datasets is only the transformation and not the function, this way the incorporated functions are synced among the datasets with the different transformations
    
            sample_info[str(i) + "__" + sample['fname']] = "Sample taken"
            create_folder_if_not_exists("datasets/input_samples/" + sample['fname'])

            with open("datasets/original_io_test/" + sample['fname'][:max_file_name_length] + ".cpp", "w") as f:
                f.write(build_program(sample=sample, empty_main=False))

            with open("datasets/input_samples/" + sample['fname'][:max_file_name_length] + "_io_samples.json", "w") as f:

                f.write(json.dumps(sample['real_io_pairs']) + "\n")

            for _,line in enumerate(sample['real_io_pairs']['input']):

                input_sample = dict()

                for var, value in zip(line['var'], line['value']):
                    input_sample[var] = eval(value)

                with open("datasets/input_samples/" + sample['fname'][:max_file_name_length] + "/input_" + str(_) + ".json", "w") as f:
                    f.write(json.dumps(input_sample) + "\n")

            is_correct = check_correctness2(sample['fname'][:max_file_name_length], sample['fname'][:max_file_name_length] + "_" + t, "datasets/original_io_test/", "datasets/obfuscated_io_test/")
            print(is_correct)

            if not type(is_correct) == int or is_correct == 0:
                print("Sample broke during obfuscation evaluation", is_correct)

                errors += 1
                continue

        if errors == 0:
            for t in transformations:
                os.system("gcc -w -fsyntax-only obfuscated_eval/" + sample['fname'][:max_file_name_length] + duplicate_string + "_" + t + "_function.c 2>> gcc.log")
                raw_deobf_datasets[t] += raw_codes[t]
                json_deobf_datasets[t].append(json_codes[t])
                json_deobf2_datasets[t].append(json_codes2[t])
                print("current length " + str(len(raw_deobf_datasets[t].split("<|OBFS|>"))))

            print("Sample added")
            print(time.time() - start_time)
            start_time = time.time()
            names.append(sample['fname'])

        else:
            training_samples += 1

    with open(f"sample_information_eval_single_{args.max_tokens}.json", "w") as f:
        json.dump(sample_info, f)

    for t in transformations:
        suffix1 = "_" + t + "_" + str(args.max_tokens) + "_eval.c"
        suffix2 = "_" + t + "_" + str(args.max_tokens) + "_eval.c"
        save_dataset_raw(raw_deobf_datasets[t], suffix1)
        save_dataset_json(json_deobf_datasets[t], suffix1)
        save_dataset_json(json_deobf2_datasets[t], suffix2)

    with open(f"datasets/syntax_{args.max_tokens}_broken.txt", "w") as f:
        f.write("\n".join(syntax_broken))

    with open(f"datasets/semantics_broken_{args.max_tokens}.txt", "w") as f:
        f.write("\n".join(semantics_broken))

    with open(f"datasets/runtime_broken_{args.max_tokens}.txt", "w") as f:
        f.write("\n".join(runtime_broken))

    with open(f"datasets/output_broken_{args.max_tokens}.txt", "w") as f:
        f.write("\n".join(output_broken))

    with open(f"datasets/output_json_broken_{args.max_tokens}.txt", "w") as f:
        f.write("\n".join(output_json_broken))

if __name__ == "__main__":
    main()
