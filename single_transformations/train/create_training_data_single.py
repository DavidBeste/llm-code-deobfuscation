import argparse
import os
import pickle
import random
import subprocess

import clang.cindex as cindex
from datasets import load_dataset
from transformers import AutoTokenizer

from randomize_idns import (
    get_identifier_names, 
    post_process, 
    randomize_identifiers, 
    randomize_identifiers2
)
from utils import (
    backup_directory,
    build_program,
    check_sample_preconditions,
    check_token_length_limits,
    count_function_parameters, 
    extract_function2 as extract_function, 
    format_obf_org_pair,
    normalize_data_types,
    remove_comments,
    save_dataset_json,
    save_dataset_raw
)
cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-14.so.1")  # Set the path to libclang.so

# define parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', help='The tokenizer to use for the sample filtering', required=True)
parser.add_argument('--max_tokens', type=int, required=True)
parser.add_argument('--number_of_samples', type=int, required=True)
args = parser.parse_args()
# Sample: python3 create_training_data_single.py --tokenizer deepseek-ai/deepseek-coder-6.7b-instruct --max_tokens 2048 --number_of_samples 3000

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

def obfuscate(filename, target_function, target_function2, parameter_count, sample):
    opaque_choice = ",".join(random.sample(["list", "array", "env"], random.randint(1,3)))
    opaque_choice2 = random.sample(["call", "bug", "true", "junk", "fake_call", "question"], random.randint(1,6))
    opaque_chain = ""

    for choice in opaque_choice2:
        opaque_chain += " --Transform=AddOpaque --Functions=" + target_function + " --AddOpaqueStructs=" + opaque_choice + " --AddOpaqueKinds=" + choice
        
#        if choice == "question":
#            opaque_chain += " --Transform=Inline --Functions=/.*QUESTION.*/"

    if "question" in opaque_choice2:
        opaque_chain += " --Transform=Inline --Functions=/.*QUESTION.*/"

    command_templates = {
#        "basic" : " --Transform=CleanUp --CleanUpKinds=names,annotations",

        "encode_arithmetic" : " --Transform=EncodeArithmetic --Functions=" + target_function, #+ " --Transform=CleanUp --CleanUpKinds=names,annotations",
#        "encode_data" : " --Transform=EncodeData --LocalVariables=" + target_function + ":" + target_local_variables + " --EncodeDataCodecs=poly1 --Transform=CleanUp --CleanUpKinds=names,annotations",
        "flatten" : " --Transform=Flatten --Functions=" + target_function + " --FlattenRandomizeBlocks=" + ["true", "false"][random.randint(0,1)] + " --FlattenSplitBasicBlocks=" + ["true", "false"][1] +  " --FlattenDispatch=" + ["switch", "goto", "indirect"][random.randint(0,2)] + " --FlattenConditionalKinds=" + ["branch", "compute", "flag"][random.randint(0,2)], #+ " --Transform=CleanUp  --CleanUpKinds=names,annotations",
#        "merge" : " --Transform=Merge --Functions=" + first_function + "," + second_function + " --Transform=CleanUp --CleanUpKinds=names,annotations",
        "opaque" : " --Transform=InitOpaque --InitOpaqueStructs=" + opaque_choice + " --Functions=init_tigress" + opaque_chain , #+ " --Transform=CleanUp --CleanUpKinds=names,annotations",  # maybe for more complex opaque predicates additional helper functions and structures are needed?
        # will sometimes create trivial obfuscation when parameter count == 1 and random calcs zero, maybe leave it in to let the model learn from a few samples when it is already deobfuscated
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
    
    #randomized_code = randomize_identifiers(f"datasets/original/{filename}")
    #print(randomized_code)
    #with open(f"datasets/original/{filename.removesuffix('.c')}_rnd.c", "w") as f:
    #    f.write(randomized_code)


    full_cmd = cmd + " datasets/original/" + filename + " --out=datasets/original/" + filename.removesuffix(".c") + "_tigress_canonicalized.c"
    p = subprocess.Popen(full_cmd.split(" "))

    try:
        p.wait(8)

    except subprocess.TimeoutExpired:
        p.kill()
        return -1

    try:
        _, canonicalized_function = extract_function("datasets/original/" + filename.removesuffix(".c") + "_tigress_canonicalized.c", target_function, extract_helpers=False)
    except:
        print("Failed to extract canonicalize")
        return -1

#        print(canonicalized_function + "\nint main() {}")
    #with open("datasets/original/" + filename.removesuffix(".c") + "_tigress_canonicalized_function.c", "w") as f:
    #    f.write(sample['real_deps'].replace("# 1", "") + ["#include <stdlib.h>\n", ""]["#include <stdlib.h>\n" in canonicalized_function] + canonicalized_function + "\nint init_tigress() {}\nint main() {}")


    with open("datasets/original/" + filename.removesuffix(".c") + "_tigress_canonicalized_function.c", "w") as f:
        f.write(canonicalized_function)

    randomized_code = randomize_identifiers(f"datasets/original/{filename.removesuffix('.c')}_tigress_canonicalized_function.c")

    with open(f"datasets/original/{filename.removesuffix('.c')}_rnd.c", "w") as f:
        f.write(sample['real_deps'].replace("# 1", "") + ["#include <stdlib.h>\n", ""]["#include <stdlib.h>\n" in canonicalized_function] + randomized_code + "\nint init_tigress() {}\nint main() {}")

#        with open(f"datasets/original/{filename.removesuffix('.c')}_rnd_function.c", "w") as f:
#            f.write(randomized_code.removesuffix("\nint init_tigress() {}\nint main() {}").replace(sample['real_deps'].replace("# 1", ""), ""))

    with open(f"datasets/original/{filename.removesuffix('.c')}_rnd_function.c", "w") as f:
        f.write(randomized_code)

    for k in command_templates.keys():
        full_cmd = cmd + command_templates[k] + " --Transform=CleanUp --CleanUpKinds=annotations"
#            full_cmd = cmd + get_single_command(k, target_function, opaque_chain, parameter_count) + " --Transform=CleanUp --CleanUpKinds=annotations"
        full_cmd += " datasets/original/" + filename.removesuffix(".c") + "_tigress_canonicalized.c --out=" + "datasets/obfuscated/" + filename.removesuffix(".c") + "_" + k + ".c"

        p = subprocess.Popen(full_cmd.split(" "))

        try:
            p.wait(8)

        except subprocess.TimeoutExpired:
            p.kill()
            return -1

    return 0

def build_sample_from_obfuscated(sample, t, max_file_name_length, duplicate_string, suffix):

    test = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'], "", False, t == "opaque", t == "encode_branches")
    errors = 0

    if type(test) == str() and test == "":
        print(f"transformation {t}: Error during the extraction")
        errors += 1
        return -1

    try:
        with open(f"datasets/original/{sample['fname'][:max_file_name_length].removesuffix('.c')}_rnd_function.c", "r") as f:
            randomized_code = f.read()
    except:
        errors += 1
        return -2

    try:
        _, code_without_helper_defs = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'],"", False, t == "opaque", t == "encode_branches", extract_helpers=False)
    except:
        errors += 1
        print("Fail to extract without helper funcs")
        return -3

    with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_partially_rnd.c", "w") as f:
        f.write(code_without_helper_defs)

    try:
        _, code_with_helper_defs = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'],"", False, t == "opaque", t == "encode_branches", extract_helpers=True)
    except:
        errors += 1
        print("Fail to extract with helper funcs")
        return -4

    with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_partially_rnd_with_helpers.c", "w") as f:
        f.write(code_with_helper_defs)

    obfs_ids, labels = get_identifier_names(code_with_helper_defs, ignore_function_declarations=False)
    obfs_ids_corrected = []

    for obfs_id in obfs_ids:
        if obfs_id.split("::")[-1] in code_without_helper_defs: # careful, heuristic might break if struct has the same attr name as e. g., a global var, although id randomization should minimize risk
            obfs_ids_corrected.append(obfs_id)

    with open("datasets/original/" + sample['fname'][:max_file_name_length] + "_tigress_canonicalized_function.c", "r") as f:
        canonicalized_function = f.read()

    if canonicalized_function == "":
        errors += 1
        return -5

    if code_without_helper_defs == "":
        errors += 1
        return -6

    code = "// Obfuscatedcode\n" + remove_comments(code_without_helper_defs) + "\n// Deobfuscatedcode\n" + remove_comments(canonicalized_function)

    with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_sample.c", "w") as f:
        f.write(code)

    randomized_code_obfs = randomize_identifiers2(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_partially_rnd_with_helpers.c", f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_sample.c", identifier_names=obfs_ids_corrected, labels=labels, ignore_func_decls=False)
    randomized_code_obfs = post_process(randomized_code_obfs).replace("__RND__", "") + "<|end|>"
    randomized_code_obfs = randomized_code_obfs.replace("// Obfuscatedcode\n", "// Obfuscated code\n").replace("// Deobfuscatedcode\n", "// Deobfuscated code\n")
    obfs, orig = randomized_code_obfs.split("// Obfuscated code\n")[1].split("<|end|>")[0].split("\n// Deobfuscated code\n")
    # also obfuscated-original-pairs that are too large for the model are left out, continue sampling until the desired sample count is achieved without the special cases here
    if not check_token_length_limits(obfs, orig, [("deepseek-coder-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct"), ("codellama", "codellama/CodeLlama-7b-hf"), ("gpt-4", "gpt-4")], args.max_tokens):
        print("code is too long")
        errors += 1
        return -7

    randomized_code_obfs = "<|OBFS|>\n" + obfs + "<|ORIG|>\n" + orig

    return randomized_code_obfs
        
def main():
    training_samples = args.number_of_samples
    random.seed(42)
    data = load_dataset("jordiae/exebench", split='train_real_compilable')
    data = data.shuffle(42)
#    data = load_from_disk("/media/superuser/Lokaler Datenträger/Datasets/ExebenchTrainReal")
    raw_deobf_dataset = ""
    json_deobf_dataset = [] # use instruction and output
    raw_deobf_datasets = {}
    json_deobf_datasets = {}
    json_deobf2_datasets = {}
    backup_directory("original")
    backup_directory("obfuscated")
    duplicate_string = ""
    transformations = ["encode_arithmetic", "encode_branches", "flatten", "opaque", "randomize_arguments"]
#    transformations = ["opaque"]

    too_longs = 0
    
    names = []
    for t in transformations:
        raw_deobf_datasets[t] = ""
        json_deobf_datasets[t] = []
        json_deobf2_datasets[t] = []

    for i, sample in enumerate(data):
        if i >= training_samples:
            break

        status_array = check_sample_preconditions(sample, names)

        if status_array != [0, 0, 0, 0, 0, 0, 0]:
            training_samples += 1
            continue

        max_file_name_length = 256-len(os.getcwd())-43 # the -43 is because of the .c ending and the added path plus the slashes and substrcating the length of _tigress_canonicalized_function to cover the longest appendix too, theoretically it could break later when obfuscating due to the transformation suffix but when the obfuscate fails the sample will be skipped anyways so this case is covered too
        # we can identify unique long names but on disk it might be overwritten, but since we only need the training dataset this is ok
        with open("datasets/original/" + sample['fname'][:max_file_name_length] + duplicate_string + ".c", "w") as f:
            f.write(normalize_data_types(build_program(sample, is_main=sample['fname'] == "main")))

        current_length = len(names)
        print(training_samples)
        print(i)
        print(current_length)
        param_count = count_function_parameters(sample['func_def'], sample['fname'])
        obfs_status = obfuscate(sample['fname'][:max_file_name_length] + duplicate_string + ".c", sample['fname'], "", param_count, sample)

        if obfs_status != 0:
            print("Error during obfuscation")
            training_samples += 1
            continue

        errors = 0
        raw_codes = {}
        json_codes = {}
        json_codes2 = {}

        for t in transformations:
            # we could randomize the identifier names even further since tigress uses a fixed naming convention after CleanUp for id names but we could also argue that any program that is to be deobfuscated could be converted into this form and therefore model generalization with this respect might not be as important
            suffix = "_" + t + ".c"

            # erroneous extractions are skipped

            test = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'], "", False, t == "opaque", t == "encode_branches")

            if type(test) == str() and test == "":
                print(f"transformation {t}: Error during the extraction")
                errors += 1
                print(sample['real_deps'])
                print(sample['func_def'])
                continue

            try:
                #with open(f"datasets/original/{sample['fname'][:max_file_name_length].removesuffix('.c')}_rnd_function.c", "r") as f:
                with open(f"datasets/original/{sample['fname'][:max_file_name_length].removesuffix('.c')}_rnd_function.c", "r") as f:
                    randomized_code = f.read()
            except:
                errors += 1
                continue

            try:
                _, code_without_helper_defs = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'],"", False, t == "opaque", t == "encode_branches", extract_helpers=False)
            except:
                errors += 1
                print("Fail to extract without helper funcs")
                continue

            with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_partially_rnd.c", "w") as f:
                f.write(code_without_helper_defs)

            try:
                _, code_with_helper_defs = extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'],"", False, t == "opaque", t == "encode_branches", extract_helpers=True)
            except:
                errors += 1
                print("Fail to extract with helper funcs")
                continue

            with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_partially_rnd_with_helpers.c", "w") as f:
                f.write(code_with_helper_defs)

#            with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_partially_rnd_without_helpers.c", "w") as f:
#                f.write(code_without_helper_defs)

            obfs_ids, labels = get_identifier_names(code_with_helper_defs, ignore_function_declarations=False)
            obfs_ids_corrected = []

            for obfs_id in obfs_ids:
                if obfs_id.split("::")[-1] in code_without_helper_defs: # careful, heuristic might break if struct has the same attr name as e. g., a global var, although id randomization should minimize risk
                    obfs_ids_corrected.append(obfs_id)

            with open("datasets/original/" + sample['fname'][:max_file_name_length] + "_tigress_canonicalized_function.c", "r") as f:
                canonicalized_function = f.read()

            if canonicalized_function == "":
                errors += 1
                continue

            if code_without_helper_defs == "":
                errors += 1
                continue

            code = "// Obfuscatedcode\n" + remove_comments(code_without_helper_defs) + "\n// Deobfuscatedcode\n" + remove_comments(canonicalized_function)

            with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_sample.c", "w") as f:
                f.write(code)

            randomized_code_obfs = randomize_identifiers2(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_partially_rnd_with_helpers.c", f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_sample.c", identifier_names=obfs_ids_corrected, labels=labels, ignore_func_decls=False)

            randomized_code_obfs = post_process(randomized_code_obfs).replace("__RND__", "") + "<|end|>"
            randomized_code_obfs = randomized_code_obfs.replace("// Obfuscatedcode\n", "// Obfuscated code\n").replace("// Deobfuscatedcode\n", "// Deobfuscated code\n")

            print("Start")

            print(randomized_code_obfs)
            print("End")
            print("----")
            obfs, orig = randomized_code_obfs.split("// Obfuscated code\n")[1].split("<|end|>")[0].split("\n// Deobfuscated code\n")
    #        randomized_code_obfs = format_obf_org_pair(obfs, orig, "deepseek-coder-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct")
    #        randomized_code_obfs_tokenized = format_obf_org_pair(obfs, orig, "deepseek-coder-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct", return_tokens=True)

            # also obfuscated-original-pairs that are too large for the model are left out, continue sampling until the desired sample count is achieved without the special cases here
            if not check_token_length_limits(obfs, orig, [("deepseek-coder-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct"), ("codellama-base", "codellama/CodeLlama-7b-hf"), ("gpt-4", "gpt-4")], args.max_tokens):
#            if len(randomized_code_obfs_tokenized) > args.max_tokens:
                print("code is too long")
                too_longs += 1
                errors += 1
                continue

            randomized_code_obfs = "<|OBFS|>\n" + obfs + "<|ORIG|>\n" + orig

#            with open(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_sample.c", "w") as f:
#                f.write(code)

#            code = randomize_identifiers(f"datasets/obfuscated/{sample['fname'][:max_file_name_length]}{duplicate_string}_{t}_sample.c", identifier_names=obfs_ids_corrected, labels=labels, ignore_func_decls=False)
            # use tigress to canonicalize the original code and use it as the deobfuscated version instead of the original one to ease the mapping between deobfuscated and obfuscated code, maybe a dataset version with the original functions is also interesting to see if the LLMs are still able to learn the mapping or if this becomes significantly harder
        
            raw_codes[t] = randomized_code_obfs
            json_codes[t] = {'instruction' : "// Obfuscated code\n" + extract_function("datasets/obfuscated/" + sample['fname'][:max_file_name_length] + duplicate_string + suffix, sample['fname'], "", False, t == "opaque", t == "encode_branches", extract_helpers=False)[1] + "\n// Deobfuscated code\n", 'input' : '', 'output' : randomized_code}
            json_codes2[t] = {str(i) + "__name__" + sample['fname'] : randomized_code_obfs}

        # throw the sample away if for any transformation any error occurred to make sure that the difference between the datasets is only the transformation and not the function, this way the incorporated functions are synced among the datasets with the different transformations
        if errors == 0:
 #           print(code)
            for t in transformations:
                raw_deobf_datasets[t] += raw_codes[t]
                json_deobf_datasets[t].append(json_codes[t])
                json_deobf2_datasets[t].append(json_codes2[t])
                #print("current length " + str(len(raw_deobf_datasets[t].split("<|end|>"))))

                if current_length % 100 == 0:
                    suffix1 = f"_{t}_{args.max_tokens}_{current_length}.c"
                    suffix2 = f"_{t}_{args.max_tokens}_{current_length}.c"
                    save_dataset_raw(raw_deobf_datasets[t], suffix1)
                    save_dataset_json(json_deobf_datasets[t], suffix1)
                    save_dataset_json(json_deobf2_datasets[t], suffix2)

            names.append(sample['fname'])

        else:
            training_samples += 1

        if current_length % 100 == 0:
            with open(f"datasets/training_dataset_generation_progress_{args.max_tokens}_{current_length}.txt", "w") as f:
                f.write(f"{i}\n{training_samples}")

            with open(f"datasets/training_dataset_generation_progress_names_{args.max_tokens}_{current_length}.txt", 'wb') as file:
                pickle.dump(names, file)

            with open(f"datasets/training_dataset_too_longs_{args.max_tokens}_{current_length}.txt", "w") as f:
                f.write(str(too_longs))

        print(current_length)

    print(names)

    for t in transformations:
        suffix1 = "_" + t + "_" + str(args.max_tokens) + ".c"
        suffix2 = "_" + t + "_" + str(args.max_tokens) + ".c"
        save_dataset_raw(raw_deobf_datasets[t], suffix1)
        save_dataset_json(json_deobf_datasets[t], suffix1)
        save_dataset_json(json_deobf2_datasets[t], suffix2)

    with open(f"datasets/training_dataset_too_longs_{args.max_tokens}.txt", "w") as f:
        f.write(str(too_longs))

if __name__ == "__main__":
    main()
