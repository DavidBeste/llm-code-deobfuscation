import collections
from functools import reduce
import itertools
import json
import math
import numpy as np
import random
from shlex import split
import os
import re
import subprocess
import time

import torch
from transformers import StoppingCriteria

from utils import (
    build_bogus_parameters,
    build_program,
    check_hardcoded_params,
    create_folder_if_not_exists,
    extract_function,
    extract_parameter_permutations,
    find_variable_type_in_code,
    get_function_parameters,
    get_function_parameters_ex,
    get_variable_types,
    insert_after_match,
    normalize_data_types
)

def count_elements(arr):
    element_count = {}
    for element in arr:
        if element in element_count:
            element_count[element] += 1
        else:
            element_count[element] = 1
    return element_count

def get_metrics_function_name(sample):
    try:
        with open(sample, "r") as f:
            metrics = json.load(f)

    except:
        return []

    return metrics[2]['name']

def load_metrics(sample):
    try:
        with open(sample, "r") as f:
            metrics = json.load(f)

    except:
        return []

#    print(metrics[1]['name'], metrics[2]['name'])

    if len(metrics) == 24:
        if metrics[22]['name'] == "main":
#            print(metrics[19]['name'])
            return [metrics[19]['value'], metrics[20]['N'], metrics[20]['eta']]

#        print(metrics[22]['name'])
        return [metrics[22]['value'], metrics[23]['N'], metrics[23]['eta']]

    if metrics[1]['name'] == "main":
#        print(metrics[4]['name'])
        return [metrics[4]['value'], metrics[5]['N'], metrics[5]['eta']]

#    print(metrics[1]['name'])
#    for m in metrics:
#        print(m['name'])
    return [metrics[1]['value'], metrics[2]['N'], metrics[2]['eta']]

def load_metrics_new(sample):
    try:
        with open(sample, "r") as f:
            metrics = json.load(f)

    except:
        return []

    return [metrics[1]['value'], metrics[2]['N'], metrics[2]['eta']]

def get_code(filename, function_name, extract_code):

    if extract_code:
        return extract_function(filename, function_name)

    try:
        with open(filename, "r") as f:
            code = f.read()
    except:
        print("Error, no such file!")
        return ""

    return code

def obfuscate_simple(filename, target_function, parameter_count, orig_path, obfs_path):
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
    
    
    # full_cmd = cmd + " datasets/original_eval/" + filename + " --out=" + "datasets/original_eval/" + filename.removesuffix(".c") + "_tigress_canonicalized.c"

    for k in command_templates.keys():
        full_cmd = cmd + command_templates[k]

        if k != "basic":
            full_cmd += command_templates['basic']

        full_cmd += " " + orig_path + filename + " --out=" + obfs_path + filename.removesuffix(".c") + "_" + k + ".c"

        print(full_cmd)

        try:
            tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=100)

        except subprocess.TimeoutExpired:
            return -1
        
    return 0

        
# source: https://stackoverflow.com/questions/69403613/how-to-early-stop-autoregressive-model-with-a-list-of-stop-words, accessed 21.11.2022 20:57
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

#    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#        if input_ids[0][-1] in self.keywords:
#            print("Input_ids: " + str(input_ids))
#            print("Input_ids[0]: " + str(input_ids[0]))
#            print("Input_ids[0][-1]: " + str(input_ids[0][-1]))
#            print("self.keywords" + str(self.keywords))
#            return True
#        return False

# changed so that the model will only stop when reaching // Vulnerable code and not just //
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        counter = 0
        for i in range(len(self.keywords)):
            if input_ids[0][i-len(self.keywords)] != self.keywords[i]:
                return False

        return True

def obfuscate(filename, sample, target_function, parameter_count, t_chain):
    tigress_header = "#include \"/usr/local/bin/tigresspkg/3.3.3/tigress.h\"\n"

    opaque_choice = ",".join(random.sample(["list", "array", "env"], random.randint(1,3)))
    opaque_choice2 = random.sample(["call", "bug", "true", "junk", "fake_call", "question"], 1)

    opaque_chain = ""

    for choice in opaque_choice2:
        opaque_chain += " --Transform=AddOpaque --Functions=" + target_function + " --AddOpaqueStructs=" + opaque_choice + " --AddOpaqueKinds=" + choice

    if "question" in opaque_choice2:
        opaque_chain += " --Transform=Inline --Functions=/.*QUESTION.*/"

    command_templates = {
        "basic" : " --Transform=CleanUp --CleanUpKinds=names,annotations",
        "encode_arithmetic" : " --Transform=EncodeArithmetic --Functions=" + target_function,
        "encode_branches" : " --Transform=AntiBranchAnalysis --Functions=" + target_function + " --AntiBranchAnalysisKinds=branchFuns --AntiBranchAnalysisObfuscateBranchFunCall=false --AntiBranchAnalysisBranchFunFlatten=true",
#        "encode_data" : " --Transform=EncodeData --LocalVariables=" + target_function + ":" + target_local_variables + " --EncodeDataCodecs=poly1,
        "flatten" : " --Transform=Flatten --Functions=" + target_function + " --FlattenRandomizeBlocks=" + ["true", "false"][random.randint(0,1)] + " --FlattenSplitBasicBlocks=" + ["true", "false"][1] + " --FlattenDispatch=" + ["switch", "goto", "indirect"][random.randint(0,2)] + " --FlattenConditionalKinds=" + ["branch", "compute", "flag"][random.randint(0,2)],
#        "merge" : " --Transform=Merge --Functions=" + first_function + "," + second_function,
        "opaque" : opaque_chain,  # maybe for more complex opaque predicates additional helper functions and structures are needed?
        "randomize_arguments" : " --Transform=RndArgs  --RndArgsBogusNo=" + str([random.randint(1,5), random.randint(int(parameter_count*0.5), parameter_count)][parameter_count > 0]) + " --Functions=" + target_function,
#        "split" : " --Transform=Split --Functions=" + target_function,
#        "encode_literals" : " --Transform=InitOpaque --Functions=" + target_function + " --Transform=EncodeLiterals --Functions=" + target_function, # needs opaque expressions for computing function addresses by default, even if all additional settings are set to false tigress still requires InitOpaque
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

    full_cmd = cmd + " datasets/original_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + ".c --out=" + "datasets/original_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_tigress_canonicalized.c"
    print(full_cmd)
#    os.system(full_cmd)
    try:
        tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=100)

    except subprocess.TimeoutExpired:
        return -1

    try:
        function_name, canonicalized_program = extract_function("datasets/original_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_tigress_canonicalized.c", target_function, "", False, False, False)

    except ValueError:
        print("Failed to extract canonicalized program")
        return -1

    with open("datasets/original_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_tigress_canonicalized_function.c", "w") as f:
        f.write(tigress_header + sample['real_deps'].replace("# 1", "") + "\n" + normalize_data_types(canonicalized_program) + "\nint main(){}\n")

    if not "randomize_arguments" in t_chain:
        chain = ""

        for t in t_chain:
            chain += command_templates[t] + " "

        chain += command_templates["basic"]

        full_cmd = cmd + chain
        # add basic at the end of every sample
#       full_cmd = cmd + command_templates[k] + " original_eval/" + filename + " --out=" + "obfuscated/" + filename.removesuffix(".c") + "_" + k + "_cleanup.c"
        if not "encode_branches" in t_chain:
            full_cmd += " --Transform=SoftwareMetrics --SoftwareMetricsKind=* --Functions=* --SoftwareMetricsJsonFileName=" + "datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_chain" + "_metrics.json"

        full_cmd += " datasets/original_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_tigress_canonicalized_function.c" + " --out=" + "datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_chain.c"
        print(full_cmd)
#            os.system(full_cmd)
        try:
            tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=100)

        except subprocess.TimeoutExpired:
            return -1

    else:
        pivot = t_chain.index('randomize_arguments') + 1
#            pre_rnd, post_rnd = t_chain[:pivot], t_chain[pivot:]
        pre_rnd = t_chain
        post_rnd = []

        chain = ""
        for t in pre_rnd:
            chain += command_templates[t] + " "

        chain += " --Transform=CleanUp --CleanUpKinds=annotations"
#            chain += command_templates["basic"]
        full_cmd = cmd + chain
#       full_cmd = cmd + command_templates[k] + " original_eval/" + filename + " --out=" + "obfuscated/" + filename.removesuffix(".c") + "_" + k + "_cleanup.c"

        full_cmd += " datasets/original_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_tigress_canonicalized_function.c" + " --out=" + "datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_chain_randomize_arguments_intermediate.c"
        #full_cmd += " datasets/original_eval/" + filename + " --out=" + "datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_chain.c"
        print(full_cmd)
#            os.system(full_cmd)
        try:
            tigress_out = subprocess.run(split(full_cmd), capture_output=True, text=True, timeout=100)

        except subprocess.TimeoutExpired:
            return -1

        chain = ""
        try:
            function_name, intermediate_program = extract_function("datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_chain_randomize_arguments_intermediate.c", target_function, "", False, "opaque" in t_chain, "encode_branches" in t_chain)

        except ValueError:
            print("Failed to extract intermediate program")
            return -2

        with open("datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_chain_randomize_arguments_intermediate_function.c", "w") as f:
            f.write(tigress_header + sample['real_deps'].replace("# 1", "") + "\n" + intermediate_program + "\nint main(){}\n")

#            regex_file_replace("datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_chain_randomize_arguments_intermediate.c",
#                                r"\/\* BEGIN FUNCTION-DECL __2_bf_1 LOC=UNKNOWN (.|\n)*\/\* END FUNCTION-DECL __2_bf_1 LOC=UNKNOWN \*\/")
#            regex_file_replace("datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_chain_randomize_arguments_intermediate_function.c",
#                                r"\/\* BEGIN FUNCTION-DECL __2_bf_1 LOC=UNKNOWN (.|\n)*\/\* END FUNCTION-DECL __2_bf_1 LOC=UNKNOWN \*\/")

        for t in post_rnd:
            chain += command_templates[t] + " "

        chain += command_templates["basic"]
        cmd2 = cmd + chain

        if not "encode_branches" in t_chain:
            cmd2 += " --Transform=SoftwareMetrics --SoftwareMetricsKind=* --Functions=* --SoftwareMetricsJsonFileName=" + "datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_chain_metrics.json"

        cmd2 += " datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_chain_randomize_arguments_intermediate_function.c" + " --out=" + "datasets/obfuscated_eval/" + filename.removesuffix(".c") + "_" + str(len(t_chain)) + "_chain.c"
        print(cmd2)
#            os.system(cmd2)


        try:
            tigress_out = subprocess.run(split(cmd2), capture_output=True, text=True, timeout=100)

        except subprocess.TimeoutExpired:
            return -1

    return 0

def similarity(out1, out2):
    total_elements = 0
    similar_elements = 0
    
    for e1, e2 in zip(out1.values(), out2.values()):
        if type(e1) == list or type(e1) == str:
            for i1, i2 in zip(e1, e2):
                total_elements += 1
                if i1 == i2:
                    similar_elements +=1
                else:
                    continue
    else:
        if e1 == e2:
            similar_elements += 1
        total_elements+=1

    return (float)(similar_elements/total_elements)

def remove_duplicate_encode_branches(filename):
    pattern = r"\/\* BEGIN FUNCTION-DECL __2_bf_1 LOC=UNKNOWN \*\/([\s\S]*?)\/\* END FUNCTION-DECL __2_bf_1 LOC=UNKNOWN \*\/"

    with open(filename, "r") as file:
        content = file.read()

    re.sub(pattern, "", content)
    
    with open(filename, "w") as file:
        file.write(content)

def remove_duplicate_encode_branches2(filename):

    pattern1 = r"\/\* BEGIN FUNCTION-DEF __2_bf_1 LOC=UNKNOWN \*\/([\s\S]*?)\/\* END FUNCTION-DEF __2_bf_1 LOC=UNKNOWN \*\/"


    with open(filename, "r") as file:
        content = file.read()

    # safeguard to prevent delete the encode branches helper function entirely
    print(content.count("/* END FUNCTION-DEF __2_bf_1 LOC=UNKNOWN */"))
    
    if content.count("/* END FUNCTION-DEF __2_bf_1 LOC=UNKNOWN */") <= 1:
        return

    # remove first decl and def using lazy regex
    content = re.sub(pattern1, "", content, 1)

    print(content.count("/* END FUNCTION-DEF __2_bf_1 LOC=UNKNOWN */"))
    
    
    pattern2 = r"\/\* BEGIN FUNCTION-DEF __2_bf_1 LOC=UNKNOWN \*\/\nvoid[^;]*;"

    # remove second decl but not def so only the top decl and one def remain
    content = re.sub(pattern2, "", content)
    print(content.count("/* END FUNCTION-DEF __2_bf_1 LOC=UNKNOWN */"))
    
    with open(filename, "w") as file:
        file.write(content)

    print(content.count("/* END FUNCTION-DEF __2_bf_1 LOC=UNKNOWN */"))

def get_permutations(initial_array, final_array):
    perm = []

    for index in range(len(initial_array)):
        if initial_array[index] not in final_array:
            perm.append(-1)
        else:
            old_idx = final_array.index(initial_array[index])
            perm.append(old_idx)

    return perm

def extract_argument_names_2(source_code, function_name):
    variables = re.search(re.escape(function_name) + r"\([^)]*\)\s*;", source_code.split("int main")[1]).group().removeprefix(function_name + "(")[:-2].split(",")
    return [var.strip() for var in variables]

def get_id_names_mapping(source_code, func_params, function_name):
    mapping = {}
    variables = re.search(function_name + r"\([^)]*\)\s*;", source_code.split("int main")[1]).group().removeprefix(function_name + "(")[:-2].split(",")
    variables = [var.strip() for var in variables]
    func_param_names = [func_param[1] for func_param in func_params]
    for func_param_name, variable in zip(func_param_names, variables):
        mapping[func_param_name] = variable

    return mapping
    #print(func_param_names)
    #print(variables)

def extract_valid_permutations(source_code, func_params, function_name, bogus_arguments = None, bogus_type_dict = None, cap=10):
#    if variables == None:

    print(f"func_params: {func_params}")

    func_param_types = [func_param[0] for func_param in func_params]
    print(f"func_param_types: {func_param_types}")
    func_param_types = ["long"]*4+["float"]*2
    func_param_type_count = dict(collections.Counter(func_param_types))
    result = reduce((lambda x, y: x * y), [math.factorial(v) for v in func_param_type_count.values()])
    print(f"Result: {result}")
    #num_possibilities = [math.factorial(value) for key, value in func_param_type_count.items()]
    #print(num_possibilities)
    variables = re.search(function_name + r"\([^)]*\)\s*;", source_code).group().removeprefix(function_name + "(")[:-2].split(",")
    print(f"Variables: {variables}")
    variables = [var.strip() for var in variables]
#    print(f"Variables: {variables}")

    types = get_variable_types(source_code, variables)
    if bogus_arguments:
        variables.extend(bogus_arguments)

    possible_candidates = {}
    permutations_2d = list(itertools.permutations(variables))

#    print(f"Types {types}")

    # If you want the result as a list of lists (2D array)
    permutations = [list(permutation) for permutation in permutations_2d]

    valid_permutations = []

#    print(f"func params {func_params}")

    for permutation in permutations:
        error = False
#        print(permutation)

        for param, (t, n) in zip(permutation, func_params):
#            print("Param " + param)
#            print("t " + t)
#            print("n " + n)
            if "&var_" in param:
                var_type = get_variable_types(source_code, [param.removeprefix("&")])

#            print(param)
            if param not in [str(0), str(0.0), "\"\""] and not "&var_" in param and t != types[param] or (param in [str(0), str(0.0), "\"\""] and check_hardcoded_params(param, t) == False) or ("&var_" in param and var_type == bogus_type_dict[param.removeprefix("&")]):
                error = True
                break

        if not error:
            valid_permutations.append(permutation)

    return valid_permutations

def get_fake_function_parameter_types(obfuscated_code, fake_call):
    if not re.search(fake_call + r"\([^)]*\)", obfuscated_code):
        return []

    parameters = re.search(fake_call + r"\([^)]*\)", obfuscated_code).group().removeprefix(fake_call + "(")[:-1].split(",")
    bogus_parameters = []

    for parameter in parameters:
            # direct numbers are used for the fake call
        if parameter.strip().isdigit():
            bogus_parameters.append("int")

        else:
            if parameter != "":
                datatype = find_variable_type_in_code(obfuscated_code, parameter.strip())

                if datatype:
                    bogus_parameters.append(datatype)

        return bogus_parameters

def modify_wrapper(sample, wrapper, obfuscated_code, suffix):    
    if not sample['fname'] in wrapper: # obfuscated_program before
        print("Function name not in exe found")
        return ""

    wrapper = wrapper

    if not re.search(sample['fname'] + r"\s*\([^)]*\)", wrapper):
        print("Function call not in exe wrapper")
        return ""

    function_arguments = re.search(sample['fname'] + r"\s*\([^)]*\)", wrapper).group().removeprefix(sample['fname'] + "(")[:-1].split(",")
    param_perm = extract_parameter_permutations(sample['func_def'], extract_function("datasets/obfuscated_eval/" + sample['fname'] + suffix, sample['fname'])[1], sample['fname'])
    new_params, additional_vars = build_bogus_parameters(function_arguments, param_perm)
    new_function_call = sample['fname'] + "(" + ",".join(new_params) + ")"
    wrapper = re.sub(sample['fname'] + r"\s*\([^)]*\)", new_function_call, wrapper)

    if additional_vars != "":
        main_head = re.search("main\s*\([^)]*\)\s*{", wrapper).group()
        wrapper = re.sub("main\s*\([^)]*\)\s*{", main_head + "\n" + additional_vars, wrapper)

    # obfuscated_program = build_program(sample=sample, empty_main=False, is_main=False, func_def_is_external=True, func_def=obfuscated_code, modified_wrapper=wrapper)
    obfuscated_program = build_program(sample=sample, empty_main=False, is_main=False, func_def_is_external=True, func_def="", modified_wrapper=wrapper, no_extern_c=True)

    return obfuscated_program

def prepare_wrapper(name, obfs_name, wrapper, function_parameters, function_return_type, transformation, is_O0=False):
    load_dll_code = """\n
    // Step 3: Load the shared object
    void* handle = dlopen("/home/llm-server/FineTuning/a.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading shared object: " << dlerror() << std::endl;
        return 1;
    }

    // Step 4: Get function pointers
    typedef void (*YourFunctionType)(/* function parameters */);
    YourFunctionType yourFunction = (YourFunctionType)dlsym(handle, "_x16");

    if (!yourFunction) {
        std::cerr << "Error getting function pointer: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }\n
    """

    load_dll_code = '\n    // Step 3: Load the shared object\n    void* handle = dlopen("/home/llm-server/FineTuning/datasets/obfuscated_eval/' + name + '_' + transformation + '_function' + ["", "_O0"][is_O0] + '.so", RTLD_LAZY);\n    if (!handle) {\n        std::cerr << "Error loading shared object: " << dlerror() << std::endl;\n        return 1;\n    }\n\n    // Step 4: Get function pointers\n    typedef ' + function_return_type + ' (*YourFunctionType)' + function_parameters + ';\n    YourFunctionType ' + name + ' = (YourFunctionType)dlsym(handle, "' + obfs_name + '");\n\n    if (!' + name + ') {\n        std::cerr << "Error getting function pointer: " << dlerror() << std::endl;\n        dlclose(handle);\n        return 1;\n    }\n'
    wrapper = insert_after_match(r"int\s+main\s*\([^)]*\)\s*\{", load_dll_code, wrapper)
    return wrapper

def prepare_wrapper_generic(name, obfs_name, wrapper, function_parameters, function_return_type, path):
    load_dll_code = """\n
    // Step 3: Load the shared object
    void* handle = dlopen("/home/llm-server/FineTuning/a.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading shared object: " << dlerror() << std::endl;
        return 1;
    }

    // Step 4: Get function pointers
    typedef void (*YourFunctionType)(/* function parameters */);
    YourFunctionType yourFunction = (YourFunctionType)dlsym(handle, "_x16");

    if (!yourFunction) {
        std::cerr << "Error getting function pointer: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }\n
    """

    load_dll_code = '\n    // Step 3: Load the shared object\n    void* handle = dlopen("' + path + '", RTLD_LAZY);\n    if (!handle) {\n        std::cerr << "Error loading shared object: " << dlerror() << std::endl;\n        return 1;\n    }\n\n    // Step 4: Get function pointers\n    typedef ' + function_return_type + ' (*YourFunctionType)' + function_parameters + ';\n    YourFunctionType ' + obfs_name + ' = (YourFunctionType)dlsym(handle, "' + obfs_name + '");\n\n    if (!' + obfs_name + ') {\n        std::cerr << "Error getting function pointer: " << dlerror() << std::endl;\n        dlclose(handle);\n        return 1;\n    }\n'
    wrapper = insert_after_match(r"int\s+main\s*\([^)]*\)\s*\{", load_dll_code, wrapper)
    return wrapper

def check_correctness(function_name, function_name_no_suffix, path):
    create_folder_if_not_exists(path + function_name)

    if not os.path.exists(path + function_name + ".exe") or True:
        os.system("g++ " + path + function_name + ".cpp" + " -o " + path + function_name + ".exe -I lib/")

    if not os.path.exists(path + function_name + ".exe"):
        return "Exe missing"

    inputs = [f for f in os.listdir("datasets/input_samples/" + function_name_no_suffix) if f.endswith(".json")]

    for inp in inputs:
#        cmd = "./original_io_test/" + function_name + ".exe " + "input_samples/" + function_name + "/" + inp + " original_io_test/" + function_name + "/output_" + inp.removeprefix("input_") + " >> original_run.log"
#        print(cmd)
#        os.system(cmd)
        cmd = ["./" + path + function_name + ".exe", "datasets/input_samples/" + function_name_no_suffix + "/" + inp, path + function_name + "/output_" + inp.removeprefix("input_")]
        start = time.time()
        p = subprocess.Popen(cmd)

        try:
            p.wait(10)

        except subprocess.TimeoutExpired:
            p.kill()
            return "Program is too slow"

        end = time.time()
#        avg_execution_time += end - start

        if os.path.exists("datasets/input_samples/" + function_name_no_suffix + "_io_samples.json"):

      #      print("input_samples/" + function_name_no_suffix + "_io_samples.json")

            with open("datasets/input_samples/" + function_name_no_suffix + "_io_samples.json", "r") as f:
                io_samples = json.load(f)

        for i, io_sample in enumerate(io_samples['output']):
      #      print(io_sample['output'])
            if not os.path.exists(path + function_name + "/output_" + str(i) + ".json"):
                return "Output sample is missing"

            with open(path + function_name + "/output_" + str(i) + ".json", "r") as f:
    #                print("obfuscated_io_test/" + sample + "_" + transformation + "/output_" + str(i) + ".json")
                try:
                    out = json.load(f)
                except:
                    return "Error at json parsing"

            io_sample_formated = {}
            for var, value in zip(io_sample['var'], io_sample['value']):
    #                print(value)
                io_sample_formated[var] = json.loads(value)

            # as soon as one output differs we consider it to be semantically incorrect
            if out != io_sample_formated:
                return 0

    return 1

def check_correctness2(function_name_no_suffix, function_name, path1, path2, no_run=False, verbose=False):
    create_folder_if_not_exists(path1 + function_name_no_suffix)
    create_folder_if_not_exists(path2 + function_name)

    # fpermissive is necessary because of some implicit casts from void * to other types but causes problems when using clang llvm since fpermissive does not exist here, unclear if it should be used or not
    if (not os.path.exists(path1 + function_name_no_suffix + ".exe") or True) and not no_run:
        os.system("g++ " + path1 + function_name_no_suffix + ".cpp" + " -o " + path1 + function_name_no_suffix + ".exe -I lib/ -fpermissive")

    if not os.path.exists(path1 + function_name_no_suffix + ".exe"):
#        syntax_broken.append(function_name_no_suffix)
        return f"Original exe {path1}{function_name_no_suffix}.exe is missing"

    if (not os.path.exists(path2 + function_name + ".exe") or True) and not no_run:
        os.system("g++ " + path2 + function_name + ".cpp" + " -o " + path2 + function_name + ".exe -I lib/ -fpermissive")

    if not os.path.exists(path2 + function_name + ".exe"):
#        syntax_broken.append(function_name)
        return f"Obfuscated exe {path2}{function_name}.exe is missing"

    inputs = [f for f in os.listdir("datasets/input_samples/" + function_name_no_suffix) if f.endswith(".json")]

#    print("datasets/input_samples/" + function_name_no_suffix)

    if len(inputs) != 10:
        return "Wrong number of io samples present"

    is_correct = True

    for i in range(10):
        if not no_run:
            cmd = ["./" + path1 + function_name_no_suffix + ".exe", "datasets/input_samples/" + function_name_no_suffix + "/input_" + str(i) + ".json", path1 + function_name_no_suffix + "/output_" + str(i) + ".json"]
            print(cmd)
            start = time.time()
            p = subprocess.Popen(cmd)

            try:
                p.wait(10)

            except subprocess.TimeoutExpired:
                p.kill()
    #            runtime_broken.append(function_name_no_suffix)
                return "Original program is too slow"

            end = time.time()

            cmd = ["./" + path2 + function_name + ".exe", "datasets/input_samples/" + function_name_no_suffix + "/input_" + str(i) + ".json", path2 + function_name + "/output_" + str(i) + ".json"]
            start = time.time()
            print(cmd)
            p = subprocess.Popen(cmd)

            try:
                p.wait(10)

            except subprocess.TimeoutExpired:
                p.kill()
    #            runtime_broken.append(function_name)
                return "Obfuscated program is too slow"

            end = time.time()

        if not os.path.exists(path1 + function_name_no_suffix + "/output_" + str(i) + ".json"):
#            output_broken.append(function_name_no_suffix)
            return "Original output sample is missing"

        if not os.path.exists(path2 + function_name + "/output_" + str(i) + ".json"):
#            output_broken.append(function_name)
            return f"Obfuscated output sample {path2}{function_name}/output_{i} is missing"

        with open(path1 + function_name_no_suffix + "/output_" + str(i) + ".json", "r") as f:
            try:
                original_out = json.load(f)
            except json.JSONDecodeError:
#                output_json_broken.append(function_name_no_suffix)
                return "Error at json parsing original"


        with open(path2 + function_name + "/output_" + str(i) + ".json", "r") as f:
    #                print("obfuscated_io_test/" + sample + "_" + transformation + "/output_" + str(i) + ".json")
            try:
                obfuscated_out = json.load(f)
            except json.JSONDecodeError:
#                output_json_broken.append(function_name)
                return "Error at json parsing obfuscated"

#        if similarity(original_out, obfuscated_out) > 0.9 and similarity(original_out, obfuscated_out) < 1:

#            pass
    #        print(original_out)
    #        print(obfuscated_out)
    #        print(f"nano datasets/original_eval/{function_name_no_suffix}_tigress_canonicalized.c datasets/deobfuscated_eval/{function_name}.c")

#        for _ in obfuscated_out.values():
#            if type(_) == list:
#                print(np.array(_).shape)

        if original_out != obfuscated_out:
#            print("nano " + path1 + function_name_no_suffix)
#            print("nano " + path2 + function_name)
#            print(original_out)
#            print(obfuscated_out)
#            semantics_broken.append(function_name)
            is_correct = False

            if verbose:
                print(original_out)
                print(obfuscated_out)
        
    if not is_correct:
        return 0

    return 1

def check_correctness3(function_name_no_suffix, function_name, function_name_plain, path1, path2, no_run=False, verbose=False):
    create_folder_if_not_exists(path1 + function_name_no_suffix)
    create_folder_if_not_exists(path2 + function_name)

    # fpermissive is necessary because of some implicit casts from void * to other types but causes problems when using clang llvm since fpermissive does not exist here, unclear if it should be used or not
    if (not os.path.exists(path1 + function_name_no_suffix + ".exe") or True) and not no_run:
        os.system("g++ " + path1 + function_name_no_suffix + ".cpp" + " -o " + path1 + function_name_no_suffix + ".exe -I lib/ -fpermissive")

    if not os.path.exists(path1 + function_name_no_suffix + ".exe"):
#        syntax_broken.append(function_name_no_suffix)
        return f"Original exe {path1}{function_name_no_suffix}.exe is missing"

    if (not os.path.exists(path2 + function_name + ".exe") or True) and not no_run:
        os.system("g++ " + path2 + function_name + ".cpp" + " -o " + path2 + function_name + ".exe -I lib/ -fpermissive")

    if not os.path.exists(path2 + function_name + ".exe"):
#        syntax_broken.append(function_name)
        return f"Obfuscated exe {path2}{function_name}.exe is missing"

    inputs = [f for f in os.listdir("datasets/input_samples/" + function_name_plain) if f.endswith(".json")]

#    print("datasets/input_samples/" + function_name_no_suffix)

    if len(inputs) != 10:
        return "Wrong number of io samples present"

    is_correct = True

    for i in range(10):
        if not no_run:
            cmd = ["./" + path1 + function_name_no_suffix + ".exe", "datasets/input_samples/" + function_name_no_suffix + "/input_" + str(i) + ".json", path1 + function_name_no_suffix + "/output_" + str(i) + ".json"]
            print(cmd)
            start = time.time()
            p = subprocess.Popen(cmd)

            try:
                p.wait(10)

            except subprocess.TimeoutExpired:
                p.kill()
    #            runtime_broken.append(function_name_no_suffix)
                return "Original program is too slow"

            end = time.time()

            cmd = ["./" + path2 + function_name + ".exe", "datasets/input_samples/" + function_name_no_suffix + "/input_" + str(i) + ".json", path2 + function_name + "/output_" + str(i) + ".json"]
            start = time.time()
    #        print(cmd)
            p = subprocess.Popen(cmd)

            try:
                p.wait(10)

            except subprocess.TimeoutExpired:
                p.kill()
    #            runtime_broken.append(function_name)
                return "Obfuscated program is too slow"

            end = time.time()

        if not os.path.exists(path1 + function_name_no_suffix + "/output_" + str(i) + ".json"):
#            output_broken.append(function_name_no_suffix)
            return "Original output sample is missing"

        if not os.path.exists(path2 + function_name + "/output_" + str(i) + ".json"):
#            output_broken.append(function_name)
            return f"Obfuscated output sample {path2}{function_name}/output_{i} is missing"

        with open(path1 + function_name_no_suffix + "/output_" + str(i) + ".json", "r") as f:
            try:
                original_out = json.load(f)
            except json.JSONDecodeError:
#                output_json_broken.append(function_name_no_suffix)
                return "Error at json parsing original"


        with open(path2 + function_name + "/output_" + str(i) + ".json", "r") as f:
    #                print("obfuscated_io_test/" + sample + "_" + transformation + "/output_" + str(i) + ".json")
            try:
                obfuscated_out = json.load(f)
            except json.JSONDecodeError:
#                output_json_broken.append(function_name)
                return "Error at json parsing obfuscated"

#        if similarity(original_out, obfuscated_out) > 0.9 and similarity(original_out, obfuscated_out) < 1:

#            pass
    #        print(original_out)
    #        print(obfuscated_out)
    #        print(f"nano datasets/original_eval/{function_name_no_suffix}_tigress_canonicalized.c datasets/deobfuscated_eval/{function_name}.c")

#        for _ in obfuscated_out.values():
#            if type(_) == list:
#                print(np.array(_).shape)

        if original_out != obfuscated_out:
#            print("nano " + path1 + function_name_no_suffix)
#            print("nano " + path2 + function_name)
#            print(original_out)
#            print(obfuscated_out)
#            semantics_broken.append(function_name)
            is_correct = False

            if verbose:
                print(original_out)
                print(obfuscated_out)

    if not is_correct:
        return 0

    return 1

def correct_function_arguments(io_wrapper, f_name, dir_name, obfs_data_suffix, data_suffix):
    with open(f"datasets/obfuscated_io_test/{f_name}{obfs_data_suffix}.cpp", "r") as f:
        code1 = f.read()

    with open(f"datasets/obfuscated_eval/{f_name}{obfs_data_suffix}.c", "r") as f:
        func1 = f.read()

    obfuscated_function_name, obfuscated_code = extract_function(f"datasets/obfuscated_eval/{f_name}{obfs_data_suffix}.c", f_name, "", False, False, False)
    args1 = get_function_parameters_ex(obfuscated_code, obfuscated_function_name)
    code2 = io_wrapper

    with open(f"datasets/deobfuscated_eval/{f_name}{data_suffix}.c", "r") as f:
        func2 = f.read()


#    print("func1")
#    print(obfuscated_code)
#    print("func2")
#    print(func2)


    
    args2 = get_function_parameters_ex(func2, obfuscated_function_name)

    if type(args2) == str:
        # probably the llm renamed the function so try to extract the function parameters anyways
        args2 = get_function_parameters(func2)


    args1 = [arg[1] for arg in args1]
    args2 = [arg[1] for arg in args2]
    invented_args = [arg2 for arg2 in args2 if arg2 not in args1]


#    print("Args1")
#    print(args1)
#    print("Args2")
#    print(args2)


    if len(invented_args) == 0:
        perms = get_permutations(args1, args2)
#        print("Perms")
        print(perms)
        try:
#            print("For code1")
            variables1 = extract_argument_names_2(code1, obfuscated_function_name)
#            print("For code2")
            variables2 = extract_argument_names_2(code2, f_name)

        except AttributeError as e:
            print(e)
            print("Error, probably the LLM renamed the function")
            

#            return code2

        new_variables = [0]*len(args2)

        for v1, p in zip(variables1, perms):
            print(v1)
            print(p)
            if p >= 0:
                new_variables[p] = str(v1)

        bogus_declarations = ""
        code2_parts = code2.split("int main(")
        code2_parts[1] = re.sub(re.escape(f_name) + r"\([^)]*\)\s*;", f"{obfuscated_function_name}({','.join(new_variables)});", code2_parts[1])
        extract_bogus_declarations = True

        for line in code1.split("int main(")[1].split("\n")[1:]:
            if not "var_" in line:
                extract_bogus_declarations = False

            if extract_bogus_declarations:
                bogus_declarations += line + "\n"

            if not extract_bogus_declarations:
                break

        if bogus_declarations != "":
            main_head = re.search("main\s*\([^)]*\)\s*{", f"int main({code2_parts[1]}").group()
            code2_parts[1] = re.sub("main\s*\([^)]*\)\s*{", main_head + "\n" + bogus_declarations, f"int main({code2_parts[1]}")
            code2 = f"{code2_parts[0]}{code2_parts[1]}"

        else:
            code2 = f"{code2_parts[0]}int main({code2_parts[1]}"

    # if the LLM invented an ID name this way of tracing the permutations will fail so skip this sample
    else:
        print("LLM has invented an argument for this sample. Skipping the procedure since this behavior is incompatible with our heuristics. Possible fallback´: Brute force wrappers by datatype matching.")

#    print(code2)
    
    return code2

def compute_deobfuscation_performance(original, obfuscated, deobfuscated):
    #return 1 - (obfuscated - original) / (deobfuscated - original))
    #print(obfuscated)
    #print(deobfuscated)
    original = np.array(original)
    obfuscated = np.array(obfuscated)
    deobfuscated = np.array(deobfuscated)
    return 1 - (deobfuscated - original) / (obfuscated - original)
    #return np.where(obfuscated <= original, np.minimum(original, obfuscated) / deobfuscated, 1 - (obfuscated - original) / (deobfuscated - original))
#        avg_execution_time += end - start

#        if os.path.exists("input_samples/" + function_name_no_suffix + "_io_samples.json"):

      #      print("input_samples/" + function_name_no_suffix + "_io_samples.json")

#            with open("input_samples/" + function_name_no_suffix + "_io_samples.json", "r") as f:
#                io_samples = json.load(f)

      #  for i, io_sample in enumerate(io_samples['output']):
#        for i in range(10):
      #      print(io_sample['output'])
#            if not os.path.exists(path1 + function_name_no_suffix + "/output_" + str(i) + ".json"):
#                return "Original output sample is missing"

#            if not os.path.exists(path2 + function_name + "/output_" + str(i) + ".json"):
#                return "Obfuscated output sample is missing"

#            is_semantically_correct = 1

#            with open(path1 + function_name_no_suffix + "/output_" + str(i) + ".json", "r") as f:
    #                print("obfuscated_io_test/" + sample + "_" + transformation + "/output_" + str(i) + ".json")
#                try:
#                    original_out = json.load(f)
#                except:
#                    return "Error at json parsing original"

#            with open(path2 + function_name + "/output_" + str(i) + ".json", "r") as f:
    #                print("obfuscated_io_test/" + sample + "_" + transformation + "/output_" + str(i) + ".json")
#                try:
#                    obfuscated_out = json.load(f)
#                except:
#                    return "Error at json parsing obfuscated"

    #        io_sample_formated = {}
    #        for var, value in zip(io_sample['var'], io_sample['value']):
    #                print(value)
    #            io_sample_formated[var] = json.loads(value)

#            if original_out != obfuscated_out:
#                is_semantically_correct = 0

#    return is_semantically_correct
