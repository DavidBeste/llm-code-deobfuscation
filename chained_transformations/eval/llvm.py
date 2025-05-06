import argparse
import json
import os
import shutil

from datasets import load_dataset
from llvmlite import ir
from llvmlite import binding as llvm

from llvm_utils import calculate_cyclomatic_complexity, calculate_halstead_metrics
from utils import extract_function2 as extract_function, get_function_parameters_re_ex, get_function_return_type_ex, load_jsonl_dataset
from utils_eval import correct_function_arguments, modify_wrapper, prepare_wrapper_generic

def create_backup(filepath):
    if not os.path.exists(f"{filepath}.bak"):
        shutil.copyfile(filepath, "{filepath}.bak")

def remove_keywords(file_path):
    try:

        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

        if not ("static" in content or "inline" in content):
            return

        create_backup(file_path)

        # Remove 'static' and 'inline' keywords
        content = content.replace('static', '').replace('__inline', '').replace('inline', '')

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.write(content)

    except FileNotFoundError:
        print(f"File not found: {file_path}")

def calculate_metrics_program(ir_code, function_name):
    # Initialize the LLVM target and data layout (needed for parsing the IR)
    llvm.initialize()
    llvm.initialize_all_targets()

    # Create an LLVM context and module
    llvm_context = llvm.get_global_context()
    module = llvm.parse_assembly(ir_code)

    for func in module.functions:
        if function_name in func.name and not func.is_declaration:
            return calculate_cyclomatic_complexity(func), calculate_halstead_metrics(func)


    # target function name not found in the intermediate representation
    return (-1,(-1,)*11)
    
def calculate_metrics_program_from_disk(filepath, function_name):
    # Load the LLVM IR from a file
    if os.path.exists(filepath):
        with open(filepath) as ir_file:
            ir_code = ir_file.read()

        return calculate_metrics_program(ir_code, function_name)

    else:
        return (-2,(-2,)*11)

def save_metrics(path, function_name, cyc_compl, N, eta):
    metrics = [{"metric": "Raw", "name": function_name},{"metric": "McCabe", "name": function_name, "value" : cyc_compl},{"metric": "Halstead", "name": function_name, "N": N, "eta": eta}]

    with open(path, "w") as f:
        json.dump(metrics, f)


def build_llvm_io_wrapper():
    pass

# define parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--orig_data_suffix', default='')
parser.add_argument('--obfs_data_suffix', default='')
parser.add_argument('--data_suffix', default='')
parser.add_argument('--eval_file', default='')
args = parser.parse_args()

data = load_dataset("jordiae/exebench", split='test_real')
data = data.shuffle(42)

d = load_jsonl_dataset(args.eval_file)
dataset_indices = [int(list(n.keys())[0].split("__name__")[0]) for n in d]
function_names = [list(n.keys())[0].split("__name__")[1] for n in d]

error_dict = {"No orig" : 0, "No obfs" : 0, "Orig func not found" : 0, "Obfs func not found" : 0}


for index, function_name in zip(dataset_indices, function_names):
    remove_keywords("datasets/original_eval/" + function_name + args.orig_data_suffix + ".c")
    os.system("clang -S -fPIC -emit-llvm -O3 datasets/original_eval/" + function_name + args.orig_data_suffix + ".c -o datasets/original_eval/" + function_name + args.orig_data_suffix + ".ll")
    os.system(f"clang -shared -fPIC -undefined dynamic_lookup datasets/original_eval/{function_name}{args.orig_data_suffix}.ll -o datasets/original_eval/{function_name}{args.orig_data_suffix}.so")
    print(calculate_metrics_program_from_disk("datasets/original_eval/" + function_name + args.orig_data_suffix + ".ll", function_name))
    cycl, (n1, n2, N1, N2, vocabulary, program_length, program_volume, difficulty, effort, time, bugs) = calculate_metrics_program_from_disk("datasets/original_eval/" + function_name + args.orig_data_suffix + ".ll", function_name)
    print(program_length)
    print(cycl)
    save_metrics("datasets/original_eval/" + function_name + args.orig_data_suffix + "_llvm_metrics.json", function_name, cycl, program_length, vocabulary)

    remove_keywords("datasets/obfuscated_eval/" + function_name + args.obfs_data_suffix + ".c_function.c")
    os.system("clang -S -fPIC -emit-llvm -O3 datasets/obfuscated_eval/" + function_name + args.obfs_data_suffix + ".c_function.c -o datasets/deobfuscated_eval/" + function_name + args.obfs_data_suffix + ".ll")
    os.system(f"clang -shared -fPIC -undefined dynamic_lookup datasets/deobfuscated_eval/{function_name}{args.obfs_data_suffix}.ll -o datasets/deobfuscated_eval/{function_name}{args.obfs_data_suffix}.so")
    function_name_obfuscated, code_obfuscated = extract_function("datasets/obfuscated_eval/" + function_name + args.obfs_data_suffix + ".c", function_name, extract_helpers=True)
    function_parameters = get_function_parameters_re_ex(code_obfuscated, function_name_obfuscated)
    function_return_type = get_function_return_type_ex(code_obfuscated, function_name_obfuscated)
    print(code_obfuscated)
    print(function_name_obfuscated)
    cycl, (n1, n2, N1, N2, vocabulary, program_length, program_volume, difficulty, effort, time, bugs) = calculate_metrics_program_from_disk("datasets/deobfuscated_eval/" + function_name + args.obfs_data_suffix + ".ll", function_name_obfuscated)
    print(program_length)
    save_metrics("datasets/deobfuscated_eval/" + function_name + args.obfs_data_suffix + "_llvm_metrics.json", function_name, cycl, program_length, vocabulary)
    w = prepare_wrapper_generic(function_name, function_name_obfuscated, "#include <dlfcn.h>\n" + "\n".join(data[index]['real_exe_wrapper'].split("\n")[3:]), function_parameters, function_return_type, "datasets/deobfuscated_eval/" + function_name + args.obfs_data_suffix + ".so")

    with open(f"datasets/obfuscated_io_test/{function_name}{args.obfs_data_suffix}.cpp", "r") as f:
        obfuscated_wrapper = f.read()

    split_id = "\n".join(data[index]['real_exe_wrapper'].split("\n")[3:])[:10]
    new_wrapper = "#include <dlfcn.h>\n#include <vector>" + obfuscated_wrapper.split("#include <vector>")[1]
    new_wrapper = prepare_wrapper_generic(function_name, function_name_obfuscated, new_wrapper, function_parameters, function_return_type, "datasets/deobfuscated_eval/" + function_name + args.obfs_data_suffix + ".so")
#    w = correct_function_arguments(w, function_name, "datasets/deobfuscated_eval", args.obfs_data_suffix, args.data_suffix)
    print(new_wrapper)
    with open(f"datasets/deobfuscated_io_test/{function_name}{args.obfs_data_suffix}_llvm.cpp", "w") as f:
        f.write(new_wrapper)
    #print(correct_function_arguments(w, function_name, "datasets/obfuscated_eval", args.obfs_data_suffix, args.data_suffix))
    os.system("clang -S -fPIC -emit-llvm -O0 datasets/obfuscated_eval/" + function_name + args.obfs_data_suffix + ".c_function.c -o datasets/obfuscated_eval/" + function_name + args.obfs_data_suffix + ".ll")
    os.system(f"clang -shared -fPIC -undefined dynamic_lookup datasets/obfuscated_eval/{function_name}{args.obfs_data_suffix}.ll -o datasets/obfuscated_eval/{function_name}{args.obfs_data_suffix}.so")
    cycl, (n1, n2, N1, N2, vocabulary, program_length, program_volume, difficulty, effort, time, bugs) = calculate_metrics_program_from_disk("datasets/obfuscated_eval/" + function_name + args.obfs_data_suffix + ".ll", function_name_obfuscated)
    new_wrapper = "#include <dlfcn.h>\n#include <vector>" + obfuscated_wrapper.split("#include <vector>")[1]
    new_wrapper = prepare_wrapper_generic(function_name, function_name_obfuscated, new_wrapper, function_parameters, function_return_type, "datasets/obfuscated_eval/" + function_name + args.obfs_data_suffix + ".so")
    print(new_wrapper)
    w = prepare_wrapper_generic(function_name, function_name_obfuscated, "#include <dlfcn.h>\n" + "\n".join(data[index]['real_exe_wrapper'].split("\n")[3:]), function_parameters, function_return_type, "datasets/obfuscated_eval/" + function_name + args.obfs_data_suffix + ".so")
#    w = correct_function_arguments(w, function_name, "datasets/obfuscated_eval", args.obfs_data_suffix, args.data_suffix)
    with open(f"datasets/obfuscated_io_test/{function_name}{args.obfs_data_suffix}_llvm.cpp", "w") as f:
        f.write(new_wrapper)

    print(program_length)
    save_metrics("datasets/obfuscated_eval/" + function_name + args.obfs_data_suffix + "_llvm_metrics.json", function_name, cycl, program_length, vocabulary)
