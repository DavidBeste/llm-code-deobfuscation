import argparse
import os
import subprocess

from utils import (
    create_folder_if_not_exists,
    extract_function,
    extract_function_name_from_file,
    load_jsonl_dataset
)

transformations = ["encode_arithmetic", "encode_branches", "flatten", "opaque", "randomize_arguments"]

# define parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--eval_dataset_path', required=True)
parser.add_argument('--original_path')
parser.add_argument('--obfuscated_path')
parser.add_argument('--deobfuscated_path')

parser.add_argument('--io_path')
parser.add_argument('--orig_data_suffix')
parser.add_argument('--obfs_data_suffix')
parser.add_argument('--data_suffix', default='')
#parser.add_argument('--obfuscation_type', choices=['single', 'chain'])

parser.add_argument('--no_metrics', action='store_true')

args = parser.parse_args()

d = load_jsonl_dataset(args.eval_dataset_path)

function_names = [list(n.keys())[0].split("__name__")[1] for n in d]

# Sample usage: python3 eval_deobf.py --eval_dataset_path datasets/obfuscation_dataset_encode_arithmetic_eval.json --original_path datasets/original --obfuscated_path datasets/obfuscated --deobfuscated_path datasets/deobfuscated_encode_arithmetic --data_suffix encode_arithmetic --io_path datasets/input_samples

def calc_metrics(files, path, suffix_to_remove, fname="", all_functions=False):
    for file, filename in files:
        if not filename:
            continue

        if all_functions == True:
            command = "tigress --Environment=x86_64:Linux:Gcc:11.3.0 --Transform=SoftwareMetrics --SoftwareMetricsKind=* --Functions=* --SoftwareMetricsJsonFileName=" + path + "/" + file.removesuffix(".c") + "_metrics.json " + path + "/" + file + " --out=" + path  + "/" + file.removesuffix(".c") + "_tigress_out.c"

        else:
            command = "tigress --Environment=x86_64:Linux:Gcc:11.3.0 --Transform=SoftwareMetrics --SoftwareMetricsKind=* --Functions=" + filename + " --SoftwareMetricsJsonFileName=" + path + "/" + file.removesuffix(".c") + "_metrics.json " + path + "/" + file + " --out=" + path + "/" + file.removesuffix(".c") + "_tigress_out.c"

        print(command)
        os.system(command)
#        os.remove(path + file.removesuffix(".c") + "_tigress_out.c") # no need to keep this file



def calc_all_metrics(original, obfuscated, deobfuscated, obfs_data_suffix, data_suffix):
#        original_files = [file for file in os.listdir(original) if file.endswith(".c") and not "tigress_out" in file]
        original_files = [(f + ".c", f) for f in function_names]

        print(obfuscated + "/" + function_names[0] + obfs_data_suffix + ".c")

        print(deobfuscated + "/" + function_names[1] + data_suffix + ".c")
        obfuscated_files = [(f + obfs_data_suffix + ".c", extract_function(out_file=obfuscated + "/" + f + obfs_data_suffix + ".c", function_name=f, extract_helpers=False)[0]) for f in function_names]
        deobfuscated_files = [(f + data_suffix + ".c", extract_function(out_file=obfuscated + "/" + f + obfs_data_suffix + ".c", function_name=f, extract_helpers=False)[0]) for f in function_names]
#        deobfuscated_files = [(f + data_suffix + ".c", extract_function_name_from_file(deobfuscated + "/" + f + data_suffix + ".c")) for f in function_names]

#        obfuscated_files = [file for file in os.listdir(obfuscated) if file.endswith(".c") and not "tigress_out" in file]
#       obfuscated_files = [file for file in os.listdir(obfuscated) if file.endswith("llm_prompt_with_main.c") and (not "_randomize_arguments_" in file or "_final_" in file)]

    #    print(obfuscated_files)
#        deobfuscated_files = [file for file in os.listdir(deobfuscated) if file.endswith(".c") and not "tigress_out" in file]
##        deobfuscated_files = [(f + "_" + t + "_ext_new_1.c", extract_function_name_from_file(deobfuscated + f + "_" + t + "_ext_new_1.c")) for f in function_names for t in transformations]


#        print("Obfuscated files")



        calc_metrics(original_files, original, ".c", False)
        calc_metrics(obfuscated_files, obfuscated, ".c", False)
        calc_metrics(deobfuscated_files, deobfuscated, ".c", False)


def run_semantical_tests(path, io_path, function_name, data_suffix): # path without file ending
    create_folder_if_not_exists(f"{path}/{function_name}{data_suffix}")
    if not os.path.exists(f"{path}/{function_name}{data_suffix}.exe") or True:
        os.system(f"g++ {path}/{function_name}{data_suffix}.cpp -o {path}/{function_name}{data_suffix}.exe -I lib/ -fpermissive")

    if os.path.exists(f"{path}/{function_name}{data_suffix}.exe"):
        inputs = [f for f in os.listdir(io_path + "/" + function_name) if f.endswith(".json")]

        for inp in inputs:
            cmd = (f"{path}/{function_name}{data_suffix}.exe", io_path + "/" + function_name + "/" + inp, f"{path}/{function_name}{data_suffix}/output_{inp.removeprefix('input_')}")
            print(cmd)
            p = subprocess.Popen(cmd)

            try:
##                    wait_time = avg_execution_times[function_name + "_chain"]
                p.wait(10)

            except subprocess.TimeoutExpired:
                p.kill()

def main():
    if not args.no_metrics:
        calc_all_metrics(f"{args.original_path}_eval", f"{args.obfuscated_path}_eval", f"{args.deobfuscated_path}_eval", args.obfs_data_suffix, args.data_suffix)

    for function_name in function_names:
        run_semantical_tests(f"{args.original_path}_io_test", args.io_path, function_name, "")
        run_semantical_tests(f"{args.obfuscated_path}_io_test", args.io_path, function_name, args.obfs_data_suffix)
        run_semantical_tests(f"{args.deobfuscated_path}_io_test", args.io_path, function_name, args.data_suffix)

if __name__ == "__main__":
    main()
