from functools import wraps
import glob
import itertools
import json
import os
import random
import re
import shutil
import time

import clang
import clang.cindex as cindex
from pycparser import c_parser

cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-14.so.1")  # Set the path to libclang.so

#  tokenizer = AutoTokenizer.from_pretrained("../Peft/codegen25-7b-multi-full", trust_remote_code=True)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def create_folder_if_not_exists(folder_name : str) -> None:
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def backup_directory(path_name):
    if not os.path.exists(f"datasets/{path_name}_old"):
        os.makedirs(f"datasets/{path_name}_old")
    
    if os.path.exists(f"datasets/{path_name}"):
        # Move data from obfuscated to obfuscated_old
        for file_name in os.listdir(f"datasets/{path_name}"):
            shutil.move(f"datasets/{path_name}/{file_name}", f"datasets/{path_name}_old/{file_name}")

    else:
        os.makedirs(f"datasets/{path_name}")

def save_dataset_raw(samples, suffix):
    with open("datasets/obfuscation_dataset" + suffix.removesuffix(".c") + ".txt", "w") as f:
        f.write(samples)

def save_dataset_json(samples, suffix):
    with open("datasets/obfuscation_dataset" + suffix.removesuffix(".c") + ".json", "w") as f:
        for sample in samples:
            json_line = json.dumps(sample)
            f.write(json_line + '\n')

def save_input_samples(sample, suffix, max_file_name_length=None):
    if max_file_name_length == None:
        max_file_name_length = len(sample['fname'])

    create_folder_if_not_exists("datasets/input_samples/" + sample['fname'][:max_file_name_length] + suffix)

    with open("datasets/input_samples/" + sample['fname'][:max_file_name_length] + suffix + "_io_samples.json", "w") as f:
        f.write(json.dumps(sample['real_io_pairs']) + "\n")

    for _,line in enumerate(sample['real_io_pairs']['input']):
        input_sample = dict()

        for var, value in zip(line['var'], line['value']):
            input_sample[var] = eval(value)

        with open("datasets/input_samples/" + sample['fname'][:max_file_name_length] + suffix + "/input_" + str(_) + ".json", "w") as f:
            f.write(json.dumps(input_sample) + "\n")

def add_int_main_to_file(fname: str):
    with open(fname, 'r') as f:
        c = f.read()
        
    if not  "int main" in c:
        c +=  "\n\nint main(){}\n"
            
    with open(fname, 'w') as f:
        f.write(c)

def add_int_main_to_files():
    # get all the files ending with _chain.c_function.c from obfuscated_eval directory
    files = sorted(glob.glob('datasets/obfuscated_eval/*1_chain.c_function.c'))
    print(*map(os.path.basename, files), sep='\n')
    for file in files:
        add_int_main_to_file(file)

def regex_file_replace(filename, pattern):
    # Open the file in read mode
    with open(filename, 'r') as f:
        content = f.read()

    # Replace all occurrences of the pattern
    new_content = re.sub(pattern, '', content)  # replace '' with your desired replacement string

    # Open the same file again but in write mode
    with open(filename, 'w') as f:
        f.write(new_content)

def insert_after_match(pattern, replacement, input):
    matches = re.findall(pattern, input)

    if len(matches) == 0:
        return input   # Return original string if no matches found

    newStr = ""
    start = 0
    for m in matches:
        idx = input.index(m, start)

        # Append everything before match to newStr
        newStr += input[start : idx]

        # Add the match and replacement to newStr
        newStr += m + replacement

        # Update start position for next iteration
        start = idx + len(m)

    # Append remaining unprocessed part of the input string
    newStr += input[start:]
    return newStr

def normalize_data_types(code):
#    /usr/lib/gcc/x86_64-linux-gnu/11/include/stddef.h
    code = code.replace("ssize_t", "long").replace("size_t", "long unsigned int").replace("wchar_t", "int").replace("uint32_t", "unsigned int").replace("uint16_t", "short unsigned int").replace("socklen_t", "unsigned int")
    return code

def check_transformation_order(t_chain, invalid_orders=[("encode_branches", "flatten")]):
    # Track the last index of each letter
    last_index = {letter: index for index, letter in enumerate(t_chain)}   

    for index, letter in enumerate(t_chain):
        # Check if the current letter violates any invalid order
        for invalid_order in invalid_orders:
            first_letter, second_letter = invalid_order

            if not first_letter in t_chain or not second_letter in t_chain:
                continue

            if letter == first_letter and last_index[second_letter] > index:
                return False
        # Update the last index of the current letter
        last_index[letter] = index
    
    return True

def get_single_command(cmd, target_function, parameter_count):
    opaque_choice = ",".join(random.sample(["list", "array", "env"], random.randint(1,3)))
    opaque_choice2 = random.sample(["call", "bug", "true", "junk", "fake_call", "question"], 1) # random.randint(1,6) for more complexity instead of 1

    opaque_chain = ""

    for choice in opaque_choice2:
        opaque_chain += " --Transform=AddOpaque --Functions=" + target_function + " --AddOpaqueStructs=" + opaque_choice + " --AddOpaqueKinds=" + choice

    if "question" in opaque_choice2:
        opaque_chain += " --Transform=Inline --Functions=/.*QUESTION.*/"

    command_templates = {
            "encode_arithmetic" : " --Transform=EncodeArithmetic --Functions=" + target_function,
            "encode_branches" : " --Transform=InitBranchFuns --InitBranchFunsCount=1 --Transform=AntiBranchAnalysis --Functions=" + target_function + " --AntiBranchAnalysisKinds=branchFuns --AntiBranchAnalysisObfuscateBranchFunCall=false --AntiBranchAnalysisBranchFunFlatten=true",
            "flatten" : " --Transform=Flatten --Functions=" + target_function + " --FlattenRandomizeBlocks=" + ["true", "false"][random.randint(0,1)] + " --FlattenSplitBasicBlocks=" + ["true", "false"][1] + " --FlattenDispatch=" + ["switch", "goto", "indirect"][random.randint(0,2)] + " --FlattenConditionalKinds=" + ["branch", "compute", "flag"][random.randint(0,2)],
            "opaque" : " --Transform=InitOpaque --InitOpaqueStructs=" + opaque_choice + " --Functions=init_tigress" + opaque_chain ,
            "randomize_arguments" : " --Transform=RndArgs  --RndArgsBogusNo=" + str([random.randint(1,5), random.randint(int(parameter_count*0.5), parameter_count)][parameter_count > 0])  + " --Functions=" + target_function
    }

    return command_templates[cmd]

def get_chain_command(cmd, target_function, opaque_chain, parameter_count):
    command_templates = {
        "basic" : " --Transform=CleanUp --CleanUpKinds=names,annotations",
        "encode_arithmetic" : " --Transform=EncodeArithmetic --Functions=" + target_function,
        "encode_branches" : " --Transform=AntiBranchAnalysis --Functions=" + target_function + " --AntiBranchAnalysisKinds=branchFuns --AntiBranchAnalysisObfuscateBranchFunCall=false --AntiBranchAnalysisBranchFunFlatten=true",
        "flatten" : " --Transform=Flatten --Functions=" + target_function + " --FlattenRandomizeBlocks=" + ["true", "false"][random.randint(0,1)] + " --FlattenSplitBasicBlocks=" + ["true", "false"][1] + " --FlattenDispatch=" + ["switch", "goto", "indirect"][random.randint(0,2)] + " --FlattenConditionalKinds=" + ["branch", "compute", "flag"][random.randint(0,2)],
        "opaque" : opaque_chain,  
        "randomize_arguments" : " --Transform=RndArgs  --RndArgsBogusNo=" + str([random.randint(1,5), random.randint(int(parameter_count*0.5), parameter_count)][parameter_count > 0]) + " --Functions=" + target_function,
    }

    return command_templates[cmd]

def get_random_chain(chain_length=-1, replacement=True):
    t_chain = []

    while not check_transformation_order(t_chain) or t_chain == []:
        transformations = ["encode_arithmetic", "encode_branches",
                               "flatten", "opaque", "randomize_arguments"]
        if chain_length == -1:
            transformation_count = random.randint(0, 5)

        elif chain_length > 0:
            transformation_count = chain_length

        else:
            return []

        t_chain = []

        for j in range(transformation_count):
            random_transformation = random.choice(transformations)
            t_chain.append(random_transformation)

            if not replacement:
                transformations.remove(random_transformation)

    return t_chain
    

def get_function_parameters(function):
    index = cindex.Index.create()
    tu = index.parse("tmp.c", args=['-std=c99'], unsaved_files=[('tmp.c', function)])

    if not tu:
        return "ERROR"

    for node in tu.cursor.get_children():
        if node.kind == cindex.CursorKind.FUNCTION_DECL:
            params = [(param.type.spelling, param.spelling) for param in node.get_children() if param.kind == cindex.CursorKind.PARM_DECL]
            return params

    return "ERROR"

#def extract_argument_names(source_code, function_name):
#    index = cindex.Index.create()
#    translation_unit = index.parse('temp.cpp', args=['-std=c++11'], unsaved_files=[('temp.cpp', source_code)])

#    for cursor in translation_unit.cursor.walk_preorder():
#        if cursor.kind == cindex.CursorKind.CALL_EXPR and cursor.spelling == function_name:
#            argument_names = [arg.spelling for arg in cursor.get_children()]
#            return argument_names
#            print(argument_names)

"""def extract_argument_names(source_code, function_name):
    index = clang.cindex.Index.create()
    translation_unit = index.parse('temp.cpp', args=['-std=c++11'], unsaved_files=[('temp.cpp', source_code)])

    argument_names = []

    for cursor in translation_unit.cursor.walk_preorder():
        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL and cursor.spelling == function_name:
            # Skip function declarations
            continue

        if cursor.kind == clang.cindex.CursorKind.CALL_EXPR and cursor.spelling == function_name:
            # Extract argument names only for function calls
            argument_names.extend(arg.spelling for arg in cursor.get_children())
#            argument_names.extend(arg.spelling for arg in cursor.get_children() if arg.kind == clang.cindex.CursorKind.PARM_DECL)

    return argument_names"""

def extract_argument_names(source_code, function_name):
    index = clang.cindex.Index.create()
    translation_unit = index.parse('temp.cpp', args=['-std=c++11'], unsaved_files=[('temp.cpp', source_code)])

    argument_names = []

    for cursor in translation_unit.cursor.walk_preorder():
        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL and cursor.spelling == function_name:
            # Skip function declarations
            continue

        if cursor.kind == clang.cindex.CursorKind.CALL_EXPR and cursor.spelling == function_name:
            # Extract argument names and constant values for function calls
            for arg in cursor.get_children():
#                if arg.kind == clang.cindex.CursorKind.PARM_DECL:
                argument_names.append(arg.spelling)
#                elif arg.kind == clang.cindex.CursorKind.INTEGER_LITERAL:
                argument_names.append(arg.displayname)

    return argument_names

def get_function_parameters_ex(function_name, identifier_name, std='-std=c99'):
    index = cindex.Index.create()
    tu = index.parse("tmp.c", args=[std], unsaved_files=[('tmp.c', function_name)])

    if not tu:
        print("No tu")
        return "ERROR"

    for node in tu.cursor.get_children():
        if node.kind == cindex.CursorKind.FUNCTION_DECL and node.spelling == identifier_name:
            params = [(param.type.spelling, param.spelling) for param in node.get_children() if param.kind == cindex.CursorKind.PARM_DECL]
            return params

    print("func not found")
    return "ERROR"

def get_function_parameters_re_ex(function_name, identifier_name):
    return re.search(re.escape(identifier_name) + r"\s*\([^;{]*;", function_name).group().removeprefix(identifier_name).removesuffix(";")

def get_function_return_type(function):
    index = cindex.Index.create()
    tu = index.parse('test.c', args=['-std=c99'], unsaved_files=[('test.c', function)])

    for child in tu.cursor.walk_preorder():
        if child.kind == cindex.CursorKind.FUNCTION_DECL:
          #  print("Function {} returns {}".format(child.spelling, child.result_type.spelling))
            return child.result_type.spelling
        
def intersect_lists_in_dict(dictionary):
    # Get the values (lists) from the dictionary
    lists = list(dictionary.values())
    
    # If the dictionary is empty or has only one list, return an empty list
    if len(lists) < 2:
        return []
    
    # Initialize the intersection with the first list
    intersection = set(lists[0])
    
    # Iterate through the lists and update the intersection
    for lst in lists[1:]:
        intersection.intersection_update(lst)
    
    return list(intersection)

def get_function_return_type_ex(function, identifier_name):
    index = cindex.Index.create()
    tu = index.parse('test.c', args=['-std=c99'], unsaved_files=[('test.c', function)])

    for child in tu.cursor.walk_preorder():
        if child.kind == cindex.CursorKind.FUNCTION_DECL and child.spelling == identifier_name:
          #  print("Function {} returns {}".format(child.spelling, child.result_type.spelling))
            return child.result_type.spelling

def build_bogus_parameters(original, permutations):
        new_parameters = []
        additional_variables = ""
        variable_counter = 0

        for permutation in permutations:
                if type(permutation) == str:
                        if permutation == "int" or permutation == "long":
                                new_parameters.append(str(0))

                        elif permutation == "float" or permutation == "double":
                                new_parameters.append(str(0.0))

                        elif permutation == "char":
                                new_parameters.append("\"\"")

                        elif permutation == "void *":
                                additional_variables += "int var_" + str(variable_counter) + ";\n"
                                new_parameters.append("&var_" + str(variable_counter))
                                variable_counter += 1

                        elif permutation == "int *":
                                additional_variables += "int var_" + str(variable_counter) + ";\n"
                                new_parameters.append("&var_" + str(variable_counter))
                                variable_counter += 1

                        elif permutation == "long *":
                                additional_variables += "long var_" + str(variable_counter) + ";\n"
                                new_parameters.append("&var_" + str(variable_counter))
                                variable_counter += 1

                        elif permutation == "float *":
                                additional_variables += "float var_" + str(variable_counter) + ";\n"
                                new_parameters.append("&var_" + str(variable_counter))
                                variable_counter += 1

                        elif permutation == "double *":
                                additional_variables += "double var_" + str(variable_counter) + ";\n"
                                new_parameters.append("&var_" + str(variable_counter))
                                variable_counter += 1

                        elif permutation == "char *":
                                additional_variables += "char var_" + str(variable_counter) + ";\n"
                                new_parameters.append("&var_" + str(variable_counter))
                                variable_counter += 1

                elif type(permutation) == int:
                        new_parameters.append(original[permutation])

        return new_parameters, additional_variables

def find_unresolved_symbols(code):
    index = cindex.Index.create()
    translation_unit = index.parse("in_memory_code.cpp",
                                   unsaved_files=[("in_memory_code.cpp", code)])

    unresolved_symbols = []
    
    for diagnostic in translation_unit.diagnostics:
        if diagnostic.severity >= cindex.Diagnostic.Error:
            message = diagnostic.spelling
            print(message)
            print(diagnostic, diagnostic.severity)


#            print("file not found" in message)
#            if not "file not found" in message:
#                exit()

            if "use of undeclared identifier" in message:
                parts = message.split("'")
                if len(parts) > 1 and not parts[1] in unresolved_symbols:
                    unresolved_symbol = parts[1]
                    unresolved_symbols.append(unresolved_symbol)

    return unresolved_symbols

def find_unresolved_symbols_function(code):
    index = cindex.Index.create()
    translation_unit = index.parse("in_memory_code.c", args=["-std=c99"],
                                   unsaved_files=[("in_memory_code.c", code)])

    unresolved_symbols = []

    for diagnostic in translation_unit.diagnostics:
    #    print(diagnostic.severity)
    #    print(diagnostic.spelling)
    #    print(cindex.Diagnostic.Error)
        
        if diagnostic.severity >= 2:
            message = diagnostic.spelling
            print(message)
            print(diagnostic, diagnostic.severity)

       #     if "use of undeclared identifier" in message:
            if "implicit declaration of function" in message:
                parts = message.split("'")
                if len(parts) > 1 and not parts[1] in unresolved_symbols:
                    unresolved_symbol = parts[1]
                    unresolved_symbols.append(unresolved_symbol)

    return unresolved_symbols    

def get_variable_types(source_file, variable_names):
#    clang.cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-14.so.1")  # Set the path to libclang.so
    index = cindex.Index.create()
    #translation_unit = index.parse("temp.c", args=["-std=c99"], unsaved_files=[("temp.c", source_code)])
    translation_unit = index.parse("temp.cpp", args=['-std=c++11'], unsaved_files=[("temp.cpp", source_file)], options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
    variable_types = {}

    for cursor in translation_unit.cursor.walk_preorder():
        if cursor.kind == clang.cindex.CursorKind.VAR_DECL and cursor.spelling in variable_names:
            variable_types[cursor.spelling] = cursor.type.spelling

    return variable_types

def find_variable_type(node, variable_name):
    if (node.kind == cindex.CursorKind.VAR_DECL and node.spelling == variable_name) or (node.kind == cindex.CursorKind.PARM_DECL and node.spelling == variable_name):
        return node.type.spelling

    for child in node.get_children():
        result = find_variable_type(child, variable_name)
        if result:
            return result

def find_variable_type_in_code(sample_code, variable_name):
    index = cindex.Index.create()
    translation_unit = index.parse("unsaved_code.cpp", unsaved_files=[('unsaved_code.cpp', sample_code)])

    if not translation_unit:
        print("Error parsing the translation unit.")
        return None

    for node in translation_unit.cursor.get_children():
        variable_type = find_variable_type(node, variable_name)
        if variable_type:
            return variable_type

    return None
"""def extract_valid_permutations(source_code, func_params, function_name):
    variables = re.search(function_name + r"\([^)]*\)", source_code).group().removeprefix(function_name + "(")[:-1].split(",")
    variables = [var.strip() for var in variables]
    types = get_variable_types(source_code, variables)
    possible_candidates = {}
    permutations_2d = list(itertools.permutations(variables))

    # If you want the result as a list of lists (2D array)
    permutations = [list(permutation) for permutation in permutations_2d]
    valid_permutations = []

    for permutation in permutations:
        error = 0
        for param, (t, n) in zip(permutation, func_params):
            if t != types[param]:
                error = 1
                break

        if error != 1:
            valid_permutations.append(permutation)"""

def remove_includes(code):
        return re.sub(r"#include\s*<[^>]*>", "", code)
        
def remove_comments(given_string):
    """
    Function to (currently) remove C++ style comments 
    given_string is a string of C/C++ code
    Source: https://github.com/whoward3/C-Code-Obfuscator, implementation here is slightly modified
    """

    #This does not take into account if a C++ style comment happens within a string
    # i.e. "Normal String // With a C++ comment embedded inside"
    cpp_filtered_code = re.findall(
        r"\/\/.*", given_string)
    for entry in cpp_filtered_code:
        given_string = given_string.replace(entry, "")
    
    # This is a barebones start for C style block comments
    # Current issue is it is only single line C style comments
    # It also finds C style comments in strings
#    c_filtered_code= re.findall(
#        r"\/\*.*\*\/", given_string)
    # This part has been changed to cover multi line comments
    # The regex template is from https://stackoverflow.com/questions/13014947/regex-to-match-a-c-style-multiline-comment
    c_filtered_code = re.findall("/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", given_string)
    
    for entry in c_filtered_code:
        given_string = given_string.replace(entry, "")

    given_string = given_string.replace("*/", "") # special case (jis2sjis.c) where a single */ was in the first line after an include that got not removed by remove_includes and caused pycparser to fail

    return given_string

def check_if_compilable(c_file_path):
    try:
        with open(c_file_path, 'r') as c_file:
            c_code = c_file.read()
            parser = c_parser.CParser()
            parser.parse(c_code, filename=c_file_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def check_if_compilable2(c_code):
    # Create a Clang index
    index = clang.cindex.Index.create()

    compilable = True

    # Parse the C code as a translation unit
    tu = index.parse('unsaved_file.c', unsaved_files=[('unsaved_file.c', c_code)])
    for diagnostic in tu.diagnostics:    
        if diagnostic.severity >= cindex.Diagnostic.Error:
            message = diagnostic.spelling
            compilable = False
            print(message)

    return compilable

def check_hardcoded_params(func_param, var_type):
    if func_param == str(0) and (var_type == "int" or var_type == "long"):
        return True

    if func_param == str(0.0) and (var_type == "float" or var_type == "double"):
        return True

    if func_param == "\"\"" and var_type == "char":
        return True

    return False

def extract_valid_permutations(source_code, func_params, function_name, bogus_arguments = None, bogus_type_dict = None):
#    if variables == None:
    variables = re.search(function_name + r"\([^)]*\)", source_code).group().removeprefix(function_name + "(")[:-1].split(",")
    variables = [var.strip() for var in variables]
    print(variables)

    types = get_variable_types(source_code, variables)
    if bogus_arguments:
        variables.extend(bogus_arguments)


    possible_candidates = {}
    permutations_2d = list(itertools.permutations(variables))

    print(types)

    # If you want the result as a list of lists (2D array)
    permutations = [list(permutation) for permutation in permutations_2d]

    valid_permutations = []


    for permutation in permutations:
        error = False


        print(permutation)

        for param, (t, n) in zip(permutation, func_params):
            print("Param " + param)
            print("t " + t)
            print("n " + n)
            if "&var_" in param:
                var_type = get_variable_types(source_code, [param.removeprefix("&")])

            print(param)
            if param not in [str(0), str(0.0), "\"\""] and not "&var_" in param and t != types[param] or (param in [str(0), str(0.0), "\"\""] and check_hardcoded_params(param, t) == False) or ("&var_" in param and var_type == bogus_type_dict[param.removeprefix("&")]):
                error = True
                break

        if not error:
            valid_permutations.append(permutation)

    return valid_permutations




def extract_parameter_permutations(function1, function2, name):
    parameters1 = get_function_parameters_ex(function1, name)
    parameters2 = get_function_parameters_ex(function2, name)

    if type(parameters1) == str or type(parameters2) == str:
        return "ERROR!"

    permutations = []

    for param2 in parameters2:
        if "bogus" in param2[1]:
            permutations.append(param2[0])

        else:
            for i, param1 in enumerate(parameters1):
                #if param1[1] == param2[1]:
                if re.match(r"\b" + param1[1] + r"(___\d+){0,1}\b", param2[1]):
                    permutations.append(i)
                    break

    return permutations

def extract_function_name(code):
    # Initialize the Clang compiler index
    index = cindex.Index.create()

    # Parse the code as a translation unit
    translation_unit = index.parse("temp.c", unsaved_files=[("temp.c", code)])
    current_file_path = translation_unit.cursor.spelling
    # Find the first function declaration in the translation unit
    for node in translation_unit.cursor.walk_preorder():
        if node.kind == cindex.CursorKind.FUNCTION_DECL and node.location.file and node.location.file.name == current_file_path:
            return node.spelling

    return None

def extract_function_name_from_file(filename):
    # Initialize the Clang compiler index
    index = cindex.Index.create()

    # Parse the code from the file as a translation unit
    try:
        translation_unit = index.parse(filename)

    except:
        return None

    # Find the first function declaration in the translation unit
    for node in translation_unit.cursor.walk_preorder():
        if node.kind == cindex.CursorKind.FUNCTION_DECL:
            return node.spelling

    return None

def load_jsonl_dataset(path : str) -> str:
    data = []

    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data

def get_sample_id(function_name):
    id = list(function_name.keys())[0].split("__name__")[0]

    return int(id)

def save_jsonl_dataset(samples, suffix):
    with open("obfuscation_dataset" + suffix + ".json", "w") as f:
        for sample in samples:
            json_line = json.dumps(sample)
            f.write(json_line + '\n')

def get_dependencies(sample, cplusplus=False):
    tigress_header = "#include \"/usr/local/bin/tigresspkg/3.3.3/tigress.h\"\n"
    deps = tigress_header + sample['real_deps'].replace("# 1", "")

    # include stdlib.h is required for opaque
    if not "stdlib.h" in deps:
        deps += "#include <stdlib.h>"

    # include pthread.h is required for concurrent dispatch flattening
    if not "<pthread.h>" in deps:
        deps += "\n#include <pthread.h>"

    # the following two have been added since some exebench samples require it
    if not "<unistd.h>" in deps:
        deps += "\n#include <unistd.h>"

    if not "<stdbool.h>" in deps:
        deps += "\n#include <stdbool.h>"

    if cplusplus:
        if not "<cstdio>" in deps:
            deps += "\n#include <cstdio>"
    else:
        if not "<stdio.h>" in deps:
            deps += "\n#include <stdio.h>"

    return deps

def build_program(sample : str, empty_main : bool = True, is_main : bool = False, func_def_is_external : bool = False, func_def : str = "", modified_wrapper : str = "", no_deps : bool = False, no_extern_c : bool = False) -> str:
    tigress_header = "#include \"/usr/local/bin/tigresspkg/3.3.3/tigress.h\"\n"
    if empty_main == True:
        # build the program for obfuscation with tigress

        deps = get_dependencies(sample, cplusplus=False)

        if no_deps:
            deps = ""

        if func_def_is_external:
            return deps + "\ninit_tigress(){}\n" + func_def + "\n" + ["int main(){}", ""][is_main]

        return deps + "\ninit_tigress(){}\n" + sample['func_def'] + "\n" + ["int main(){}", ""][is_main]

    else:
        # build the program for io testing
        code = ""
        if not no_extern_c:
            
            code += "extern \"C\" {\n"

            deps = get_dependencies(sample, cplusplus=True)
            code += deps + "\n"

            if func_def_is_external:
                # cstdio causes the clang bindings to fail so temporarily remove this header for this step
                fake_call = find_unresolved_symbols(code.removeprefix("extern \"C\" {\n").replace("\n#include <cstdio>", "") + func_def)

                for f in fake_call:
                    if not re.search(f + r"\([^)]*\)", code.removeprefix("extern \"C\" {") + func_def):
                        continue

                    parameters = re.search(f + r"\([^)]*\)", code.removeprefix("extern \"C\" {") + func_def).group().removeprefix(f + "(")[:-1].split(",")
                    bogus_parameters = []

                    for parameter in parameters:
                        # direct numbers are used for the fake call
                        if parameter.strip().isdigit():
                            bogus_parameters.append("int")

                        else:
                            if parameter != "":
                                datatype = find_variable_type_in_code(code.removeprefix("extern \"C\" {") + func_def, parameter.strip())

                                if datatype:
                                    bogus_parameters.append(datatype)

                    # resolve the main conflict in the io wrapper with minimal invasiveness
                    if f == "main":
                        f = "_main"

                #    code += "unsigned int " + f + "(" + ",".join(bogus_parameters) + "){return 0;}\n"
                    if len(bogus_parameters) <= 0:
                        code += "unsigned int " + f + "(){return 0;}\n"

                    else:
                        code += "unsigned int " + f + "(" + bogus_parameters[0] + ", ...){return 0;}\n"

                code += re.sub(r"main\s*\(", "_main(", func_def)

            else:
                code += sample['func_def'] + "\n"

            code += "}\n\n"

        if modified_wrapper == "":
            code += "\n".join(sample['real_exe_wrapper'].split("\n")[3:])
        else:
            code += "\n".join(modified_wrapper.split("\n")[3:])

        return code

def extract_function(out_file : str, function_name : str, second_function_name : str = "", is_merged : bool = False, opaque : bool = False, encode_branches : bool = False, extract_helpers : bool = True, extract_only_helpers : bool = False) -> None:
    target_file = out_file
#        function_name = function_name

#        if "__merged__" in function_name
#       signature =

    try:
        with open(target_file, "r") as f:
            lines = f.readlines()
    except:
        print("Error opening the file. Skipping to the next file")
        return str()

    real_function = str()
    tigress_data = str()
    extracted_function = str()

    do_extract = False
    do_extract_tigress_data = False
    do_extract_real_function = False
    # read line by line until the function signature comment is found

    for i in range(len(lines)):
        if is_merged:
            if lines[i-1] == "/* BEGIN FUNCTION-DEF __1_" + function_name +  "_" + second_function_name + " LOC=UNKNOWN */\n" or lines[i] == "/* BEGIN FUNCTION-DEF __1_" + second_function_name +  "_" + function_name + " LOC=UNKNOWN */\n":
                do_extract = True

            if lines[i] == "/* END FUNCTION-DEF __1_" + function_name +  "_" + second_function_name + " LOC=UNKNOWN */\n" or lines[i] == "/* END FUNCTION-DEF __1_" + second_function_name +  "_" + function_name + " LOC=UNKNOWN */\n":
                do_extract = False

        else:
            if (
                "/* BEGIN" in lines[i-1] and "LOC=UNKNOWN */" in lines[i-1] and
                not re.search(r"( __bswap_(16|32|64) | __uint(16|32|64)_identity | main )", lines[i-1])
                ):

                if not re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) " + function_name, lines[i-1]):
                    do_extract_tigress_data = True

                do_extract = True

            if "/* END" in lines[i] and "LOC=UNKNOWN */" in lines[i]:
                do_extract_tigress_data = False
                do_extract = False

            if re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) init_tigress LOC=UNKNOWN \*\/", lines[i-1]):
                do_extract_tigress_data = True

            if re.search(r"\/* END FUNCTION-(DECL|DEF) init_tigress LOC=UNKNOWN \*\/", lines[i]):
                do_extract_tigress_data = False

            if re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) __\d+_" + (function_name + "_")*2 + "FLATTEN_SPLIT_\d+ LOC=UNKNOWN \*\/", lines[i-1]):
                do_extract_tigress_data = True

            if re.search(r"\/* END FUNCTION-(DECL|DEF) __\d+_" + (function_name + "_")*2 + "FLATTEN_SPLIT_\d+ LOC=UNKNOWN \*\/", lines[i]):
                do_extract_tigress_data = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_entropy LOC=UNKNOWN \*\/", lines[i-1]):
                do_extract_tigress_data = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_entropy LOC=UNKNOWN \*\/", lines[i]):
                do_extract_tigress_data = False

            if encode_branches:

                if re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) __\d+_bf_\d+ LOC=UNKNOWN \*\/", lines[i-1]):
                    do_extract_tigress_data = True

                if re.search(r"\/* END FUNCTION-(DECL|DEF) __\d+_bf_\d+ LOC=UNKNOWN \*\/", lines[i]):
                    do_extract_tigress_data = False

#            if opaque:

                #if "/* BEGIN STRUCT __" + function_name in lines[i-1]: # /* BEGIN STRUCT __1_
                #    do_extract = True

            if re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) __\d+_" + function_name + "_flag_func_\d+ LOC=UNKNOWN \*\/", lines[i-1]):
                do_extract_tigress_data = True

            if re.search(r"\/* END FUNCTION-(DECL|DEF) __\d+_" + function_name + "_flag_func_\d+ LOC=UNKNOWN \*\/", lines[i]):
                do_extract_tigress_data = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_init_tigress_\d+_opaque_(ptr|list|array)_\d+ LOC=\S* \*\/", lines[i-1]):
                do_extract_tigress_data = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_init_tigress_\d+_opaque_(ptr|list|array)_\d+ LOC=\S* \*\/", lines[i]):
                do_extract_tigress_data = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_init_tigress__opaque_array LOC=\S* \*\/", lines[i-1]):
                do_extract_tigress_data = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_init_tigress__opaque_array LOC=\S* \*\/", lines[i]):
                do_extract_tigress_data = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_alwaysZero LOC=UNKNOWN \*\/", lines[i-1]):
                do_extract_tigress_data = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_alwaysZero LOC=UNKNOWN \*\/", lines[i]):
                do_extract_tigress_data = False

            if re.search(r"\/* BEGIN STRUCT __\d+_init_tigress_\d+_opaque_NodeStruct LOC=\S* \*\/", lines[i-1]):
                do_extract_tigress_data = True

            if re.search(r"\/* END STRUCT __\d+_init_tigress_\d+_opaque_NodeStruct LOC=\S* \*\/", lines[i]):
                do_extract_tigress_data = False

            if re.search(r"\/* BEGIN STRUCT __\d+_" + function_name, lines[i-1]): # /* BEGIN STRUCT __1_
                do_extract_tigress_data = True

                #if "/* END STRUCT __" + function_name in lines[i]:
                #    do_extract = False

            if re.search(r"\/* END STRUCT __\d+_" + function_name, lines[i]): # /* BEGIN STRUCT __1_
                do_extract_tigress_data = False

                #if "/* BEGIN VARIABLE-DEF __" + function_name in lines[i-1]: # problem if i = 0?
                #    do_extract = True

                #if "/* END VARIABLE-DEF __" + function_name in lines[i]:
                #    do_extract = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_" + function_name, lines[i-1]): # /* BEGIN STRUCT __1_
                do_extract_tigress_data = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_" + function_name, lines[i]): # /* BEGIN STRUCT __1_
                do_extract_tigress_data = False

            if (
                re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) .*" + function_name + r".*", lines[i-1]) and
                not re.search(r"( __bswap_(16|32|64) | __uint(16|32|64)_identity | main )", lines[i-1])
                ):
                do_extract = True
                if "/* BEGIN FUNCTION-DEF " + function_name + " LOC=UNKNOWN */" in lines[i-1]:
                    do_extract_real_function = True

            if re.search(r"\/* END FUNCTION-(DECL|DEF) .*" + function_name + r".*", lines[i]):
                do_extract = False
                do_extract_real_function = False

            if re.search(r"/* BEGIN VARIABLE-DEF iff LOC=UNKNOWN */", lines[i-1]):
                do_extract = True

            if re.search(r"/* END VARIABLE-DEF iff LOC=UNKNOWN */", lines[i]):
                do_extract = False

            if "FUNCTION-DECL-EXTERN" in lines[i-1]:
                do_extract = False
                do_extract_real_function = False
                do_extract_tigress_data = False

            #if lines[i-1] == "/* BEGIN FUNCTION-(DECL|DEF) " + function_name + " LOC=UNKNOWN */\n":
            #    do_extract = True

            #if lines[i] == "/* END FUNCTION-(DECL|DEF) " + function_name + " LOC=UNKNOWN */\n":
            #    do_extract = False

        if do_extract_tigress_data == True:
            tigress_data += lines[i]

        if do_extract_real_function == True:
            real_function += lines[i]

        if do_extract == True:
            extracted_function += lines[i]

    obfuscated_function_name = extract_function_name(real_function)

    if extract_only_helpers:
        return tigress_data

    if not extract_helpers:
       return (obfuscated_function_name, real_function)

#    extracted_function = tigress_data + extracted_function
    return (obfuscated_function_name, "\n".join(extracted_function.split("\n"))) # remove the first comment and the additional line

def extract_function2(out_file : str, function_name : str, second_function_name : str = "", is_merged : bool = False, opaque : bool = False, encode_branches : bool = False, extract_helpers : bool = True, extract_only_helpers : bool = False) -> None:
    target_file = out_file

    try:
        with open(target_file, "r") as f:
            lines = f.readlines()
    except:
        print("Error opening the file. Skipping to the next file")
        return str()

    real_function = str()
    tigress_data = str()
    extracted_function = str()
    temp = str()

    append_to_beginning = False
    do_extract = False
    do_extract_tigress_data = False
    do_extract_real_function = False
    extracted___2_bf_1 = False
    __2_bf_1_ids = []

    # read line by line until the function signature comment is found
    for i in range(len(lines)):
        if is_merged:
            if lines[i-1] == "/* BEGIN FUNCTION-DEF __1_" + function_name +  "_" + second_function_name + " LOC=UNKNOWN */\n" or lines[i] == "/* BEGIN FUNCTION-DEF __1_" + second_function_name +  "_" + function_name + " LOC=UNKNOWN */\n":
                do_extract = True

            if lines[i] == "/* END FUNCTION-DEF __1_" + function_name +  "_" + second_function_name + " LOC=UNKNOWN */\n" or lines[i] == "/* END FUNCTION-DEF __1_" + second_function_name +  "_" + function_name + " LOC=UNKNOWN */\n":
                do_extract = False

        else:
            if (
                "/* BEGIN" in lines[i-1] and ("LOC=UNKNOWN */" in lines[i-1] or "LOC=datasets/obfuscated_eval/" + function_name + "_chain_randomize_arguments_intermediate_function.c" in lines[i-1]) and
                not re.search(r"( __bswap_(16|32|64) | __uint(16|32|64)_identity | main )", lines[i-1])
                ):

                if not re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) " + function_name, lines[i-1]):
                    do_extract_tigress_data = True

                do_extract = True

            if "/* END" in lines[i] and ("LOC=UNKNOWN */" in lines[i] or "LOC=datasets/obfuscated_eval/" + function_name + "_chain_randomize_arguments_intermediate_function.c" in lines[i]):
                do_extract_tigress_data = False
                do_extract = False

            if re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) init_tigress LOC=*\S \*\/", lines[i-1]):
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END FUNCTION-(DECL|DEF) init_tigress LOC=*\S \*\/", lines[i]):
                do_extract_tigress_data = False

            if re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) __\d+_" + (function_name + "_")*2 + "FLATTEN_SPLIT_\d+ LOC=UNKNOWN \*\/", lines[i-1]):
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END FUNCTION-(DECL|DEF) __\d+_" + (function_name + "_")*2 + "FLATTEN_SPLIT_\d+ LOC=UNKNOWN \*\/", lines[i]):
                do_extract_tigress_data = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_entropy LOC=UNKNOWN \*\/", lines[i-1]):
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_entropy LOC=UNKNOWN \*\/", lines[i]):
                do_extract_tigress_data = False

            if encode_branches:

                if re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) __\d+_bf_\d+ LOC=UNKNOWN \*\/", lines[i-1]):
                    do_extract_tigress_data = True
                    do_extract = True

                if re.search(r"\/* END FUNCTION-(DECL|DEF) __\d+_bf_\d+ LOC=UNKNOWN \*\/", lines[i]):
                    do_extract_tigress_data = False

            if re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) __\d+_" + function_name + "_flag_func_\d+ LOC=UNKNOWN \*\/", lines[i-1]):
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END FUNCTION-(DECL|DEF) __\d+_" + function_name + "_flag_func_\d+ LOC=UNKNOWN \*\/", lines[i]):
                do_extract_tigress_data = False
                do_extract = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_init_tigress_\d+_opaque_(ptr|list|array)_\d+ LOC=\S* \*\/", lines[i-1]):
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_init_tigress_\d+_opaque_(ptr|list|array)_\d+ LOC=\S* \*\/", lines[i]):
                do_extract_tigress_data = False
                do_extract = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_init_tigress__opaque_array LOC=\S* \*\/", lines[i-1]):
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_init_tigress__opaque_array LOC=\S* \*\/", lines[i]):
                do_extract_tigress_data = False
                do_extract = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_alwaysZero LOC=UNKNOWN \*\/", lines[i-1]):
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_alwaysZero LOC=UNKNOWN \*\/", lines[i]):
                do_extract_tigress_data = False
                do_extract = False

            if re.search(r"\/* BEGIN STRUCT __\d+_init_tigress_\d+_opaque_NodeStruct LOC=\S* \*\/", lines[i-1]):
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END STRUCT __\d+_init_tigress_\d+_opaque_NodeStruct LOC=\S* \*\/", lines[i]):
                do_extract_tigress_data = False
                do_extract = False

            if re.search(r"\/* BEGIN STRUCT __\d+_" + function_name, lines[i-1]): # /* BEGIN STRUCT __1_
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END STRUCT __\d+_" + function_name, lines[i]): # /* BEGIN STRUCT __1_
                do_extract_tigress_data = False
                do_extract = False

            if re.search(r"\/* BEGIN VARIABLE-(DECL|DEF) __\d+_" + function_name, lines[i-1]): # /* BEGIN STRUCT __1_
                do_extract_tigress_data = True
                do_extract = True

            if re.search(r"\/* END VARIABLE-(DECL|DEF) __\d+_" + function_name, lines[i]): # /* BEGIN STRUCT __1_
                do_extract_tigress_data = False
                do_extract = False

            if re.search(r"\/* BEGIN FUNCTION-(DECL|DEF) " + function_name, lines[i-1]):
                do_extract = True
                

            if re.search(r"\/* BEGIN FUNCTION-DEF " + function_name, lines[i-1]):
               do_extract_real_function = True
               do_extract = True

            if re.search(r"\/* END FUNCTION-(DECL|DEF) " + function_name, lines[i]):
                do_extract = False
                

            if re.search(r"\/* END FUNCTION-DEF " + function_name, lines[i]):
                do_extract_real_function = False
                do_extract = False

            if re.search(r"/* BEGIN VARIABLE-DEF iff LOC=UNKNOWN */", lines[i-1]):
                do_extract = True
                

            if re.search(r"/* END VARIABLE-DEF iff LOC=UNKNOWN */", lines[i]):
                do_extract = False

            # Some special cases
            # bug causing tigress to declare __2_bf_ in case of the encode branches transformation twice which causes problems when calling tigress again since it then also adds a duplicate definition
#            if "/* BEGIN FUNCTION-DECL __2_bf_1 LOC=UNKNOWN */" in lines[i-1]:
#                do_extract = False
#                do_extract_real_function = False
#                do_extract_tigress_data = False

            if "FUNCTION-DECL-EXTERN" in lines[i-1]:
                do_extract = False
                do_extract_real_function = False
                do_extract_tigress_data = False

            #if "/* BEGIN FUNCTION-DEF __2_bf_1 LOC=UNKNOWN */" in lines[i-1]:
            #    do_extract = False
            #    do_extract_real_function = False
            #    do_extract_tigress_data = False

            if "/* BEGIN FUNCTION-DECL __2_bf_1 LOC=UNKNOWN */" in lines[i-1]:
                do_extract = False
                do_extract_tigress_data = False

            if "/* BEGIN FUNCTION-DEF __2_bf_1 LOC=UNKNOWN */" in lines[i-1]:
                do_extract = False
                do_extract_tigress_data = False
                append_to_beginning = True

            if "/* END FUNCTION-DEF __2_bf_1 LOC=UNKNOWN */" in lines[i]:
                do_extract = False
                do_extract_real_function = False
                do_extract_tigress_data = False
                append_to_beginning = False

                __2_bf_1_id = extract_function_name(temp)
                if not __2_bf_1_id in __2_bf_1_ids:
                    tigress_data = temp + tigress_data
                    extracted_function = temp + extracted_function
                    __2_bf_1_ids.append(__2_bf_1_id)

                temp = str()

        if append_to_beginning == True:
            temp += lines[i]

        if do_extract_tigress_data == True:
            tigress_data += lines[i]

        if do_extract_real_function == True:
            real_function += lines[i]

        if do_extract == True:
            extracted_function += lines[i]

    obfuscated_function_name = extract_function_name(real_function)

    if extract_only_helpers:
        return tigress_data

    if not extract_helpers:
       return (obfuscated_function_name, real_function)

    return (obfuscated_function_name, "\n".join(extracted_function.split("\n"))) # remove the first comment and the additional line

def rename_key(dictionary, old_key, new_key):
    """
    Rename a key in a dictionary.

    Parameters:
    - dictionary (dict): The input dictionary.
    - old_key: The key to be renamed.
    - new_key: The new key name.

    Returns:
    - dict: A new dictionary with the specified key renamed.
    """
    if old_key in dictionary:
        dictionary[new_key] = dictionary.pop(old_key)
    return dictionary

def count_function_parameters(source_code, function_name):
    index = cindex.Index.create()
    translation_unit = index.parse("temp.c", args=["-std=c99"], unsaved_files=[("temp.c", source_code)])

    parameter_count = 0

    for node in translation_unit.cursor.get_children():
        if node.kind == cindex.CursorKind.FUNCTION_DECL and node.spelling == function_name:
            for param in node.get_children():
                if param.kind == cindex.CursorKind.PARM_DECL:
                    parameter_count += 1

    return parameter_count

# Function to count the number of conditions and ternary operators in a C code string
def count_conditions_and_ternary_operators(code):
    # Define a regular expression pattern to match if, else if, else statements and ternary operators
    pattern = r'\b(if|else if|else|for|while)\b|\?'
    
    # Use regex to find matches in the code
    matches = re.findall(pattern, code)
    
    # Count the matches
    count = len(matches)
    
    return count



# Function to count arithmetic operations in a C code string, excluding those in strings
def count_arithmetic_operations(code):
    # Create a Clang index
    index = cindex.Index.create()

    # Parse the code and create an abstract syntax tree (AST)
    translation_unit = index.parse('temp.c', args=['-std=c99'], unsaved_files=[('temp.c', code)])

    # Function to recursively traverse the AST and count arithmetic operations
    def count_arithmetic_ops(node):
        count = 0
        for child in node.get_children():
            if child.kind == cindex.CursorKind.BINARY_OPERATOR:
                count += 1
#            elif child.kind == cindex.CursorKind.STRING_LITERAL:
                # Exclude string literals
#                continue
            count += count_arithmetic_ops(child)
        return count

    # Start counting from the root of the AST
    return count_arithmetic_ops(translation_unit.cursor)

# Function to count arithmetic operations in a C code string
def count_arithmetic_operations_re(c_code):
    # Define a regular expression pattern to match common arithmetic operators (+, -, *, /, %)
    pattern = r'[-+/%]'
    
    # Use regex to find matches in the code
    matches = re.findall(pattern, c_code)
    
    # Count the matches
    count = len(matches)
    
    return count

# Function to count all possible branches in a C code string
def count_branches(code):
    # Create a Clang index
    index = cindex.Index.create()

    # Parse the code and create an abstract syntax tree (AST)
    translation_unit = index.parse('temp.c', args=['-std=c99'], unsaved_files=[('temp.c', code)])

    # Function to recursively traverse the AST and count branches
    def count_branches_recursive(node):
        count = 0

        for child in node.get_children():
            if child.kind == cindex.CursorKind.IF_STMT:
                # If statement
                count += 1
                count += count_branches_recursive(list(child.get_children())[1])  # Count branches in 'if' block
                if len(list(child.get_children())) == 3:
                    count += count_branches_recursive(list(child.get_children())[2])  # Count branches in 'else' block if present
            elif child.kind == cindex.CursorKind.WHILE_STMT or child.kind == cindex.CursorKind.FOR_STMT:
                # While or for loop
                count += 1
                count += count_branches_recursive(list(child.get_children())[1])  # Count branches in loop body
            elif child.kind == cindex.CursorKind.CONDITIONAL_OPERATOR:
                # Ternary operator
                count += 2  # Two branches: true and false
                count += count_branches_recursive(list(child.get_children())[1])  # Count branches in true expression
                count += count_branches_recursive(list(child.get_children())[2])  # Count branches in false expression
            else:
                # For other nodes, recursively visit their children
                count += count_branches_recursive(child)

        return count

    # Start counting from the root of the AST
    return count_branches_recursive(translation_unit.cursor)

# Source from openai of the following function https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def format_obf(obfs, model_type, model, tokenizer=None, return_tokens=False):
    prompt = f"Provide the deobfuscated version of this program ```{obfs}```"
    response = f"Sure, here is the deobfuscated version of the program: ```"
    if model_type == "codegen2.5" or model_type == "codellama" or model_type == "codellama-base":
        if not tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        if return_tokens:
            return tokenizer.encode("// Obfuscated code\n" + obfs + "\n// Deobfuscated code\n")

        return "// Obfuscated code\n" + obfs + "\n// Deobfuscated code\n"

    elif model_type == "deepseek-coder-instruct":
        if not tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        messages=[
            {'role' : 'system', 'content' : 'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. '},
            {'role': 'user', 'content': prompt},
        ]

        if return_tokens:
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
            return inputs

        return tokenizer.apply_chat_template(messages, tokenize=False, return_tensors="pt")

    elif model_type == "gpt-4":
        if not tokenizer:
            import tiktoken
            tokenizer = tiktoken.encoding_for_model("gpt-4")

        messages=[
            {'role' : 'system', 'content' : 'You are a helpful assistant. '},
            {'role': 'user', 'content': prompt},
        ]

        # dummy so that len can be called later, does not contain the correct tokens since they are not needed for our purpose
        if return_tokens:
            return [[1]*num_tokens_from_messages(messages, model="gpt-4")]

        return None

def format_obf_org_pair(obfs, orig, model_type, model, tokenizer=None, return_tokens=False):
    prompt = f"Provide the deobfuscated version of this program ```{obfs}```"
    response = f"Sure, here is the deobfuscated version of the program: ```{orig}```"
    if model_type == "codegen2.5" or model_type == "codellama" or model_type == "codellama-base":
        if not tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        if return_tokens:
            return tokenizer.encode("// Obfuscated code\n" + obfs + "\n// Deobfuscated code\n" + orig + tokenizer.eos_token + "<|end|>")
            
        return "// Obfuscated code\n" + obfs + "\n// Deobfuscated code\n" + orig + tokenizer.eos_token + "<|end|>"
    
    elif model_type == "deepseek-coder-instruct":
        if not tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        
        messages=[
            {'role' : 'system', 'content' : 'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. '},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content' : response}
        ]

        if return_tokens:
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
            return inputs
        
        return tokenizer.apply_chat_template(messages, tokenize=False, return_tensors="pt")

    elif model_type == "gpt-4":
        if not tokenizer:
            import tiktoken
            tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        messages=[
            {'role' : 'system', 'content' : 'You are a helpful assistant. '},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content' : response}
        ]

        # dummy so that len can be called later, does not contain the correct tokens since they are not needed for our purpose
        if return_tokens:
            return [[1]*num_tokens_from_messages(messages, model="gpt-4")]
        
        return None

def check_token_length_limits(obfs, orig, model_list, max_token_length):
    for model_type, model in model_list:
        if len(format_obf_org_pair(obfs, orig, model_type, model, return_tokens=True)) > max_token_length:
            return False
        
    return True

def normalize_data_types(code):
#    /usr/lib/gcc/x86_64-linux-gnu/11/include/stddef.h
    code = code.replace("ssize_t", "long").replace("size_t", "long unsigned int").replace("wchar_t", "int").replace("__inline", "")
    return code

def check_sample_preconditions(sample, names):
    # check arithmetic
    error_array = [0, 0, 0, 0, 0, 0, 0]
    try:
        num_ops = count_arithmetic_operations(sample['func_def'])
        num_brns = count_branches(sample['func_def'])

    except Exception as e:

        error_array[0] = 1
        print(e)

        return error_array
    
    if num_ops < 1:
        error_array[1] = 1

    if num_brns < 1:
        error_array[2] = 1
    
    # check duplicate
    function_name_counter = names.count(sample['fname'])

    if function_name_counter > 0:
        error_array[3] = 1
    
    # check if function is main
    if sample['fname'] == "main":
        error_array[4] = 1

    # check function name length
 #   if len(sample['fname']) > 256-len(os.getcwd()):
 #       error_array[5] = 1

    # prevent samples that don't have data for semantical correctness testing but this filter is only necessary for evaluation
#       if sample['real_exe_wrapper'] == None or sample['real_io_pairs'] == None:
#        error_array[6] = 1
    
    return error_array

def convert_precondition_status_to_hr(status_code_index):
    return ["Failed to extract number of arithmetic operations and / or branches", 
            "Too little arithmetic operations", 
            "Too little branches",
            "Duplicate function name",
            "Function is main function", 
            "Function name is too long",
            "Input output wrapper or samples are missing"][status_code_index]
    
     

"""def build_program(sample : str, empty_main : bool = True, is_main : bool = False, func_def_is_external : bool = False, func_def : str = "", modified_wrapper : str >    
tigress_header = "#include \"/usr/local/bin/tigresspkg/3.3.3/tigress.h\"\n"
    if empty_main == True:
        # build the program for obfuscation with tigress

        deps = tigress_header + sample['real_deps'].replace("# 1", "")

        # include stdlib.h is required for opaque
        if not "<stdlib.h>" in deps:
            deps += "#include <stdlib.h>"

        # include pthread.h is required for concurrent dispatch flattening
        if not "<pthread.h>" in deps:
            deps += "\n#include <pthread.h>"

        if func_def_is_external:
            return deps + "\ninit_tigress(){}\n" + func_def + "\n" + ["int main(){}", ""][is_main]

        return deps + "\ninit_tigress(){}\n" + sample['func_def'] + "\n" + ["int main(){}", ""][is_main]

    else:
        # build the program for io testing

        code = ""
        code += "extern \"C\" {\n"

        deps = tigress_header + sample['real_deps'].replace("# 1", "")

        if not "<stdlib.h>" in deps:
            deps += "#include <stdlib.h>"

        # include pthread.h is required for concurrent dispatch flattening
        if not "<pthread.h>" in deps:
            deps += "\n#include <pthread.h>"


        # the following two have been added since some exebench samples require it
        if not "<unistd.h>" in deps:
            deps += "\n#include <unistd.h>"

        if not "<cstdio>" in deps:
            deps += "\n#include <cstdio>"

        # Here, the dependiencies don't need to be modified since the added libraries are only for the opaque transformations of tigress
        code += deps + "\n"

        if func_def_is_external:
            print(code.removeprefix("extern \"C\" {\n").replace("\n#include <cstdio>", "") + func_def)
            #if sample['fname'] == "IoList_sliceIndex":
            #    input()

            # cstdio breaks the clang diagnostics and the unfound library error will prevent the unresolved symbol error to show up
            fake_call = find_unresolved_symbols(code.removeprefix("extern \"C\" {\n").replace("\n#include <cstdio>", "") + func_def)
            #print(fake_call)
            #if len(fake_call) > 0:
            #    exit()
            for f in fake_call:
                if not re.search(f + r"\([^)]*\)", code.removeprefix("extern \"C\" {") + func_def):
                    continue

                print(re.search(f + r"\([^)]*\)", code.removeprefix("extern \"C\" {") + func_def).group())
                parameters = re.search(f + r"\([^)]*\)", code.removeprefix("extern \"C\" {") + func_def).group().removeprefix(f + "(")[:-1].split(",")
                print(parameters)
                bogus_parameters = []

                for parameter in parameters:
                    # direct numbers are used for the fake call
                    if parameter.strip().isdigit():
                        bogus_parameters.append("int")

                    # most likely, variables are used, theoretically chars, stings, etc. are possible but I didn't observe their occurrence
                    else:
                        if parameter != "":
                            datatype = find_variable_type_in_code(code.removeprefix("extern \"C\" {") + func_def, parameter.strip())
                            print(datatype)

                            if datatype:
                                bogus_parameters.append(datatype)

                # resolve the main conflict in the io wrapper with minimal invasiveness
                if f == "main":
                    f = "_main"

                code += "unsigned int " + f + "(" + ",".join(bogus_parameters) + "){}\n"

               # if len(parameters) > 1:
               #     input()

            code += re.sub(r"main\s*\(", "_main(", func_def)

#            if len(fake_call) > 0: exit()

        else:
            code += sample['func_def'] + "\n"

        code += "}\n\n"

#        if "__bench" in sample['real_exe_wrapper']:
#            code = re.sub(sample['fname'] + r"\s*\(", sample['fname'] + "__bench(" , code)

        if modified_wrapper == "":
            code += "\n".join(sample['real_exe_wrapper'].split("\n")[3:])
        else:
            code += "\n".join(modified_wrapper.split("\n")[3:])

        return code
"""



