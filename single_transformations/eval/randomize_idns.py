# important !! to include all correct calls make sure to add all helper functions otherwise the id name of the call to helper functions in the actual function will not be randomized correctly
import random
import re
import subprocess
import uuid

import clang.cindex

from utils import extract_function, load_jsonl_dataset

clang.cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-14.so.1")  # Set the path to libclang.so
file_name = "test_programs/main.c"

def get_function_calls_and_decl_refs(cursor):
    function_calls = []
    if cursor.kind == clang.cindex.CursorKind.CALL_EXPR or cursor.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
        function_calls.append(cursor.spelling)
    for child in cursor.get_children():
        function_calls.extend(get_function_calls_and_decl_refs(child))
    return function_calls

def get_function_declarations(node, translation_unit):
    functions = []
    if node.kind == clang.cindex.CursorKind.FUNCTION_DECL and node.location.file is not None and node.location.file.name == translation_unit.spelling:
        functions.append(node.spelling)
    for child in node.get_children():
        functions.extend(get_function_declarations(child, translation_unit))
    return functions

def randomize_function_names(code):
    # Create an Index
    index = clang.cindex.Index.create()

    # Parse the code
    translation_unit = index.parse('tmp.c', unsaved_files=[('tmp.c', code)])
    function_names = get_function_declarations(translation_unit.cursor, translation_unit)
    if len(function_names) == 0:
        return code

    for function_name in function_names:
        name = str(uuid.uuid4())[:8]
        while name[0].isdigit():
            name = str(uuid.uuid4())[:8]

        obfuscated_program = re.sub(function_name + r"\s*\(", "__RND__" + name + "(", code)

    return obfuscated_program

def generate_random_idn():
    """
    Generate a list of random variable names.

    Parameters:
    - num_names (int): The number of random variable names to generate.

    Returns:
    - list: A list of random variable names.
    """
    
    name = str(uuid.uuid4())[:8]
    # Ensure the name doesn't start with a number
    while name[0].isdigit():
        name = str(uuid.uuid4())[:8]
    
    return name

def generate_random_id_names(num_names):
    """
    Generate a list of random ID names.

    Parameters:
    - num_names (int): The number of random ID names to generate.

    Returns:
    - list: A list of random ID names.
    """
    random_id_names = [str(uuid.uuid4())[:8] for _ in range(num_names)]
    return random_id_names

def is_variable_in_struct(variable_cursor):
    """
    Check if a variable declaration is inside a struct.

    Parameters:
    - variable_cursor (clang.cindex.Cursor): The cursor representing the variable declaration.

    Returns:
    - bool: True if the variable is inside a struct, False otherwise.
    """

    # Get the parent cursor
    parent_cursor = variable_cursor.semantic_parent

    # Check if the parent is a struct declaration
    if parent_cursor.kind == clang.cindex.CursorKind.STRUCT_DECL:
        return True
    elif parent_cursor.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
        return False
    else:
        return is_variable_in_struct(parent_cursor)

def get_struct_parents(variable_cursor, structs):
    """
    Check if a variable declaration is inside a struct.

    #Parameters:
    - variable_cursor (clang.cindex.Cursor): The cursor representing the variable declaration.

    #Returns:
    - bool: True if the variable is inside a struct, False otherwise.
    """

    # Get the parent cursor
    parent_cursor = variable_cursor.semantic_parent
    # Check if the parent is a struct declaration
    if parent_cursor.kind == clang.cindex.CursorKind.STRUCT_DECL:
        structs.append(parent_cursor.spelling)
        return get_struct_parents(parent_cursor,structs)

    elif parent_cursor.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
        return structs
    else:
        return get_struct_parents(parent_cursor,structs)

def get_identifier_names(c_code, ignore_function_declarations=True, function_name=""):
    """
    Extracts and returns a list of all identifier names from C code.

    Parameters:
    - c_code (str): The C code as a string.

    Returns:
    - list: A list of identifier names.
    """
    # Create an Index
    index = clang.cindex.Index.create()

    # Parse the code
    translation_unit = index.parse('tmp.c', unsaved_files=[('tmp.c', c_code)])

    # get function declarations to check if the identifier name is actually from a custom implemented function or some standard function like printf
    function_declarations = get_function_declarations(translation_unit.cursor, translation_unit)

    # Function to recursively extract identifier names
    def extract_identifiers(node):
        identifiers = []
        #labels = []
        if node.kind.is_declaration(): #or node.kind.is_expression():

            if node.spelling and node.location.file is not None and node.location.file.name == translation_unit.spelling:
                if (
                    node.kind not in [clang.cindex.CursorKind.STRING_LITERAL, clang.cindex.CursorKind.INTEGER_LITERAL] and
                    (node.kind != clang.cindex.CursorKind.FUNCTION_DECL or (ignore_function_declarations == False and (function_name == "" or node.spelling != function_name))) and
                    "__RND__" not in node.spelling #and # exclude already randomized ids
                    #node.spelling not in ["main, init_tigress"] # and
#                    (node.kind in [clang.cindex.CursorKind.CALL_EXPR, clang.cindex.CursorKind.DECL_REF_EXPR] and node.referenced.spelling in function_declarations)
                ):
                    
                    if is_variable_in_struct:
                        structs = []
                        get_struct_parents(node, structs)
                        identifiers.append("::".join(structs) + "::" + node.spelling)

                    else:
                        identifiers.append(node.spelling)

        for child in node.get_children():
            identifiers.extend(extract_identifiers(child))
        return identifiers

    def extract_labels(node):
        labels = []
        if node.spelling and node.location.file is not None and node.location.file.name == translation_unit.spelling:
            if node.kind in [clang.cindex.CursorKind.LABEL_REF, clang.cindex.CursorKind.LABEL_STMT]:
                if node.spelling not in labels:
                    labels.append(node.spelling)

        for child in node.get_children():
            labels.extend(extract_labels(child))
        return labels

    # Extract identifiers from the translation unit
    identifiers = extract_identifiers(translation_unit.cursor)
    labels = extract_labels(translation_unit.cursor)

    return identifiers, labels

# don't use a numbering because higher numbers in var_x might indicate aded helper structs so the model learns to remove the vars with the highest count
def create_random_idn_mapping(code, identifiers):
    random_mapping = {}
    for identifier in identifiers:
        new_id = generate_random_idn()
        # prevent collision
        while f"__RND__{new_id}" in random_mapping.values() or f"__RND__{new_id}" in code:
            new_id = generate_random_idn()

        random_mapping[identifier] = f"__RND__{new_id}"

    return random_mapping

def regex_replace(code, rnd_idns, do_exclude_seps=False):
    new_code = code
    if not do_exclude_seps:
        exclude_seps = ""
    else:
        exclude_seps = r"(?<!\/\/ Obfuscated )(?<!\/\/ Deobfuscated )"

    for old, new in rnd_idns.items():
        if "::" in old:
            #print(r"\b" + old.split("::")[-1] + r"\b", new)
            #print("re 1", r"\b" + old.split("::")[-1] + r"\b")
            new_code = re.sub(exclude_seps + r"\b" + old.split("::")[-1] + r"\b", new, new_code)
        else:
            #print(r"\b" + old + r"\b", new)
            #print("re 2", r"\b" + old + r"\b")
            new_code = re.sub(exclude_seps + r"\b" + old + r"\b", new, new_code)

    #print("original code")
    #print(code)
    #print("before")
    #print(new_code)
    #print("after")
    #input()
    return new_code

def randomize_identifiers(filename, identifier_names=None, labels=None, ignore_func_decls=True, function_name=""):
    with open(filename, "r") as f:
        code = f.read()

    if identifier_names == None or labels == None:
        identifier_names, labels = get_identifier_names(code, ignore_function_declarations=ignore_func_decls, function_name=function_name)

    randomized_identifiers = create_random_idn_mapping(code, identifier_names)
    randomized_labels = create_random_idn_mapping(code, labels)

    rename_command = ["clang-rename-14"]

    for old, new in randomized_identifiers.items():
        rename_command.append(f"--qualified-name={old}")
        rename_command.append(f"--new-name={new}")

    rename_command.append(filename)

    # trivial case with zero identifier names to change
    if len(rename_command) <= 2:
        return code

    new_code = subprocess.run(rename_command, capture_output=True, text=True)

    if new_code.stdout == "":
        return regex_replace(regex_replace(code, randomized_identifiers), randomized_labels)

    return regex_replace(regex_replace(new_code.stdout, randomized_identifiers), randomized_labels)

def randomize_identifiers2(filename_with_helpers, filename_without_helpers, identifier_names=None, labels=None, ignore_func_decls=True, function_name=""):
    with open(filename_with_helpers, "r") as f:
        code = f.read()

    if identifier_names == None or labels == None:
        identifier_names, labels = get_identifier_names(code, ignore_function_declarations=ignore_func_decls, function_name=function_name)

    randomized_identifiers = create_random_idn_mapping(code, identifier_names)
    randomized_labels = create_random_idn_mapping(code, labels)

    rename_command = ["clang-rename-14"]

    for old, new in randomized_identifiers.items():
        rename_command.append(f"--qualified-name={old}")
        rename_command.append(f"--new-name={new}")

    rename_command.append(filename_without_helpers)

    # trivial case with zero identifier names to change
    if len(rename_command) <= 2:
        return code

    with open(filename_without_helpers, "r") as f:
        code_without_helpers = f.read()

    new_code = subprocess.run(rename_command, capture_output=True, text=True)

    if new_code.stdout == "":
        return regex_replace(regex_replace(code_without_helpers, randomized_identifiers), randomized_labels)

    return regex_replace(regex_replace(new_code.stdout, randomized_identifiers), randomized_labels)

def randomize_training_data(dataset_path):
    transformations = ["encode_arithmetic", "encode_branches", "flatten", "opaque", "randomize_arguments"]
    dataset = load_jsonl_dataset(dataset_path)

    for sample in dataset:
        function_name = list(sample.keys())[0].split("__name__")[1]
        _, function_definition = extract_function(f"datasets/original/{function_name}_tigress_canonicalized.c", function_name)
#        print(function_definition)

        with open(f"datasets/original/{function_name}_tigress_canonicalized_function.c", "w") as f:
            f.write(function_definition)

        new_function_definition = randomize_identifiers(f"datasets/original/{function_name}_tigress_canonicalized_function.c")
#        print(new_function_definition)
        with open(f"datasets/original/{function_name}_rnd_function.c", "r") as f:
            original_randomized = f.read()

#        print(original_randomized)
#        continue
        for transformation in transformations:
            _, function_definition_obfs = extract_function(f"datasets/obfuscated/{function_name}_{transformation}.c", function_name)
            #print(function_definition_obfs)
            #input()

            with open(f"datasets/obfuscated/{function_name}_{transformation}_function.c", "w") as f:
                f.write(function_definition_obfs)

        #print(new_function_definition)

def post_process(code):
    index = clang.cindex.Index.create()
#    translation_unit = index.parse(file_path)
    translation_unit = index.parse('tmp.c', unsaved_files=[('tmp.c', code)])
    func_calls_and_decl_refs = get_function_calls_and_decl_refs(translation_unit.cursor)
#    print(func_calls_and_decl_refs)
    remaining_identifier_names = []
    for func_call_or_decl_ref in func_calls_and_decl_refs:
        if not "__RND__" in func_call_or_decl_ref:
            remaining_identifier_names.append(func_call_or_decl_ref)

#    print(remaining_identifier_names)
    # Remove empty id names that were added for whatever reason, this causes the code to be broken for some samples 
    remaining_identifier_names = list(filter(None, remaining_identifier_names))

#    input()
    randomized_identifiers = create_random_idn_mapping(code, remaining_identifier_names)
    return regex_replace(code, randomized_identifiers, do_exclude_seps=True)

def main():
    # Example usage
    c_code = """
    #include <stdio.h>

    int add(int a, int b) {
        int result;
        result = a + b;
        return result;
    }

    /*int main() {
        int x = 5;
        int y = 10;
        int sum = add(x, y);
        printf("Sum: %d\\n", sum);
        return 0;
    }*/
    """

    rd = random.Random()
    rd.seed(0)

    u = uuid.UUID(int=rd.getrandbits(128))
#    print(u)

#    randomize_identifiers(file_name)

    randomize_training_data("obfuscation_dataset_encode_arithmetic_ext2_no_basic.json")

    """identifier_names = get_identifier_names(code)
    print(identifier_names)
    randomized_identifiers = create_random_idn_mapping(identifier_names)

    rename_command = ["clang-rename-14"]

    for old, new in randomized_identifiers.items():
        rename_command.append(f"--qualified-name={old}")
        rename_command.append(f"--new-name={new}")

    rename_command.append(file_name)

    print(rename_command)
    subprocess.run(rename_command)"""

if __name__ == "__main__":
    main()
