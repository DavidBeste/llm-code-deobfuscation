import json
import jsondiff

import clang.cindex

clang.cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-14.so.1")  # Set the path to libclang.so

#def extract_constants(cursor, source_code):
#    constants = []

#    for child in cursor.get_children():
#        extent = child.extent
#        print(f"{child.kind}: {child.spelling} and {child.location.line, child.location.column, child.location.offset, child.extent.start.offset, child.extent.end.offset}")
#        print(f"{source_code[child.extent.start.offset:child.extent.end.offset]}")
        #print(list(child.get_tokens())[0].spelling)
    #    if child.kind == clang.cindex.CursorKind.INTEGER_LITERAL:
#        constants.append(child.spelling)
    #    elif child.kind == clang.cindex.CursorKind.CALL_EXPR:
            # Recursively search for constants in function calls
#        constants.extend(extract_constants(child))

#    return constants

#def find_constants_in_function(translation_unit, source_code, function_name):
#    current_file_path = translation_unit.cursor.spelling
#    for node in translation_unit.cursor.walk_preorder():
#        if node.location.file and node.location.file.name == current_file_path:
#            print(node.kind)
#            print(node.spelling)
#            return extract_constants(node, source_code)
        #if node.kind == clang.cindex.CursorKind.FUNCTION_DECL and node.spelling == function_name:

#    return []

#def get_constants_in_c_function(source_code, function_name):
#    index = clang.cindex.Index.create()
#    unsaved_files = [("temp.c", source_code)]
#    translation_unit = index.parse("temp.c", unsaved_files=unsaved_files)
#    return find_constants_in_function(translation_unit, source_code, function_name)
import inspect
import random

import clang.cindex

clang.cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-14.so.1")  # Set the path to libclang.so

def check_output_similarity(path1, name1, path2, name2):
    score = 0
    for i in range(10):
        with open(f"{path1}/{name1}/output_{i}.json", "r") as f:
            out1 = json.load(f)

        with open(f"{path2}/{name2}/output_{i}.json", "r") as f:
            out2 = json.load(f)
        if out1 == out2:
            score += 1
#    print(jsondiff.diff(out1, out2))
    return score

def extract_macro_literals_from_c_code(c_code):
    """
    Extracts macro literals from C code using Clang.

    Args:
    c_code (str): The C code to extract macro literals from.

    Returns:
    dict: A dictionary containing macro names and their corresponding literals.
    """
    macros = {}

    # Create a Clang index
    index = clang.cindex.Index.create()

    # Parse the C code as a translation unit
    tu = index.parse('unsaved_file.c', unsaved_files=[('unsaved_file.c', c_code)])

    # Traverse the abstract syntax tree (AST)
    for node in tu.cursor.walk_preorder():
        if node.kind == clang.cindex.CursorKind.MACRO_DEFINITION:
            macro_name = node.spelling
            macro_value = c_code[node.extent.start.offset:node.extent.end.offset].split(maxsplit=2)[-1]
            macros[macro_name] = macro_value

    return macros

def extract_constants_from_c_code(c_code):
    """
    Extracts constants from C code using Clang.

    Args:
    c_code (str): The C code to extract constants from.

    Returns:
    dict: A dictionary containing lists of all constants found in the C code, categorized by type.
    """
    constants = {
        'integers': [],
        'floats': [],
        'chars': [],
        'strings': []
    }

    # Create a Clang index
    index = clang.cindex.Index.create()

    # Parse the C code as a translation unit
    tu = index.parse('unsaved_file.c', unsaved_files=[('unsaved_file.c', c_code)])

    # Traverse the abstract syntax tree (AST)
    for node in tu.cursor.walk_preorder():
        if node.location.file and node.location.file.name == 'unsaved_file.c':
            if node.kind == clang.cindex.CursorKind.INTEGER_LITERAL and c_code[node.extent.start.offset-1] != "[" and c_code[node.extent.end.offset] != "]":
                constant = c_code[node.extent.start.offset:node.extent.end.offset]
                constants['integers'].append(constant)
            elif node.kind == clang.cindex.CursorKind.FLOATING_LITERAL:
                constant = c_code[node.extent.start.offset:node.extent.end.offset]
                constants['floats'].append(constant)
            elif node.kind == clang.cindex.CursorKind.CHARACTER_LITERAL:
                constant = c_code[node.extent.start.offset:node.extent.end.offset]
                constants['chars'].append(constant)
            elif node.kind == clang.cindex.CursorKind.STRING_LITERAL:
                constant = c_code[node.extent.start.offset:node.extent.end.offset]
                constants['strings'].append(constant)

    return constants

def check_constant_minimum(c_code):
    """
    Checks if there are any constants (literals) in the provided C code.

    Args:
    c_code (str): The C code to check.

    Returns:
    bool: True if there are constants, False otherwise.
    """
    constants = extract_constants_from_c_code(c_code)
    total_constants = sum(len(values) for values in constants.values())
    return total_constants > 0

def randomize_literals(c_code, dtype, start, end):
    """
    Randomizes a literal in the C code.

    Args:
    c_code (str): The C code containing the literal.
    dtype (str): The type of the literal to randomize ('int', 'float', 'char', 'string').
    start (int): The start position of the literal in the code.
    end (int): The end position of the literal in the code.

    Returns:
    tuple: The new C code and the length difference caused by the replacement.
    """
    if dtype == 'int':
        new_literal = str(random.randint(0, 1000))
    elif dtype == 'float':
        new_literal = f"{random.uniform(0.0, 1000.0):.2f}"
    elif dtype == 'char':
        new_literal = f"{chr(random.randint(32, 126))}"
        new_literal = "'" + new_literal.replace('\\', '\\\\').replace("'", "\\'").replace('"','\\"') + "'"
    elif dtype == 'string':
        new_literal = '"' + ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(5, 15))) + '"'
    else:
        raise ValueError("Unsupported data type")

    new_c_code = c_code[:start] + new_literal + c_code[end:]
    length_diff = len(new_literal) - (end - start)
    return new_c_code, length_diff

def randomize_constants_from_c_code(c_code):
    """
    Randomizes all constants in the C code using Clang.

    Args:
    c_code (str): The C code to randomize constants from.

    Returns:
    str: The C code with all constants randomized.
    """
    # Create a Clang index
    index = clang.cindex.Index.create()

    # Parse the C code as a translation unit
    tu = index.parse('unsaved_file.c', unsaved_files=[('unsaved_file.c', c_code)])

    new_c_code = c_code
    offset_change = 0

    # Traverse the abstract syntax tree (AST)
    for node in tu.cursor.walk_preorder():
        if node.location.file and node.location.file.name == 'unsaved_file.c':
            start = node.extent.start.offset + offset_change 
            end = node.extent.end.offset + offset_change 
            if node.kind == clang.cindex.CursorKind.INTEGER_LITERAL and new_c_code[start:end].isnumeric() and new_c_code[start-1] != "[" and new_c_code[end] != "]":
                print("new_c_code before rand")
                print(new_c_code)
                print("before const")
                print(new_c_code[start-1])
                print("after const") 
                print(new_c_code[end])
                print("actual const")
                print(new_c_code[start:end])
              #  if new_c_code[start-1] == "[" and new_c_code[end] == "]": input()
                new_c_code, length_diff = randomize_literals(new_c_code, 'int', start, end)
                offset_change += length_diff

                #input()
                print("new_c_code after rand")
                print(new_c_code)

            elif node.kind == clang.cindex.CursorKind.FLOATING_LITERAL:
                print(new_c_code[start:end])
                new_c_code, length_diff = randomize_literals(new_c_code, 'float', start, end)
                offset_change += length_diff
            elif node.kind == clang.cindex.CursorKind.CHARACTER_LITERAL:
                print(new_c_code[start:end])
                new_c_code, length_diff = randomize_literals(new_c_code, 'char', start, end)
                offset_change += length_diff
            elif node.kind == clang.cindex.CursorKind.STRING_LITERAL:
                print(new_c_code[start:end])
                new_c_code, length_diff = randomize_literals(new_c_code, 'string', start, end)
                offset_change += length_diff

    print("new_c_code")
    print(new_c_code)
    #input()

    return new_c_code

