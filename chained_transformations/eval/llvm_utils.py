from math import log2

from llvmlite import ir
from llvmlite import binding as llvm

# Function to check if an instruction is a terminator
def is_terminator(instruction):
    terminator_opcodes = {'ret', 'br', 'switch', 'indirectbr', 'invoke', 'unwind', 'resume', 'unreachable'}
    return instruction.opcode in terminator_opcodes

# Function to calculate McCabe's Cyclomatic Complexity
def calculate_cyclomatic_complexity(func):
    complexity = 1
    for block in func.blocks:
        num_branches = sum(1 for instr in block.instructions if is_terminator(instr))
        complexity += num_branches
    return complexity

"""# Function to calculate Halstead metrics
def calculate_halstead_metrics_for_globals(module):
    operators = set()
    operands = set()
    for global_var in module.global_variables:
        print(global_var)

        # Analyze global variable instructions (if applicable)
        print(global_var.name)
        print(global_var.__dir__())

        for instr in global_var.initializer.instructions:
            if instr.opcode:
                operators.add(instr.opcode)
            for op in instr.operands:
                if isinstance(op, ir.Instruction) and op.opcode:
                    operators.add(op.opcode)
                else:
                    operands.add(op.name)

    n1 = len(operators)  # distinct operators
    n2 = len(operands)   # distinct operands
    print(operators)
    N1 = len([instr for global_var in module.global_variables for instr in global_var.initializer.instructions if instr.opcode])  # total operators
    N2 = len([instr for global_var in module.global_variables for instr in global_var.initializer.instructions if isinstance(instr, ir.Instruction)])  # total operands

    return n1, n2, N1, N2"""

"""# Function to calculate Halstead metrics for global variables
def calculate_halstead_metrics_for_globals(module):
    operators = set()
    operands = set()

    # Iterate over all functions in the module
    for func in module.functions:
        # Iterate over the basic blocks and instructions in each function
        for block in func.basic_blocks:
            for instr in block:
                if isinstance(instr, ir.LoadInstr) and isinstance(instr.operands[0], ir.GlobalVariable):
                    # Consider the global variable used in the Load instruction
                    global_var = instr.operands[0]
                    # Analyze the global variable instructions
                    for init_instr in global_var.initializer.instructions:
                        if init_instr.opcode:
                            operators.add(init_instr.opcode)
                        for op in init_instr.operands:
                            if isinstance(op, ir.Instruction) and op.opcode:
                                operators.add(op.opcode)
                            else:
                                operands.add(op.name)

    n1 = len(operators)  # distinct operators
    n2 = len(operands)   # distinct operands
    N1 = len([instr for instr in module.global_variables for init_instr in instr.initializer.instructions if init_instr.opcode])  # total operators
    N2 = len([instr for instr in module.global_variables for init_instr in instr.initializer.instructions if isinstance(init_instr, ir.Instruction)])  # total operands

    return n1, n2, N1, N2"""

# Function to calculate Halstead metrics
def calculate_halstead_metrics(func):
    operators = []
    operands = []
    for block in func.blocks:
        for instr in block.instructions:
            if instr.opcode:
                operators.append(instr.opcode)
#                print(instr.opcode)

            else:
                print("Not an instr.opcode : " + instr)
                exit(1)

            for op in instr.operands:
                if isinstance(op, ir.Instruction) and op.opcode:
                    operators.append(op.opcode)
                else:
                    operands.append(op.name)

    n1 = len(set(operators))  # distinct operators
    n2 = len(set(operands))   # distinct operands
#    N1 = sum(1 for instr in func.body if instr.opcode)  # total operators
#    N2 = sum(1 for instr in func.body if isinstance(instr, ir.Instruction))  # total operands
#    N1 = len([instr for block in func.blocks for instr in block.instructions if instr.opcode])  # total operators
#    N2 = len([instr for block in func.blocks for instr in block.instructions if isinstance(instr, ir.Instruction)])  # total operands
#    N2 = len([instr for block in func.blocks for instr in block.instructions if isinstance(instr, ir.Instruction)])  # total operands
    N1 = len(operators)
    N2 = len(operands)
    vocabulary = n1 + n2
    program_length = N1 + N2
    program_volume = program_length * log2(vocabulary)
    difficulty = (n1 / 2) * (N2 / n2) if n2 != 0 else -1
    effort = difficulty * program_volume
    time = effort / 18
    bugs = effort / 3000

    return n1, n2, N1, N2, vocabulary, program_length, program_volume, difficulty, effort, time, bugs

"""# Function to calculate Halstead metrics
def calculate_halstead_metrics(func):
    operators = set()
    operands = set()
    for block in func.blocks:
        for instr in block.instructions:
            if instr.opcode:
                operators.add(instr.opcode)
                print(instr.opcode)

            for op in instr.operands:
                if isinstance(op, ir.Instruction) and op.opcode:
                    operators.add(op.opcode)
                else:
                    operands.add(op.name)

    n1 = len(operators)  # distinct operators
    n2 = len(operands)   # distinct operands
#    N1 = sum(1 for instr in func.body if instr.opcode)  # total operators
#    N2 = sum(1 for instr in func.body if isinstance(instr, ir.Instruction))  # total operands
    N1 = len([instr for block in func.blocks for instr in block.instructions if instr.opcode])  # total operators
#    N2 = len([instr for block in func.blocks for instr in block.instructions if isinstance(instr, ir.Instruction)])  # total operands
    N2 = len([instr for block in func.blocks for instr in block.instructions if isinstance(instr, ir.Instruction)])  # total operands
    vocabulary = n1 + n2
    program_length = N1 + N2
    program_volume = program_length * log2(vocabulary)
    difficulty = (n1 / 2) * (N2 / n2) if n2 != 0 else -1
    effort = difficulty * program_volume
    time = effort / 18
    bugs = effort / 3000

    return n1, n2, N1, N2, vocabulary, program_length, program_volume, difficulty, effort, time, bugs"""

"""if __name__ == "__main__":
    # Load the LLVM IR from a file
    with open("datasets/obfuscated_eval/add_underscore_encode_branches_function.ll", "r") as ir_file:
        ir_code = ir_file.read()

    # Initialize the LLVM target and data layout (needed for parsing the IR)
    llvm.initialize()
    llvm.initialize_all_targets()

    # Create an LLVM context and module
    llvm_context = llvm.get_global_context()
    module = llvm.parse_assembly(ir_code)

#    print(calculate_halstead_metrics_for_globals(module))


    # Find and analyze functions in the module
    for func in module.functions:
        if not func.is_declaration:
            print(f"Function: {func.name}")
            complexity = calculate_cyclomatic_complexity(func)
            n1, n2, N1, N2, vocabulary, program_length, program_volume, difficulty, effort, time, bugs = calculate_halstead_metrics(func)
            print(f"Cyclomatic Complexity: {complexity}")
            print(f"Halstead Metrics:")
            print(f"  n1 (Distinct Operators): {n1}")
            print(f"  n2 (Distinct Operands): {n2}")
            print(f"  N1 (Total Operators): {N1}")
            print(f"  N2 (Total Operands): {N2}")
            print(f"  Vocabulary: {vocabulary}")
            print(f"  Program Length: {program_length}")
            print(f"  Program Volume: {program_volume}")
            print(f"  Difficulty: {difficulty}")
            print(f"  Effort: {effort}")
            print(f"  Time: {time}")
            print(f"  Bugs: {bugs}")"""
