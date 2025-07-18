# Exploring the Potential of LLMs for Code Deobfuscation

[Dataset Link](https://zenodo.org/records/15831457)

## Instructions to reproduce

### Single Transformations

1. Create the training dataset
2. Fine-tune the model
3. Copy and switch to the eval directory
4. Create the evaluation dataset
5. Evaluate the model
6. Build the eval files based on the LLM output
7. Evaluate Clang
8. Evaluate correctness and compute the metrics
9. Show the evaluation

#### Example:
1. ```cd single_transformations/train; python3 create_training_data_single.py --tokenizer deepseek-ai/deepseek-coder-6.7b-instruct --max_tokens 6144 --number_of_samples 3000```
2. ```python3 llm.py --model_type deepseek-coder-instruct --train_model deepseek-ai/deepseek-coder-6.7b-instruct --train_file datasets/obfuscation_dataset_encode_arithmetic_6144.txt --trained_model_path models/deepseek-coder-instruct-7b-encode_arithmetic --max_tokens 6144```
3. ```mkdir ../eval/models ../eval/datasets; cp -r models/deepseek-coder-instruct-7b-encode_arithmetic ../eval/models/deepseek-coder-instruct-7b-encode_arithmetic; cd ../eval # It is important to set two different directories for training and eval data to prevent source files from being overwritten!```
4. ```python3 create_eval_data_single.py --tokenizer deepseek-ai/deepseek-coder-6.7b-instruct --max_tokens 6144 --number_of_samples 200```
5. ```python3 llm.py --model_type deepseek-coder-instruct --eval_model models/deepseek-coder-instruct-7b-encode_arithmetic/ --eval_out_path datasets/deobfuscated --eval_file datasets/obfuscation_dataset_encode_arithmetic_6144_eval.json --max_tokens 6144 --data_suffix _encode_arithmetic```
6. ```python3 llm.py --model_type deepseek-coder-instruct --eval_model models/deepseek-coder-instruct-7b-encode_arithmetic/ --eval_out_path datasets/deobfuscated --eval_file datasets/obfuscation_dataset_encode_arithmetic_6144_eval.json --max_tokens 6144 --obfs_data_suffix _encode_arithmetic --data_suffix _encode_arithmetic --build_eval_files 1```
7. ```python3 llvm.py --eval_file datasets/obfuscation_dataset_encode_arithmetic_6144_eval.json --obfs_data_suffix _encode_arithmetic --data_suffix _encode_arithmetic```
8. ```python3 eval_deobf.py --eval_dataset_path datasets/obfuscation_dataset_encode_arithmetic_6144_eval.json --original_path datasets/original --obfuscated_path datasets/obfuscated --deobfuscated_path datasets/deobfuscated_encode_arithmetic --data_suffix encode_arithmetic --io_path datasets/input_samples```
9. ```python3 show_eval.py --eval_dataset_path datasets/obfuscation_dataset_encode_arithmetic_6144_eval.json --data_suffix _encode_arithmetic --original_path datasets/original --obfuscated_path datasets/obfuscated --deobfuscated_path datasets/deobfuscated```

### Chained Transformations

1. Create the training dataset
2. Fine-tune the model
3. Copy and switch to the eval directory
4. Create the evaluation dataset
5. Evaluate the model
6. Build the evaluation files around the LLM-generated samples
7. Evaluate clang
8. Evaluate correctness and compute the metrics
9. Show the evaluation

#### Example:

1. ```python3 create_training_data_chain.py --tokenizer deepseek-ai/deepseek-coder-6.7b-instruct --max_tokens 6144 --chain_length 1 --number_of_samples 3000```
2. ```python3 llm.py --model_type deepseek-coder-instruct --train_model deepseek-coder-instruct-7b-chain-6144 --train_file datasets/obfuscation_dataset_chain_6144_all_training.txt --trained_model_path models/deepseek-coder-instruct-7b-chain-6144 --max_tokens 6144```
3. ```cp -r models/deepseek-coder-instruct-7b-chain-6144 eval/models/deepseek-coder-instruct-7b-chain-6144; cd ../eval```
4. ```python3 create_eval_data_chain.py --tokenizer deepseek-ai/deepseek-coder-6.7b-instruct --max_tokens 6144 --chain_length 1 --number_of_samples 1000```
5. ```python3 llm.py --model_type deepseek-coder-instruct --eval_model models/deepseek-coder-instruct-7b-chain-6144 --eval_out_path datasets/deobfuscated --eval_file datasets/obfuscation_dataset_1_chain_eval2_l.json --max_tokens 6144 --obfs_data_suffix _1_chain --data_suffix _1_chain_6144```
6. ```python3 llm.py --model_type deepseek-coder-instruct --eval_model models/deepseek-coder-instruct-7b-chain-6144 --eval_out_path datasets/deobfuscated --eval_file datasets/obfuscation_dataset_1_chain_eval2_l.json --max_tokens 6144 --obfs_data_suffix _1_chain --data_suffix _1_chain_6144 --build_eval_files 1```
7. ```python3 llvm.py --eval_file datasets/obfuscation_dataset_1_chain_eval2_l.json --orig_data_suffix _1 --obfs_data_suffix _1_chain --data_suffix _1_chain_6144;python3 llvm.py --eval_file datasets/obfuscation_dataset_2_chain_eval2_l.json --orig_data_suffix _2 --obfs_data_suffix _2_chain --data_suffix _2_chain_6144```
8. ```python3 eval_deobf.py --eval_dataset_path datasets/obfuscation_dataset_1_chain_eval2_l.json --no_metrics --original_path datasets/original --obfuscated_path datasets/obfuscated --deobfuscated_path datasets/deobfuscated --orig_data_suffix _1 --obfs_data_suffix _1_chain --data_suffix _1_chain_6144 --io_path datasets/input_samples```
9. ```python3 show_eval.py --eval_dataset_path datasets/obfuscation_dataset_1_chain_eval2_l.json --orig_data_suffix _1 --obfs_data_suffix _1_chain --data_suffix _1_chain_6144 --original_path datasets/original --original_io_path datasets/original_eval --obfuscated_path datasets/obfuscated --deobfuscated_path datasets/deobfuscated```

### Memorization

1. Build the memorization dataset
2. Evaluate the model with the memorized samples
3. Build evaluation files around the LLM generated samples
4. Evaluate correctness and compute the metrics
5. Show the evaluation
6. Manually check for memorized constants (We only need the correctness part of the evaluation here since we only have to examine semantically incorrect samples and don't need the deobfuscation performance)

#### Example: 

1. ```python3 build_memorization_dataset.py --input_dataset ../FineTuning-6144-eval/datasets/obfuscation_dataset_encode_arithmetic_6144_eval.json --output_dataset _gpt-4-32k-0314-memorization --original_path ../FineTuning-6144-eval/datasets/original --obfuscated_path ../FineTuning-6144-eval/datasets/obfuscated --deobfuscated_path ../FineTuning-6144-eval/datasets/deobfuscated --data_suffix _encode_arithmetic_gpt-4-32k-0314,_encode_branches_gpt-4-32k-0314,_flatten_gpt-4-32k-0314,_opaque_gpt-4-32k-0314,_randomize_arguments_gpt-4-32k-0314```
2. ```python3 llm.py --model_type openai --eval_model ../FineTuning-6144-eval/models/codellama-7b-encode_arithmetic-6144/ --eval_out_path MemTest/deobfuscated_modified --eval_file obfuscation_dataset_gpt-4-32k-0314-memorization_encode_arithmetic.json --max_tokens 6144 --data_suffix _encode_arithmetic_gpt-4-32k-0314```
3. ```python3 llm.py --model_type openai --eval_model ../FineTuning-6144-eval/models/codellama-7b-encode_arithmetic-6144/ --eval_out_path MemTest/deobfuscated_modified --eval_file obfuscation_dataset_gpt-4-32k-0314-memorization_encode_arithmetic.json --obfs_data_suffix _encode_arithmetic --max_tokens 6144 --data_suffix _encode_arithmetic_gpt-4-32k-0314 --build_eval_files 1```
4. ```python3 eval_deobf.py --eval_dataset_path obfuscation_dataset_gpt-4-32k-0314-memorization_encode_arithmetic.json --original_path MemTest/original_modified --obfuscated_path MemTest/obfuscated_modified --deobfuscated_path MemTest/deobfuscated_modified --obfs_data_suffix _encode_arithmetic --data_suffix _encode_arithmetic_gpt-4-32k-0314 --io_path ../FineTuning-6144-eval/datasets/input_samples```
5. ```python3 show_eval.py --eval_dataset_path obfuscation_dataset_gpt-4-32k-0314-memorization_encode_arithmetic.json --original_path MemTest/original_modified --obfuscated_path MemTest/obfuscated_modified --deobfuscated_path MemTest/deobfuscated_modified --obfs_data_suffix _encode_arithmetic --data_suffix _encode_arithmetic_gpt-4-32k-0314```
