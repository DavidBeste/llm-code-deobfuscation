1. Call python3 create_training_data_single.py
2. Call python3 llm.py to train a model
3. Copy and switch to the eval directory
4. run python3 eval_deobf.py to evaluate correctness and deobfuscation performance
5. run python3 show_eval.py to show the results

Example:
1. python3 create_training_data_single.py --tokenizer deepseek-ai/deepseek-coder-6.7b-instruct --max_tokens 2048 --number_of_samples 3000
2. python3 llm.py --model_type deepseek-coder-instruct --train_model deepseek-ai/deepseek-coder-6.7b-instruct --train_file datasets/obfuscation_dataset_encode_arithmetic.txt --trained_model_path models/deepseek-coder-instruct-7b-encode_arithmetic --max_tokens 2048
3. cp models/deepseek-coder-instruct-7b-encode_arithmetic eval/models/deepseek-coder-instruct-7b-encode_arithmetic; cd .. eval
# It is important to set two different directories for training and eval data to prevent source files from being overwritten!
4. python3 create_eval_data_single.py --tokenizer deepseek-ai/deepseek-coder-6.7b-instruct --max_tokens 2048 --number_of_samples 200
5. python3 llm.py --model_type deepseek-coder-instruct --eval_model models/deepseek-coder-instruct-7b-encode_arithmetic/ --eval_out_path datasets/deobfuscated --eval_file datasets/obfuscation_dataset_encode_arithmetic_eval.json --max_tokens 2048 --model_suffix _encode_arithmetic
6. python3 eval_deobf.py --eval_dataset_path datasets/obfuscation_dataset_encode_arithmetic_eval.json --original_path datasets/original --obfuscated_path datasets/obfuscated --deobfuscated_path datasets/deobfuscated_encode_arithmetic --data_suffix encode_arithmetic --io_path datasets/input_samples
7. python3 llvm.py --eval_file datasets/obfuscation_dataset_encode_arithmetic_eval.json --obfs_data_suffix encode_arithmetic --data_suffix encode_arithmetic_llvm
8. python3 show_eval.py --eval_dataset_path datasets/obfuscation_dataset_encode_arithmetic_eval.json --data_suffix _encode_arithmetic --original_path datasets/original --obfuscated_path datasets/obfuscated --deobfuscated_path datasets/deobfuscated