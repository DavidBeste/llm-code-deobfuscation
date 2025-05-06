import argparse
import json
import os
import re

from datasets import Dataset, load_dataset
from openai import OpenAI
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import transformers

from utils import (
    build_program,
    create_folder_if_not_exists,
    extract_function,
    extract_function_name,
    get_function_parameters_ex,
    format_obf,
    format_obf_org_pair,
    load_jsonl_dataset
)
from utils_eval import correct_function_arguments, extract_argument_names_2, get_permutations

torch.set_printoptions(profile="full")

deepseek_chat_template = ""

# Example: python3 llm.py --model_type deepseek-coder-instruct --eval_model models/deepseek-coder-instruct-7b-encode_arithmetic/ --eval_out_path datasets/deobfuscated --eval_file datasets/obfuscation_dataset_encode_arithmetic_eval.json --max_tokens 2048 --model_suffix _encode_arithmetic
# Example: python3 llm.py --model_type deepseek-coder-instruct --eval_model models/deepseek-coder-instruct-7b-encode_arithmetic/ --eval_out_path datasets/deobfuscated_encode_arithmetic/ --eval_file datasets/obfuscation_dataset_encode_arithmetic_eval.json --max_tokens 2048
# Example: python3 llm.py --model_type codegen --eval_model codegen25_7B_multi_encode_arithmetic_ext_no_basic_lora_1 --eval_out_path datasets/deobfuscated_encode_arithmetic --eval_file 
# Example: python3 llm.py --model_type deepseek-coder-instruct --train_model deepseek-ai/deepseek-coder-6.7b-instruct --train_file datasets/obfuscation_dataset_encode_arithmetic.txt --trained_model_path models/deepseek-coder-instruct-7b-encode_arithmetic --max_tokens 2048

# define parser and arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', choices=['deepseek-coder-instruct', 'codellama', 'codegen', 'openai'], required=True)

parser.add_argument('--train_model')
parser.add_argument('--train_file', help='Run the training file')
parser.add_argument('--trained_model_path')

parser.add_argument('--eval_model')
parser.add_argument('--eval_file', help='Run the evaluation file')
parser.add_argument('--eval_out_path')
parser.add_argument('--build_eval_files')

parser.add_argument('--max_tokens', type=int)
parser.add_argument('--obfs_data_suffix', default='')
parser.add_argument('--data_suffix', default='')

#subparsers = parser.add_subparsers()
#subparsers.add_parser("train", help='Run the training file')

args = parser.parse_args()

def get_model(model_id):
#    training_data = load_jsonl_dataset(args.train_file) if args.train_file else None
#    evaluation_data = load_jsonl_dataset(args.eval_file) if args.eval_file else None
#    data = (training_data, evaluation_data)

    llms = {
        "deepseek-coder-instruct": DeepSeekLLM(model_id),
        "codellama": CodeLlamaLLM(model_id),
        "codegen": Codegen25LLM(model_id),
        "openai" : OpenAILLM(model_id)
    }

    # select model based on CLI input
    selected_model = llms.get(args.model_type)

    return selected_model

def prepare_training_dataset(training_dataset_path, model_type, model_name, tokenizer):
    with open(training_dataset_path, "r") as f:
        content = f.read()

    samples = content.split("<|OBFS|>\n") # after the call to split we have the pair of obfuscated and original sample for each item
    samples = samples[1:] # remove the empty element that has been created by split, CHECK IF STILL CORRECT SINCE WE SPLIT DIFFERENT NOW

#    for i in range(len(samples)):
#        obfs, orig = samples[i].split("<|ORIG|>") 
#        samples[i] = format_obf_org_pair(obfs, orig, "deepseek-coder-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct")

    print(f"Sample count before {len(samples)}")
    samples = [sample for sample in samples if len(format_obf_org_pair(*sample.split("<|ORIG|>"), model_type=model_type, model=model_name, tokenizer=tokenizer, return_tokens=True)) <= args.max_tokens]
    print(f"Sample count after {len(samples)}")

    samples = [format_obf_org_pair(*sample.split("<|ORIG|>"), model_type=model_type, model=model_name, tokenizer=tokenizer) for sample in samples]

    d = {'prediction' : samples}
    dataset = Dataset.from_dict(d)
    dataset = dataset.map(lambda samples: tokenizer(samples['prediction']), batched=True)
    
#    for i in range(dataset['prediction']):
#        dataset['prediction'][i] = dataset['prediction'][i] 

    return dataset

def prepare_eval_dataset(eval_dataset_path, model_type, model_name, tokenizer):
    samples = load_jsonl_dataset(eval_dataset_path)

#    print(f"Sample count before {len(samples)}")
#    samples = [sample for sample in samples if len(format_obf_org_pair(*sample.split("<|ORIG|>"), model_type=model_type, model=model_name, tokenizer=tokenizer, return_tokens=True)) <= args.max_tokens]
#    print(f"Sample count after {len(samples)}")

    print(samples[0])
  #  samples = [format_obf_org_pair(*sample.split("<|ORIG|>"), model_type=model_type, model=model_name, tokenizer=tokenizer) for sample in samples]
    samples = [{key: value.split("<|ORIG|>")[0].split("<|OBFS|>\n")[1]} for sample in samples for key, value in sample.items()]
    print(samples[0])
    samples = [{key: tokenizer(format_obf(value, model_type=model_type, model=model_name, tokenizer=tokenizer), return_tensors='pt').to("cuda:0")} for sample in samples for key, value in sample.items()]

    print(samples[0])
    return samples

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_code_from_response(response, model_type):
    if model_type == "deepseek-coder-instruct":
        print(response)
        return response.split("Sure, here is the deobfuscated version of the program: ```")[1].split("```")[0]

    elif model_type == "openai":
        print(json.loads(response)['choices'][0]['message']['content'])
        return json.loads(response)['choices'][0]['message']['content'].replace("```", "")
    
    elif model_type == "codellama":
        return response.split("// Deobfuscated code")[1].replace("<s>", "").replace("</s>", "")

# also the outputs are kept as fp32
class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

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
#            return True                                                                                                                                                                                         #        return False

# changed so that the model will only stop when reaching // Vulnerable code and not just //
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        counter = 0
        for i in range(len(self.keywords)):
            if input_ids[0][i-len(self.keywords)] != self.keywords[i]:
                return False

        return True

class LLMBase:
    def __init__(self, model_id):
        self.model = None
        self.model_id = model_id

    def train(self, dataset, dir_name):
        raise NotImplementedError

    def evaluate(self, dataset, dir_name):
        raise NotImplementedError

class LocalLLM(LLMBase):
    def __init__(self, model_id, chat_template = None):
        super().__init__(model_id)
        self.tokenizer = None

    def _load_models(self):
        if not self.model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, trust_remote_code=True).cuda()

    def _load_loras(self):
        self.config = PeftConfig.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
        self.model = PeftModel.from_pretrained(self.model, self.model_id)
        

    def train(self, dataset, dir_name):
        self._load_models()
        for param in self.model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        self.model.gradient_checkpointing_enable()  # reduce number of stored activations
        self.model.enable_input_require_grads()

        self.config = LoraConfig(
            r=16, #attention heads
            lora_alpha=32, #alpha scaling
            target_modules=["q_proj", "v_proj"], #if you know the
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
        )

        self.model = get_peft_model(self.model, self.config)
        print_trainable_parameters(self.model)
        training_data = prepare_training_dataset(dataset, args.model_type, args.train_model, self.tokenizer)

        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=training_data,
            args=transformers.TrainingArguments(
                report_to=None,
                per_device_train_batch_size=1, # 2
                gradient_accumulation_steps=16, # 8 takes 7 min, 4 4 too
                gradient_checkpointing=True,
                warmup_ratio=0.5,
                num_train_epochs=1,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                output_dir='outputs'
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        self.trainer.train()

        self.model.save_pretrained(dir_name)

    def evaluate(self, dataset, dir_name, data_suffix, stop_token=None):
        self._load_loras()
        #print(stop_token)
        if stop_token:
            stopping_criterium = KeywordsStoppingCriteria(self.tokenizer.encode(stop_token)[1:])

        create_folder_if_not_exists(f"{dir_name}_eval")
        create_folder_if_not_exists(f"{dir_name}_io_test")
        dataset = prepare_eval_dataset(dataset, args.model_type, args.eval_model, self.tokenizer)

        #print(self.tokenizer.encode(stop_token)[1:])

        for sample in dataset:
            name = list(sample.keys ())[0].split("__name__")[1]
            #if not os.path.exists(f"{dir_name}_eval/{name}{data_suffix}.c"):
   #        print("test")
            print(list(sample.values())[0])            

            if stop_token:
                with torch.cuda.amp.autocast():
                    print("Hi")

                    print(stopping_criterium)
                    output_tokens = self.model.generate(**list(sample.values())[0], min_length=0, max_new_tokens=args.max_tokens, stopping_criteria=StoppingCriteriaList([stopping_criterium]))

                    print(f"Output tokens: {output_tokens}")

                    
            else:
                with torch.cuda.amp.autocast():
                    output_tokens = self.model.generate(**list(sample.values())[0], min_length=0, max_new_tokens=args.max_tokens)

            output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=False)
            print(output)

            print(get_code_from_response(output, args.model_type))

            with open(f"{dir_name}_eval/{name}{data_suffix}.json", "w") as f:
                f.write(output)

    def build_evaluatable_code(self, dataset, dir_name, obfs_data_suffix, data_suffix):
#        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        seed = 42
        exebench = load_dataset("jordiae/exebench", split='test_real')
        exebench = exebench.shuffle(seed)
#        dataset = prepare_eval_dataset(dataset, args.model_type, args.eval_model, self.tokenizer)
        dataset = load_jsonl_dataset(dataset)

        for sample in dataset:
            name = list(sample.keys ())[0].split("__name__")[1]

            with open(f"{dir_name}_eval/{name}{data_suffix}.json", "r") as f:
                output = f.read()

            # filter the actual code from other stuff
            code = get_code_from_response(output, args.model_type)

            # save the c code with a main function
            with open(f"{dir_name}_eval/{name}{data_suffix}.c", "w") as f:
                f.write(code + "\nint main(){}")

            # retrieve the original program from exebench, thats why we also store the index in the eval dataset, make sure to shuffle with the same seed as during the eval dataset creation
            program = exebench[int(list(sample.keys())[0].split("__name__")[0])]


            fname_orig = program['fname']


            # extract helper functions from obfuscated, data suffix can differentiate between specific transformations and chains
            helper_functions = extract_function("datasets/obfuscated_eval/" + fname_orig + obfs_data_suffix + ".c", name, "", False, True, True, extract_helpers=True, extract_only_helpers=True)

            print(helper_functions)
            # building the base io program with headers and more
            io_program = build_program(sample=program, empty_main=False, is_main=False, func_def_is_external=True, func_def=f"{helper_functions}\n{code}")
            # extract function arguments from the call in the wrapper to adjust them in case of randomize arguments
            function_arguments = re.search(program['fname'] + r"\([^)]*\)", program['real_exe_wrapper']).group().removeprefix(program['fname'] + "(")[:-1].split(",")
            print(function_arguments)
            
            # extract the function name from the deobfuscated code so we know the name the call in the wrapper has to be replaced with
            try:
                fname = extract_function_name(code)

            except Exception as e:
                print(e)
                continue

            if fname == None:
                print("Function name is blank")
                continue


            io_program = correct_function_arguments(io_program, fname_orig, "obfuscated", args.obfs_data_suffix, args.data_suffix)

            # handle cases where a standard name is given for a function and to resolve conflicts __bench is appended
            if fname_orig + "__bench(" in io_program:
                io_program = re.sub(fname_orig + r"(?!__bench)\(", fname_orig + "__bench(", io_program)

            # replace the name with the deobfuscated one
            io_program = re.sub(fname_orig + r"\s*\(", fname + "(", io_program)

            # write the io wrapper
            with open(f"{dir_name}_io_test/{name}{data_suffix}.cpp", "w") as f:
                f.write(io_program)


class RemoteLLM(LLMBase):
    def __init__(self, model_id):
        self.model = None
        self.model_id = model_id

class DeepSeekLLM(LocalLLM):
    def __init__(self, model_id):
        super().__init__(
            model_id=model_id,
            chat_template=deepseek_chat_template)

    def train(self, dataset, dir_name):
        super().train(dataset, dir_name)

    def evaluate(self, dataset, dir_name, data_suffix):
        super().evaluate(dataset, dir_name, data_suffix)

    def build_evaluatable_code(self, dataset, dir_name, obfs_data_suffix, data_suffix):
        super().build_evaluatable_code(dataset, dir_name, obfs_data_suffix, data_suffix)


class CodeLlamaLLM(LocalLLM):
    def __init__(self, model_id):
        super().__init__(
            model_id=model_id,
            chat_template=None)

    def train(self, dataset, dir_name):
        super().train(dataset, dir_name)

    def evaluate(self, dataset, dir_name, data_suffix):
        super().evaluate(dataset, dir_name, data_suffix, stop_token="<|end|>")

    def build_evaluatable_code(self, dataset, dir_name, obfs_data_suffix, data_suffix):
        super().build_evaluatable_code(dataset, dir_name, obfs_data_suffix, data_suffix)

class Codegen25LLM(LocalLLM):
    def __init__(self, model_id):
        super().__init__(
            model_id=model_id,
            chat_template=None)

    def train(self, dataset, dir_name):
        super().train(dataset, dir_name)

    def evaluate(self, dataset, dir_name):
        super().evaluate(dataset, dir_name, data_suffix)

    def build_evaluatable_code(self, dataset, dir_name, obfs_data_suffix, data_suffix):
        super().build_evaluatable_code(dataset, dir_name, obfs_data_suffix, data_suffix)

class OpenAILLM(RemoteLLM):
    def __init__(self, model_id):
        super().__init__(model_id=model_id)
        
        self.client = OpenAI(api_key="")
        self.model_type = model_id

    def train(self, dataset, dir_name):
        raise NotImplementedError

    def evaluate(self, dataset, dir_name, data_suffix):
#        self._load_loras()
        create_folder_if_not_exists(f"{dir_name}_eval")
        create_folder_if_not_exists(f"{dir_name}_io_test")
        dataset = load_jsonl_dataset(dataset)


        for sample in dataset:
            name = list(sample.keys ())[0].split("__name__")[1]
            #if not os.path.exists(f"{dir_name}_eval/{name}{data_suffix}.c"):
   #        print("test")
            print(list(sample.values())[0])
            obfs = list(sample.values())[0].split("<|ORIG|>")[0].split("<|OBFS|>\n")[1]

            prompt = f"Provide the deobfuscated version of this program ```{obfs}```"


            #with torch.cuda.amp.autocast():
                #output_tokens = self.model.generate(**list(sample.values())[0], min_length=0, max_new_tokens=args.max_tokens)

            #output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=False)
            #print(output)

            #print(get_code_from_response(output, args.model_type))

            messages=[
                {'role' : 'system', 'content' : "You are a deobfuscation expert. You immediately provide the deobfuscated code without any further text as an output. Do not rename identifier names, just lower the code complexity.\n\nExample:\n// Obfuscated code\nvoid ddafee92(...)\n\t...\n}\n\n// Deobfuscated code\nvoid ddafee92(...)\n\t...\n}"},
                {'role': 'user', 'content': prompt},
            ]
            response = self.client.chat.completions.create(model=self.model_id, messages=messages, max_tokens=args.max_tokens, temperature=0)
            print(response)

            generated_code = response.choices[0].message.content

            with open(f"{dir_name}_eval/{name}{data_suffix}.json", "w") as f:
                json.dump(response.json(), f)

            time.sleep(8)

    def build_evaluatable_code(self, dataset, dir_name, obfs_data_suffix, data_suffix):
        seed = 42
        exebench = load_dataset("jordiae/exebench", split='test_real')
        exebench = exebench.shuffle(seed)
        samples = load_jsonl_dataset(dataset)
        
        for sample in samples:
            name = list(sample.keys ())[0].split("__name__")[1]

            with open(f"{dir_name}_eval/{name}{data_suffix}.json", "r") as f:
                output = json.load(f)

            # filter the actual code from other stuff
            code = get_code_from_response(output, args.model_type)

            # save the c code with a main function
            with open(f"{dir_name}_eval/{name}{data_suffix}.c", "w") as f:
                f.write(code + "\nint main(){}")

            # retrieve the original program from exebench, thats why we also store the index in the eval dataset, make sure to shuffle with the same seed as during the eval dataset creation
            program = exebench[int(list(sample.keys())[0].split("__name__")[0])]


            fname_orig = program['fname']


            # extract helper functions from obfuscated, data suffix can differentiate between specific transformations and chains
            helper_functions = extract_function("datasets/obfuscated_eval/" + fname_orig + obfs_data_suffix + ".c", name, "", False, True, True, extract_helpers=True, extract_only_helpers=True)

            print(helper_functions)
            # building the base io program with headers and more
            io_program = build_program(sample=program, empty_main=False, is_main=False, func_def_is_external=True, func_def=f"{helper_functions}\n{code}")
            # extract function arguments from the call in the wrapper to adjust them in case of randomize arguments
            function_arguments = re.search(program['fname'] + r"\([^)]*\)", program['real_exe_wrapper']).group().removeprefix(program['fname'] + "(")[:-1].split(",")
            print(function_arguments)
            
            # extract the function name from the deobfuscated code so we know the name the call in the wrapper has to be replaced with
            try:
                fname = extract_function_name(code)

            except Exception as e:
                print(e)
                continue

            if fname == None:
                print("Function name is blank")
                continue

            io_program = correct_function_arguments(io_program, fname_orig, "obfuscated", args.obfs_data_suffix, args.data_suffix)

            # handle cases where a standard name is given for a function and to resolve conflicts __bench is appended
            if fname_orig + "__bench(" in io_program:
                io_program = re.sub(fname_orig + r"(?!__bench)\(", fname_orig + "__bench(", io_program)

            # replace the name with the deobfuscated one
            io_program = re.sub(fname_orig + r"\s*\(", fname + "(", io_program)

            # write the io wrapper
            with open(f"{dir_name}_io_test/{name}{data_suffix}.cpp", "w") as f:
                f.write(io_program)

if __name__ == "__main__":
    if args.train_file:
        # dataset = load_jsonl_dataset(args.train_file)
        llm = get_model(args.train_model)
        llm.train(dataset=args.train_file, dir_name=args.trained_model_path)

    if args.eval_model and not args.build_eval_files:
        dataset = load_jsonl_dataset(args.eval_file)
        llm = get_model(args.eval_model)
        llm.evaluate(dataset=args.eval_file, dir_name=args.eval_out_path, data_suffix=args.data_suffix)

    if args.build_eval_files:
        llm = get_model(args.eval_model)
        dataset = load_jsonl_dataset(args.eval_file)
        llm.build_evaluatable_code(dataset=args.eval_file, dir_name=args.eval_out_path, obfs_data_suffix=args.obfs_data_suffix, data_suffix=args.data_suffix)
