import random

large_dataset = ""

for c in range(1,6):
    with open("datasets/obfuscation_dataset_" + str(c) + "_chain_4096_training.txt", "r") as f:
        dataset = f.read()

    large_dataset += dataset

random.seed(42)
large_dataset = large_dataset.split("<|OBFS|>")[1:]
print(len(large_dataset))
random.shuffle(large_dataset)
large_dataset = "<|OBFS|>" + "<|OBFS|>".join(large_dataset)

with open("datasets/obfuscation_dataset_chain_4096_all_training.txt", "w") as f:
    f.write(large_dataset)
