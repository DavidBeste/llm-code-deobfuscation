import random

transformations = ["encode_arithmetic", "encode_branches", "flatten", "opaque", "randomize_arguments"]

large_dataset = ""

for transformation in transformations:
    with open("datasets/obfuscation_dataset_" + transformation + "_6144.txt", "r") as f:
        dataset = f.read()

    large_dataset += dataset

random.seed(42)
large_dataset = large_dataset.split("<|OBFS|>")[1:]
print(len(large_dataset))
random.shuffle(large_dataset)
large_dataset = "<|OBFS|>" + "<|OBFS|>".join(large_dataset)

with open("datasets/obfuscation_dataset_6144_all.txt", "w") as f:
    f.write(large_dataset)
