from datasets import load_dataset
from tqdm import tqdm
import os

# Define the output file path
output_file = "data_generator/data/input/oscar_ro_train.txt"
N = 2000

# Check if the file already exists
if os.path.exists(output_file):
    print(f"File {output_file} already exists.")
    # Print the first N characters from the file
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read(N)
        print(f"First {N} characters of {output_file}:\n{content}")
else:
    # Write the dataset to a .txt file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Load the Romanian subset of the OSCAR corpus
    dataset = load_dataset("oscar", "unshuffled_deduplicated_ro")
    # Print the dataset details
    print(dataset)
    # Access the train split
    train_split = dataset['train']
    # Write the dataset to a .txt file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(train_split):
            f.write(example['text'] + "\n")
    print(f"Dataset saved to {output_file}")






