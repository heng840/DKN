import json
from transformers import GPT2Tokenizer

def find_max_token_length(file_path, tokenizer):
    max_length = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            answer_name = data['answer'][0]['name']  # Extracting the answer's name
            tokens = tokenizer.encode(answer_name)  # Tokenizing the answer
            max_length = max(max_length, len(tokens))  # Updating the max token length if necessary

    return max_length

# Example usage
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# file_path = '/home/chenyuheng/chenyuheng/KN2/Templama/train.jsonl'
# max_token_length = find_max_token_length(file_path, tokenizer)
# print(f"The maximum token length in the answers is: {max_token_length}")
# file_path = '/home/chenyuheng/chenyuheng/KN2/Templama/val.jsonl'
# max_token_length = find_max_token_length(file_path, tokenizer)
# print(f"The maximum token length in the answers is: {max_token_length}")
# Using cls_token, but it is not set yet.
# Using mask_token, but it is not set yet.
# Using pad_token, but it is not set yet.
# Using sep_token, but it is not set yet.
# The maximum token length in the answers is: 18 test
# The maximum token length in the answers is: 18 train
# The maximum token length in the answers is: 16 val

import json
import os
import random
random.seed(42)
def clean_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Clean the data
    for key in data.keys():
        if isinstance(data[key], dict):
            for subkey, value in data[key].items():
                if {} in value:
                    data[key][subkey] = []

    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def clean_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            clean_json_file(filepath)

def split_jsonl_file(filename, num_splits):
    os.makedirs(f'/home/chenyuheng/KN2/kn2/Templama/train_split', exist_ok=True)
    # Read lines from the original file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Calculate the number of lines per split
    lines_per_split = len(lines) // num_splits

    # Split lines into chunks and write to new files
    for i in range(num_splits):
        start_index = i * lines_per_split
        end_index = None if i == num_splits - 1 else start_index + lines_per_split
        chunk = lines[start_index:end_index]

        with open(f"/home/chenyuheng/KN2/kn2/Templama/train_split/part_{i+1}.jsonl", 'w') as out_file:
            out_file.writelines(chunk)

# Example usage
split_jsonl_file('/home/chenyuheng/KN2/kn2/Templama/train.jsonl', 3)

x=0
def random_sample_jsonl(filename, sample_size):
    os.makedirs(f'/home/chenyuheng/KN2/kn2/Templama/train_split', exist_ok=True)
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Randomly sample the lines
    sampled_lines = random.sample(lines, min(sample_size, len(lines)))

    # Write the sampled lines to a new file
    with open(f"/home/chenyuheng/KN2/kn2/Templama/train_split/random_sample.jsonl", 'w') as out_file:
        out_file.writelines(sampled_lines)

# Example usage
# random_sample_jsonl('/home/chenyuheng/KN2/kn2/Templama/train.jsonl', 1000)
# Usage
# clean_json_files(directory='/home/chenyuheng/KN2/kn2/temporal_res/1205/main_temporal_res')
