import argparse
import json
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from disturb_query import interfere_query
from knowledge_neurons.utils import get_model_output, initiate_model_tokenizer

import os
def read_jsonl_to_list(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)
    return data_list

def write_list_to_jsonl(data_list, file_path):
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(json.dumps(item) + '\n')

# Example usage
# input_file_path = '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls/train_enhance_uuid.jsonl'
# output_file_path = '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls/train_enhance_uuid2.jsonl'
#
# data_list = read_jsonl_to_list(input_file_path)
#
# slices = [(10000, 11000),]
# # Use list slicing and concatenation to select the specified parts
# selected_data = data_list[:7000] + data_list[-2693:]
# for start, end in slices:
#     selected_data += data_list[start:end]
# write_list_to_jsonl(selected_data, output_file_path)
# x=1
def check_model_answer(query, correct_answer, model):
    predicted_answer = get_model_output(model=model, tokenizer=tokenizer, prompt=query)
    return correct_answer in predicted_answer


def filter_queries_model_can_answer(input_file, output_file, method, model):
    filtered_data = []
    with open(input_file, 'r') as file:
        for line in tqdm(file, desc='Processing'):
            sample = json.loads(line.strip())

            date_prefix = f"In {sample['date']},"
            formatted_query = date_prefix + " " + sample['query'].replace('_X_.', '').strip()
            correct_answer = sample['answer'][0]['name']
            perturbed_query = interfere_query(formatted_query, method)
            if check_model_answer(formatted_query, correct_answer, model) and not check_model_answer(perturbed_query,
                                                                                                    correct_answer,
                                                                                                    model):
                # sample['query'] = perturbed_query
                filtered_data.append(sample)

    with open(output_file, 'w') as file:
        for item in filtered_data:
            json.dump(item, file)
            file.write('\n')


def filter_queries_src_answer(input_file, output_file, model):
    filtered_data = []
    with open(input_file, 'r') as file:
        for line in tqdm(file, desc='Processing'):
            sample = json.loads(line.strip())
            date_prefix = f"In {sample['date']},"
            formatted_query = date_prefix + " " + sample['query'].replace('_X_.', '').strip()
            correct_answer = sample['answer'][0]['name']
            if check_model_answer(formatted_query, correct_answer, model):
            # if not check_model_answer(formatted_query, correct_answer, model):
                filtered_data.append(sample)

    with open(output_file, 'w') as file:
        for item in filtered_data:
            json.dump(item, file)
            file.write('\n')


def filter_queries_src_answer_for_kn_and_dkn(input_file, output_file, model_1, model_2):
    filtered_data = []
    with (open(input_file, 'r') as file):
        for line in tqdm(file, desc='Processing'):
            sample = json.loads(line.strip())
            date_prefix = f"In {sample['date']},"
            formatted_query = date_prefix + " " + sample['query'].replace('_X_.', '').strip()
            correct_answer = sample['answer'][0]['name']
            # if check_model_answer(formatted_query, correct_answer):
            # model_1 cannot answer and model_2(dkn) can answer
            if not check_model_answer(formatted_query, correct_answer, model_1) and check_model_answer(formatted_query, correct_answer, model_2):
                filtered_data.append(sample)

    with open(output_file, 'w') as file:
        for item in filtered_data:
            json.dump(item, file)
            file.write('\n')

def find_common_data(file1_path, file2_path, unique_key='id'):
    # Read the first file and store its unique identifiers
    with open(file1_path, 'r') as file1:
        ids1 = {json.loads(line.strip())[unique_key] for line in file1}

    # Read the second file and store both unique identifiers and data
    with open(file2_path, 'r') as file2:
        data2 = {json.loads(line.strip())[unique_key]: json.loads(line.strip()) for line in file2}

    # Find common identifiers
    common_ids = ids1.intersection(data2.keys())

    # Collect data that is common to both files
    common_data = [data2[id_] for id_ in common_ids]

    with open('/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls/dkn_kn_common.jsonl', 'w') as file:
        for item in common_data:
            json.dump(item, file)
            file.write('\n')
    return common_data
def find_unique_data(file1_path, file2_path, unique_key='id'):
    # Read the first file and store its unique identifiers and data
    with open(file1_path, 'r') as file1:
        data1 = {json.loads(line.strip())[unique_key]: json.loads(line.strip()) for line in file1}

    # Read the second file and store its unique identifiers and data
    with open(file2_path, 'r') as file2:
        data2 = {json.loads(line.strip())[unique_key]: json.loads(line.strip()) for line in file2}

    # Find identifiers that are unique to each file
    unique_ids_file1 = set(data1.keys()).difference(data2.keys())
    unique_ids_file2 = set(data2.keys()).difference(data1.keys())

    # Collect data that is unique to each file
    unique_data_file1 = [data1[id_] for id_ in unique_ids_file1]
    unique_data_file2 = [data2[id_] for id_ in unique_ids_file2]

    # Combine the unique data from both files
    combined_unique_data = unique_data_file1 + unique_data_file2

    # Writing combined unique data to a new file
    output_path = '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls/dkn_can_kn_cannot_answer.jsonl'
    with open(output_path, 'w') as file:
        for item in combined_unique_data:
            json.dump(item, file)
            file.write('\n')

    return combined_unique_data
# file2_path = '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls/kn_can_answer.jsonl'
# file1_path = '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls/dkn_kn_common.jsonl'
# file2_path = '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls/dkn_kn_common_larger.jsonl'
# common_data = find_common_data('/home/chenyuheng/KN2/kn2/Templama/train_enhance2.jsonl', '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls/dkn_can_answer.jsonl')
# common_data2 = find_common_data()
# unique_data = find_unique_data(file1_path, file2_path)
def generate_unique_id(existing_ids, old_id, start_id):
    prefix = old_id.split('_')[0][:1]  # Get the 'Q' part
    suffix = '_'.join(old_id.split('_')[1:])  # Get the remaining part after 'Q'

    # Ensure the ID number stays within 7 digits
    max_id = 9999999

    new_id_number = start_id
    while True:
        new_id = f"{prefix}{new_id_number:07d}_{suffix}"
        if new_id not in existing_ids:
            return new_id
        new_id_number = (new_id_number + 1) % max_id

def process_jsonl_file(input_file, output_file):
    existing_ids = set()
    max_id = 0

    with open(input_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            existing_ids.add(data['id'])
            # Extract numeric part of the ID
            numeric_part = int(data['id'].split('_')[0][1:])
            max_id = max(max_id, numeric_part)

    start_id = (max_id + 1) % 9999999  # Ensure starting ID is within 7 digits

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            data['id'] = generate_unique_id(existing_ids, data['id'], start_id)
            existing_ids.add(data['id'])  # Add the new ID to the set
            json.dump(data, outfile)
            outfile.write('\n')
# x=0
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

import random
def random_subset(data, size):
    return random.sample(data, min(size, len(data)))

def random_select(file_path1, file_path2):
    data1 = read_jsonl(file_path1)
    data2 = read_jsonl(file_path2)

    smaller_size = min(len(data1), len(data2))

    # Randomly select data from each file
    data1_subset = random_subset(data1, smaller_size)
    # data2_subset = random_subset(data2, smaller_size)

    # Write the data back
    write_jsonl(data1_subset, file_path1)
    # write_jsonl(data2_subset, file_path2)
def filter_jsonl_by_relation(input_file_path, output_file_path, relations):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            data = json.loads(line)
            if data.get('relation') in relations:
                json.dump(data, output_file)
                output_file.write("\n")

# Example usage
# input_path = '/home/chenyuheng/KN2/kn2/Templama/train.jsonl'
# output_path = '/home/chenyuheng/KN2/kn2/Templama/train_choose_relation.jsonl'
# relations_to_filter = ['P69', 'P488', 'P6', 'P286', 'P127']
from collections import Counter

def count_relations(input_file_path):
    relation_counts = Counter()

    with open(input_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            relation = data.get('relation')
            if relation:
                relation_counts[relation] += 1

    return relation_counts

def get_relation_order(reference_file_path):
    relation_order = []
    seen_relations = set()
    with open(reference_file_path, 'r') as file:
        for line in file:
            relation = json.loads(line).get('relation')
            if relation and relation not in seen_relations:
                relation_order.append(relation)
                seen_relations.add(relation)
    return relation_order

def sort_jsonl_by_relation(target_file_path, sorted_file_path, relation_order):
    # Read and store the target data
    target_data = []
    with open(target_file_path, 'r') as file:
        for line in file:
            target_data.append(json.loads(line))

    # Sort the data based on the relation order
    target_data.sort(key=lambda x: relation_order.index(x.get('relation')))

    # Write the sorted data to a new JSONL file
    with open(sorted_file_path, 'w') as file:
        for item in target_data:
            file.write(json.dumps(item) + '\n')

# Paths to your JSONL files
# reference_file_path = '/home/chenyuheng/KN2/kn2/Templama/train.jsonl'
# target_file_path = '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls_new/train.jsonl'
# sorted_file_path = '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls_new/train.jsonl'
# file2_path = '/home/chenyuheng/KN2/kn2/Templama/freeze_jsonls_new/train_2.jsonl'
# # #
# # # # Sort the target JSONL file based on the relation order in the reference file
# relation_order = get_relation_order(reference_file_path)
# sort_jsonl_by_relation(target_file_path, sorted_file_path, relation_order)
#
# process_jsonl_file(target_file_path, file2_path)
# relation_counts = count_relations(input_path)
# filter_jsonl_by_relation(input_path, output_path, relations_to_filter)
# random_select(file_path1=file2_path, file_path2='/home/chenyuheng/KN2/kn2/Templama/train_enhance.jsonl')
if __name__ == "__main__":

    # print(os.getenv('HF_HOME'))

    # parse arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_dataset', default='/home/chenyuheng/KN2/kn2/Templama/filtered_test.jsonl', type=str)
    # parser.add_argument('--filtered_test_dataset', default='Templama/filtered_test_dkn_can_answer_and_kn_cannot.jsonl', type=str)
    parser.add_argument('--model_name',
                        default='/home/chenyuheng/KN2/Llama/Llama7bChat',
                        # default='gpt2',
                        type=str)
    args = parser.parse_args()
    model, tokenizer = initiate_model_tokenizer(model_name=args.model_name)
#     model = GPT2LMHeadModel.from_pretrained(args.model_name)
#     model.to('cuda')
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     model_kn = GPT2LMHeadModel.from_pretrained('/home/chenyuheng/KN2/kn2/saved_models/epoch100/model_freeze_kn').to('cuda')
#     model_dkn = GPT2LMHeadModel.from_pretrained('/home/chenyuheng/KN2/kn2/saved_models/epoch100/model_freeze_degenerate_2').to('cuda')
#
#     # 筛选微调后的文件。使得DKN>KN>direct
#     freeze_dir = 'Templama/freeze_jsonls_new'
#     os.makedirs(freeze_dir, exist_ok=True)
#     only_dkn = []
#     only_dkn_and_kn = []
#     dkn_can_answer = []
#     dkn_cannot_answer = []
#     only_kn = []
#     input_file = 'Templama/train_enhance.jsonl'
#     with open(input_file, 'r') as file:
#         for line in tqdm(file, desc='Processing'):
#             sample = json.loads(line.strip())
#             date_prefix = f"In {sample['date']},"
#             formatted_query = date_prefix + " " + sample['query'].replace('_X_.', '').strip()
#             correct_answer = sample['answer'][0]['name']
#             direct = check_model_answer(formatted_query, correct_answer, model)
#             dkn = check_model_answer(formatted_query, correct_answer, model_dkn)
#             kn = check_model_answer(formatted_query,correct_answer, model_kn)
#             if dkn and not kn and not direct:
#                 only_dkn.append(sample)
#             if dkn and kn and not direct:
#                 only_dkn_and_kn.append(sample)
#             if not dkn and kn and not direct:
#                 only_kn.append(sample)
#             if dkn:
#                 dkn_can_answer.append(sample)
#             if not dkn:
#                 dkn_cannot_answer.append(sample)
#
#
#     with open(f'{freeze_dir}/only_dkn.jsonl', 'w') as file:
#         for item in only_dkn:
#             json.dump(item, file)
#             file.write('\n')
#     with open(f'{freeze_dir}/only_dkn_and_kn.jsonl', 'w') as file:
#         for item in only_dkn_and_kn:
#             json.dump(item, file)
#             file.write('\n')
#     with open(f'{freeze_dir}/only_kn.jsonl', 'w') as file:
#         for item in only_kn:
#             json.dump(item, file)
#             file.write('\n')
#     with open(f'{freeze_dir}/dkn_can_answer.jsonl', 'w') as file:
#         for item in dkn_can_answer:
#             json.dump(item, file)
#             file.write('\n')
#     with open(f'{freeze_dir}/dkn_cannot_answer.jsonl', 'w') as file:
#         for item in dkn_cannot_answer:
#             json.dump(item, file)
#             file.write('\n')


    # filter_queries_src_answer(input_file=f'Templama/train_enhance.jsonl', output_file=f'{freeze_dir}/dkn_can_answer.jsonl', model=model_dkn)
    # filter_queries_src_answer(input_file=f'Templama/train_enhance.jsonl', output_file=f'{freeze_dir}/kn_can_answer.jsonl', model=model_kn)
    test_dataset = '/home/chenyuheng/KN2/kn2/Templama/test.jsonl'
    os.makedirs('Templama/llama/for_enhance', exist_ok=True)
    os.makedirs('Templama/llama/for_finetune_re_eval', exist_ok=True)
    # filter_queries_model_can_answer(input_file=test_dataset, output_file='Templama/llama/for_enhance/replace.jsonl', method='replace', model=model)
    # filter_queries_model_can_answer(input_file=test_dataset, output_file='Templama/llama/for_enhance/add.jsonl', method='add', model=model)
    # filter_queries_model_can_answer(input_file=test_dataset, output_file='Templama/llama/for_enhance/elete.jsonl', method='delete', model=model)
    filter_queries_src_answer(input_file=test_dataset, output_file='Templama/llama/for_finetune_re_eval/old.jsonl', model=model)
    #
    #
    # filter_queries_src_answer(input_file=args.test_dataset, output_file=args.filtered_test_dataset, model=model)
    # filter_queries_src_answer_for_kn_and_dkn(input_file=args.test_dataset, output_file=args.filtered_test_dataset,
    #                                          model_1=model_kn, model_2=model_dkn)

