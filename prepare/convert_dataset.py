import json
from collections import defaultdict
import os
import random
random.seed(42)
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data
def convert_dataset2wrong(src_path='/home/chenyuheng/KN2/kn2/Templama/updated_train.jsonl',
                          wrong_dir='/home/chenyuheng/KN2/kn2/datasets/wrong_fact_dataset/temporal/train-llama-complex'):
    # Load the original dataset
    original_data = read_jsonl(src_path)

    # Process and transform the dataset
    transformed_data = {}
    grouped_by_query = defaultdict(list)
    grouped_by_relation = defaultdict(list)

    # Group data by query and relation name
    for item in original_data:
        grouped_by_query[item['query']].append(item)
        grouped_by_relation[item.get('relation', '')].append(item)

    # Process each group
    for query, items in grouped_by_query.items():
        for item in items:
            true_answer = item['answer'][0]['name']
            wrong_facts = set()

            for other_item in items:
                if len(wrong_facts) >= 3:
                    # Break the loop if we already have 3 wrong facts
                    break
                if other_item['id'] != item['id'] and other_item['answer'][0]['name'] != true_answer:
                    wrong_facts.add(other_item['answer'][0]['name'])
            if not wrong_facts:
                same_relation_items = grouped_by_relation[item.get('relation', '')]
                potential_wrong_facts = [x['answer'][0]['name'] for x in same_relation_items if
                                         x['answer'][0]['name'] != true_answer]

                if potential_wrong_facts:
                    wrong_facts.add(random.choice(potential_wrong_facts))

            uuid = item['id']
            new_query = f"In {item['date']}, {query}"
            transformed_data[uuid] = {
                "sentences": [new_query],
                "relation_name": item.get('relation', ''),
                "obj_label": true_answer,
                "wrong_fact": list(wrong_facts)
            }
    os.makedirs(wrong_dir, exist_ok=True)
    with open(f'{wrong_dir}/lama.jsonl', 'w') as outfile:
        for uuid, entry in transformed_data.items():
            entry_with_uuid = {"uuid": uuid, **entry}
            json.dump(entry_with_uuid, outfile)
            outfile.write('\n')

convert_dataset2wrong()


# Convert a JSON file to JSONL format
def convert_json_to_jsonl(json_file, jsonl_file):
    with open(json_file, 'r') as infile, open(jsonl_file, 'w') as outfile:
        json_dict = json.load(infile)
        for uuid, entry in json_dict.items():
            # Include the uuid in each entry
            entry_with_uuid = {"uuid": uuid, **entry}
            json.dump(entry_with_uuid, outfile)
            outfile.write('\n')

# Example usage
# convert_json_to_jsonl('datasets/wrong_fact_dataset/en.json', 'datasets/wrong_fact_dataset/lama.jsonl')
