import json

def extract_and_save_unique_queries():
    seen_queries = set()
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            data = json.loads(line)
            query = data['query']
            # Only add to output if query hasn't been seen before
            if query not in seen_queries:
                seen_queries.add(query)
                output_file.write(json.dumps({"query": query}) + '\n')

input_file_path = '/home/chenyuheng/KN2/kn2/Templama/train.jsonl'  # Replace with your input file path
output_file_path = '/home/chenyuheng/KN2/kn2/Templama/train_convert_tmp.jsonl'  # Replace with your desired output file path

# extract_and_save_unique_queries()

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Read the original and enhanced JSONL files
original_data = read_jsonl(input_file_path)
enhanced_data = read_jsonl('/home/chenyuheng/KN2/kn2/Templama/converted_by_gpt4.jsonl')

# Initialize variables
enhanced_index = 0
last_base_id = None
new_data = []

# Iterate through the original data and create new data entries
for entry in original_data:
    base_id = entry['id'].rsplit('_', 1)[0]

    # Check if the base_id has changed and move to the next enhanced query
    if base_id != last_base_id and enhanced_index < len(enhanced_data):
        last_base_id = base_id
        enhanced_index += 1

    # Create a new entry with the enhanced query and other original key-values
    if enhanced_index <= len(enhanced_data):
        new_entry = entry.copy()
        new_entry['query'] = enhanced_data[enhanced_index - 1]['query']
        new_data.append(new_entry)

# Save the updated data
save_jsonl(new_data, '/home/chenyuheng/KN2/kn2/Templama/train_enhance.jsonl')

