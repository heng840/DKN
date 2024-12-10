import argparse
import json

from knowledge_neurons.utils import get_model_output, initiate_model_tokenizer


def expand_data_item(model, tokenizer, item, device):
    # Generate correct context
    correct_prompt = f"Expand this fact into a detailed sentence: In 2010, Tom Brady plays for {item['obj_label']}."
    correct_context = get_model_output(model, tokenizer, correct_prompt, device=device)

    # Generate incorrect context
    incorrect_prompt = f"Expand this fact into a detailed sentence: In 2010, Tom Brady plays for {item['wrong_fact'][0]}."
    incorrect_context = get_model_output(model, tokenizer, incorrect_prompt, device=device)

    return {
        "uuid": item["uuid"],
        "correct_context": correct_context,
        "incorrect_context": incorrect_context,
        "relation_name": item["relation_name"]
    }
def extract_triplet_from_context(model, tokenizer, context, device='cuda'):
    prompt = f"Summarize the following sentence into a simple fact: {context}"
    simplified_fact = get_model_output(model, tokenizer, prompt, device=device)
    return simplified_fact
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        # default="EleutherAI/gpt-j-6b",
        default='gpt2',
        # default='/home/chenyuheng/KN2/Llama/Llama7bChat',
        # default="meta-llama/Llama-2-70b-chat-hf",
        # default='meta-llama/Llama-2-13b-chat-hf',
    )
    parser.add_argument('--source_triple_data', type=str,
                        default='Templama/train.jsonl',
                        )
    parser.add_argument('--expand_data', type=str,
                        default='Templama/expand_data/train.jsonl',
                        )
    parser.add_argument('--new_triple_data', type=str,
                        default='Templama/new_triple_data/train.jsonl',
                        )
    args = parser.parse_args()
    model, tokenizer = initiate_model_tokenizer(args.model_name)
    def expand_data():
        expanded_dataset = []
        with open(args.source_triple_data, 'r') as file:
            for line in file:
                item = json.loads(line)
                expanded_item = expand_data_item(model, tokenizer, item, device='cuda')
                expanded_dataset.append(expanded_item)

        # Save the expanded dataset
        with open(args.expand_data, 'w') as file:
            for item in expanded_dataset:
                file.write(json.dumps(item) + '\n')
    def extract_data():
        extracted_dataset = []
        with open(args.expand_data, 'r') as file:
            for line in file:
                item = json.loads(line)
                extracted_correct_fact = extract_triplet_from_context(model, tokenizer, item['correct_context'])
                extracted_incorrect_fact = extract_triplet_from_context(model, tokenizer, item['incorrect_context'])
                extracted_item = {
                    "uuid": item["uuid"],
                    "extracted_correct_fact": extracted_correct_fact,
                    "extracted_incorrect_fact": extracted_incorrect_fact,
                    "relation_name": item["relation_name"]
                }
                extracted_dataset.append(extracted_item)

        # Save the extracted dataset
        with open(args.new_triple_data, 'w') as file:
            for item in extracted_dataset:
                file.write(json.dumps(item) + '\n')
    expand_data()
    extract_data()
