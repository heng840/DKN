import argparse
import json
import os

from tqdm import tqdm

from knowledge_neurons.utils import initiate_model_tokenizer, load_json_files_from_directory, \
    load_jsonl_files_from_directory
from knowledge_neurons import Dkn, model_type
from datasets import concatenate_datasets
def enhance_and_predict(kn, item, uuid, neurons_result, method='value', weight=2):
    # Find the neurons associated with this uuid
    neuron_data = neurons_result.get(uuid, {})
    dkn = neuron_data.get('dkn_cluster_2', [])
    dkn_2d = [neuron for subset in dkn for neuron in subset]
    date_prefix = f"In {item['date']},"
    formatted_query = date_prefix + " " + item['query']
    correct_answer = item['answer'][0]['name']
    # Enhance the neurons
    if method == 'value':
        predicted_answer =  kn.enhance_or_suppress_dkn_predict_answer(query=formatted_query, correct_answer=correct_answer,
                                                                      neurons_list_2d=dkn_2d, mode='enhance')
    elif method == 'weight':
        predicted_answer = kn.change_weights_dkn_predict_answer(query=formatted_query, neurons_list_2d=dkn_2d, value=weight)
    else:
        raise Exception('method is not deployment')
    return correct_answer in predicted_answer



def process_dataset(file_path, kn, neurons_result, method='value'):
    with open(file_path, 'r') as file:
        data = [json.loads(line.strip()) for line in file]
    # correct_data = []
    correct_predictions = 0
    total_predictions = 0
    for item in tqdm(data):
        uuid = item['id']
        predict_correctly = enhance_and_predict(kn, item, uuid, neurons_result, method=method)
        if predict_correctly:
            correct_predictions += 1
            # correct_data.append(item)
        total_predictions += 1
    # with open(answer_correct_data, 'a') as f:
    #     for record in correct_data:
    #         json_record = json.dumps(record)
    #         f.write(json_record + '\n')
    return correct_predictions, total_predictions


# Process the results as needed
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered_test_dataset', default='Templama/enhance_value_or_weight', type=str)
    parser.add_argument('--model_name', default='gpt2', type=str)
    parser.add_argument('--method', default='value', choices=['value', 'weight'], type=str)
    parser.add_argument('--neurons_result_dir', default='temporal_res/1118_3', type=str)
    parser.add_argument('--save_dir', default='Disturb_or_Enhance/llama/enhance_res', type=str)
    args = parser.parse_args()

    model, tokenizer = initiate_model_tokenizer(model_name=args.model_name)
    if args.model_name == 'gpt2':
        kn = Dkn(model, tokenizer, model_type='gpt')
        neurons_result = load_json_files_from_directory(args.neurons_result_dir, 'temporal_results')
    else:
        kn = Dkn(model, tokenizer, model_type='llama')
        neurons_result = load_jsonl_files_from_directory(args.neurons_result_dir, 'temporal_results')

    if args.model_name == 'gpt2':
        filtered_datasets = [
            f'{args.filtered_test_dataset}/filtered_train_enhance_add.jsonl',
            f'{args.filtered_test_dataset}/filtered_train_enhance_delete.jsonl',
            f'{args.filtered_test_dataset}/filtered_train_enhance_replace.jsonl',
        ]
    else:
        filtered_datasets = [
            f'{args.filtered_test_dataset}/add.jsonl',
            f'{args.filtered_test_dataset}/delete.jsonl',
            f'{args.filtered_test_dataset}/replace.jsonl',
        ]

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # answer_correct_data = f'{args.neurons_result_dir}/enhance_perturb/filtered_can_answer.jsonl'
    # answer_correct_data_value_add = f'{args.neurons_result_dir}/enhance_perturb/filtered_train_enhance_value_add.jsonl'
    # answer_correct_data_value_delete = f'{args.neurons_result_dir}/enhance_perturb/filtered_train_enhance_value_delete.jsonl'
    # answer_correct_data_value_replace = f'{args.neurons_result_dir}/enhance_perturb/filtered_train_enhance_value_replace.jsonl'
    # answer_correct_data_weight_add = f'{args.neurons_result_dir}/enhance_perturb/filtered_train_enhance_weight_add.jsonl'
    # answer_correct_data_weight_delete = f'{args.neurons_result_dir}/enhance_perturb/filtered_train_enhance_weight_delete.jsonl'
    # answer_correct_data_weight_replace = f'{args.neurons_result_dir}/enhance_perturb/filtered_train_enhance_weight_replace.jsonl'

    overall_correct_predictions_value = 0
    overall_total_predictions_value = 0
    overall_correct_predictions_weight = 0
    overall_total_predictions_weight = 0

    for file_path in filtered_datasets:
        correct_predictions_value, total_predictions_value = process_dataset(file_path, kn, neurons_result, method='value')
        correct_predictions_weight, total_predictions_weight = process_dataset(file_path, kn, neurons_result, method='weight')
        overall_correct_predictions_value += correct_predictions_value
        overall_total_predictions_value += total_predictions_value
        overall_correct_predictions_weight += correct_predictions_weight
        overall_total_predictions_weight += total_predictions_weight

        res = {
            'correct_predictions_value': correct_predictions_value,
            'total_predictions_value': total_predictions_value,
            'correct_predictions_weight': correct_predictions_weight,
            'total_predictions_weight': total_predictions_weight,
            'accuracy_value': correct_predictions_value / total_predictions_value,
            'accuracy_weight': correct_predictions_weight / total_predictions_weight,
        }
        with open(f'{save_dir}/{os.path.basename(file_path)}_results.json', 'w') as f:
            json.dump(res, f, indent=4)

    overall_res = {'res_overall': overall_correct_predictions_value / overall_total_predictions_value,
                   'res_overall_weight': overall_correct_predictions_weight/ overall_total_predictions_weight}
    with open(f'{save_dir}/res_overall.json', 'w') as f:
        json.dump(overall_res, f, indent=4)
