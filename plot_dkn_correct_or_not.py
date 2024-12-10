import argparse
import json
import os
import itertools
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from knowledge_neurons import Dkn
from knowledge_neurons.utils import load_json_files_from_directory, initiate_model_tokenizer


def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Plot Degeneracy Analysis")

    # Define arguments
    parser.add_argument("--data_dir", type=str, default='temporal_res/1118_3')
    parser.add_argument("--model_name", type=str, default='gpt2')

    # Parse arguments
    args = parser.parse_args()
    new_acc_data_dir = f'{args.data_dir}/new_acc/model_direct'
    os.makedirs(new_acc_data_dir, exist_ok=True)

    def get_result():
        model_name = args.model_name
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.to('cuda')
        kn = Dkn(model, tokenizer, model_type='gpt')
        training_data = load_jsonl('Templama/train.jsonl')
        neuron_data = load_json_files_from_directory('temporal_res/1118_3', keyword='temporal_results')
        neuron_map = {uuid: data['dkn_cluster_2'] for uuid, data in neuron_data.items()}
        query_map = {}
        for sample in training_data:
            uuid = sample['id']
            date_prefix = f"In {sample['date']},"
            formatted_query = date_prefix + " " + sample['query'].replace('_X_.', '').strip()
            correct_answer = sample['answer'][0]['name']

            # Retrieve the inner dictionary for the uuid
            neuron_info = neuron_data.get(uuid, {})

            # Now retrieve dkn_cluster_2 from the inner dictionary
            dkn_cluster_2 = neuron_info.get('dkn_cluster_2', [])

            query_map[uuid] = {
                'formatted_query': formatted_query,
                'correct_answer': correct_answer,
                'dkn_cluster_2': dkn_cluster_2
            }
        grouped_queries = defaultdict(list)
        for uuid, data in query_map.items():
            neuron_count = len(data['dkn_cluster_2'])
            grouped_queries[neuron_count].append({**data, 'uuid':uuid})
        # Initialize a dictionary to store detailed results
        detailed_results = defaultdict(lambda: defaultdict(list))

        for neuron_count, queries in grouped_queries.items():
            for suppress_size in range(1, neuron_count + 1):
                for query_data in tqdm(queries):
                    # Extract necessary data from each query
                    formatted_query = query_data['formatted_query']
                    correct_answer = query_data['correct_answer']
                    neurons = query_data['dkn_cluster_2']

                    # Get all combinations of neurons to suppress
                    all_combinations = list(itertools.combinations(neurons, suppress_size))

                    # Randomly select one combination to suppress
                    selected_combination = random.choice(all_combinations)

                    # Merge selected neurons for suppression
                    suppressed_neurons = [neuron for sublist in selected_combination for neuron in sublist]

                    # Predict the answer
                    predicted_answer = kn.enhance_or_suppress_dkn_predict_answer(
                        query=formatted_query,
                        correct_answer=correct_answer,
                        neurons_list_2d=suppressed_neurons,
                        mode='suppress'
                    )

                    # Check if prediction is correct and record the result
                    is_correct = correct_answer in predicted_answer
                    detailed_results[neuron_count][suppress_size].append(is_correct)

        # Save the results to a JSON file
        with open(f'{new_acc_data_dir}/results.json', 'w') as json_file:
            json.dump(detailed_results, json_file, indent=4)


    # Function to calculate accuracy
    def calculate_accuracy(responses):
        return sum(responses) / len(responses)

    with open(f'{new_acc_data_dir}/results.json', 'r') as json_file:
        detailed_results = json.load(json_file)

    # Initialize data structures for the two categories
    partial_suppression_accuracy = {}
    full_suppression_accuracy = {}

    # Process the data
    for neuron_count, suppressions in detailed_results.items():
        neuron_count = int(neuron_count)
        for suppress_count, results in suppressions.items():
            suppress_count = int(suppress_count)
            accuracy = calculate_accuracy(results)
            if suppress_count < neuron_count:
                # Partial suppression
                partial_suppression_accuracy.setdefault(neuron_count, []).append((suppress_count, accuracy))
            else:
                # Full suppression
                full_suppression_accuracy[neuron_count] = accuracy

    # Determine the unique neuron counts in the results
    neuron_counts = sorted(map(int, detailed_results.keys()))
    num_plots = len(neuron_counts)

    # Determine the number of rows and columns for subplots
    num_rows = num_plots // 3 + (num_plots % 3 > 0)
    num_cols = min(num_plots, 3)

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle('Accuracy vs. Suppressed Neuron Count')

    # Flatten axes array for easy indexing
    if num_rows > 1:
        axs = [ax for row in axs for ax in row]

    # Iterate over the neuron counts and plot each one
    for i, neuron_count in enumerate(neuron_counts):
        ax = axs[i]

        # Prepare data for plotting
        x_values = list(detailed_results[str(neuron_count)].keys())  # Suppressed neuron counts
        y_values = [calculate_accuracy(detailed_results[str(neuron_count)][str(n)]) for n in x_values]

        # Plotting
        ax.plot(x_values, y_values, marker='o')
        ax.set_title(f'Original Neuron Set Size: {neuron_count}')
        ax.set_xlabel('Suppressed Neuron Count')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])  # Set y-axis to range from 0 to 1

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    fig.savefig(f'{new_acc_data_dir}/neuron_suppression_accuracy.png', dpi=300)
