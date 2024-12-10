import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer, LlamaForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from knowledge_neurons.utils import get_model_output, load_json_files_from_directory, load_jsonl_files_from_directory
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from knowledge_neurons import (
    model_type,
    Dkn,
)
from knowledge_neurons.utils import initiate_model_tokenizer


def aggregate_results(data_dir):
    aggregated_results = defaultdict(lambda: defaultdict(int))
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename)) as file:
            data = json.load(file)
            for key in data:
                for relation, value in data[key].items():
                    aggregated_results[key][relation] += value
    return aggregated_results


def convert_complex_to_triple(model, tokenizer, complex_text, model_type='llama'):
    """for llama2"""
    prompt = (
        f"Given a complex statement, simplify it into a straightforward sentence that clearly states the subject, predicate, and object. "
        "The simplified sentence should capture the key information from the complex statement in a concise format. Here are a few examples to illustrate the conversion:\n\n"
        "1. Complex: _X_, renowned for their career in sports, plays for a team or club. This association is a significant part of their professional journey, reflecting their skill and dedication in their field.\n"
        "   Simplified: Valentino Rossi plays for _X_.\n\n"
        "2. Complex: _X_, a key figure in their domain, holds an important position. This role is marked by responsibilities and influence, shaping the direction and success of their organization or country.\n"
        "   Simplified: Robert Mugabe holds the position of _X_.\n\n"
        "3. Complex: _X_, known for their intellectual and professional achievements, attended a notable institution. Their time at this institution played a crucial role in shaping their career and perspectives.\n"
        "   Simplified: Malala Yousafzai attended _X_.\n\n"
        "4. Complex: As the chair of _X_, this individual is a prominent figure in the business world. This position highlights their leadership and strategic vision in guiding the company towards growth and innovation.\n"
        "   Simplified: _X_ is the chair of bp.\n\n"
        "5. Complex: _X_, a figure of significant influence and ideals, is a member of a notable group or organization. Their membership reflects their commitment to certain values and goals, contributing to the group's impact.\n"
        "   Simplified: Alexei Navalny is a member of the _X_.\n\n"
        "6. Complex: _X_ works for a notable company or organization, playing a pivotal role in driving innovation and excellence in their field.\n"
        "   Simplified: Elon Musk works for _X_.\n\n"
        "7. Complex: As the head of the government of _X_, this political figure plays a central role in shaping the nation's policies and international relations.\n"
        "   Simplified: _X_ is the head of the government of France.\n\n"
        "8. Complex: The head coach of _X_ is known for their expertise in coaching, with leadership and strategic skills crucial in steering the team to success.\n"
        "   Simplified: _X_ is the head coach of Philadelphia Eagles.\n\n"
        "9. Complex: _X_, a significant player in its sector, is owned by a larger entity. This ownership structure is key to understanding the company's operations and market influence.\n"
        "   Simplified: Google is owned by _X_.\n\n"
        "Now, given the following complex statement, apply the same process to simplify it:\n"
        f"{complex_text}"
    )
    if model_type == 'llama':
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to('cuda')  # Ensure inputs are on the same device as the model
        output_ids = model.generate(inputs.input_ids, max_new_tokens=50)
        new_ids = output_ids[0]

        full_generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        generated_text = full_generated_text[len(prompt):].strip()
        final_text = generated_text.replace("Simplified:", "").strip()
        return final_text
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to('cuda')  # Ensure inputs are on the same device as the model
        output_ids = model.generate(inputs.input_ids, max_new_tokens=50)
        new_ids = output_ids[0]

        full_generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        generated_text = full_generated_text[len(prompt):].strip()
        final_text = generated_text.replace("Simplified:", "").strip()
        return final_text
        # input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        # output = model.generate(input_ids, max_new_tokens=50)
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # return generated_text

def initial_data(args):
    RESULTS_DIR = Path(args.neurons_result_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(args.seed)
    data_file = args.data_file
    model, tokenizer = initiate_model_tokenizer(args.model_name)
    if args.model_name == 'gpt2':
        kn = Dkn(model, tokenizer, model_type='gpt')
        neurons_result = load_json_files_from_directory(args.neurons_result_dir,'temporal_results')
        NUM_REPLICAS = torch.cuda.device_count()
    else:
        kn = Dkn(model, tokenizer, model_type='llama')
        neurons_result = load_jsonl_files_from_directory(args.neurons_result_dir, 'temporal_results')
        NUM_REPLICAS = 1
    dataset = []
    with open(data_file, 'r') as file:
        for line in file:
            data_item = json.loads(line)
            if data_item['uuid'] in neurons_result:
                dataset.append(data_item)
    # Group data by relation_name
    if not os.path.exists(f'{args.wrong_fact_dir}/data_grouped_by_relation.pkl'):
        os.makedirs(args.wrong_fact_dir, exist_ok=True)
        data_grouped_by_relation = defaultdict(list)
        for item in dataset:
            relation_name = item['relation_name']
            data_grouped_by_relation[relation_name].append(item)
        with open(f'{args.wrong_fact_dir}/data_grouped_by_relation.pkl', 'wb') as f:
            pickle.dump(data_grouped_by_relation, f)
    else:
        with open(f'{args.wrong_fact_dir}/data_grouped_by_relation.pkl', 'rb') as f:
            data_grouped_by_relation = pickle.load(f)

    if not os.path.exists(f'{args.wrong_fact_dir}/neurons_result_grouped.pkl'):
        neurons_result_grouped = defaultdict(dict)
        for item in dataset:
            relation_name = item['relation_name']
            uuid = item['uuid']
            neurons_result_grouped[relation_name][uuid] = neurons_result.get(uuid, {})
        with open(f'{args.wrong_fact_dir}/neurons_result_grouped.pkl', 'wb') as f:
            pickle.dump(neurons_result_grouped, f)
    else:
        with open(f'{args.wrong_fact_dir}/neurons_result_grouped.pkl', 'rb') as f:
            neurons_result_grouped = pickle.load(f)
    if use_complex_text:
        return data_grouped_by_relation, neurons_result_grouped, kn, NUM_REPLICAS, model, tokenizer
    else:
        return data_grouped_by_relation, neurons_result_grouped, kn, NUM_REPLICAS


def not_split_check(args, check_in_real=False, test_size=0.01):
    """单个的fact-checking实验，无法用在实际情况中，但是更容易证明效果。"""

    if use_complex_text:
        data_grouped_by_relation, neurons_result_grouped, kn, NUM_REPLICAS, model, tokenizer= initial_data(args)
        # triple_sentence = llama_convert_complex_to_triple(model, tokenizer, "_X_, renowned for their career in sports, plays for a team or club. This association is a significant part of their professional journey, reflecting their skill and dedication in their field.")
    else:
        data_grouped_by_relation, neurons_result_grouped, kn, NUM_REPLICAS = initial_data(args)

    correct_true_by_relation_d = defaultdict(int)
    total_true_by_relation = defaultdict(int)
    correct_false_by_relation_d = defaultdict(int)
    total_false_by_relation = defaultdict(int)
    correct_true_by_relation_kn = defaultdict(int)
    correct_false_by_relation_kn = defaultdict(int)
    correct_true_by_relation_directly_use_PLMs = defaultdict(int)
    correct_false_by_relation_directly_use_PLMs = defaultdict(int)

    for relation_name, items in data_grouped_by_relation.items():
        _, test_items = train_test_split(items, test_size=test_size, random_state=42)

        test_indices = list(range(len(test_items)))
        test_indices = test_indices[args.local_rank: len(test_items): NUM_REPLICAS]
        # for i, test_idx in enumerate(tqdm(test_indices, position=args.local_rank)):
        for i, test_idx in enumerate(tqdm(test_indices, position=0)):
            item = test_items[test_idx]
            uuid = item['uuid']
            if use_complex_text:
                if args.model_name == 'gpt2':
                    triple_sentence = convert_complex_to_triple(model, tokenizer, item['sentences'][0], model_type='gpt')
                else:
                    triple_sentence = convert_complex_to_triple(model, tokenizer, item['sentences'][0], model_type='llama')
                item['sentences'][0] = triple_sentence

            degenerate_kn_3d = neurons_result_grouped[relation_name][uuid]['dkn_cluster_2']

            degenerate_kn = [neuron for subset in degenerate_kn_3d for neuron in subset]
            if args.model_name == 'gpt2':
                k_neurons = neurons_result_grouped[relation_name][uuid]["neurons"]
            else:
                k_neurons = neurons_result_grouped[relation_name][uuid]["kn"]

            # Perform fact-checking with the specific neurons
            d_correct_true, d_correct_false = kn.test_detection_system(
                item=item, neurons=degenerate_kn,
                threshold=args.threshold,
                batch_size=args.batch_size, steps=args.steps,
                baseline_vector_path=args.baseline_vector_path,
                score_path=score_dir
            )
            correct_true_by_relation_d[relation_name] += d_correct_true
            correct_false_by_relation_d[relation_name] += d_correct_false

            k_correct_true, k_correct_false = kn.test_detection_system(
                item=item, neurons=k_neurons,
                threshold=5e-6,
                batch_size=args.batch_size, steps=args.steps,
                baseline_vector_path=args.baseline_vector_path,
                score_path=score_dir
            )
            correct_true_by_relation_kn[relation_name] += k_correct_true
            correct_false_by_relation_kn[relation_name] += k_correct_false
            # # Perform fact-checking directly using PLMs
            correct_true_directly_use_PLMs, correct_false_directly_use_PLMs = kn.test_detection_system_PLMs(item=item)
            correct_true_by_relation_directly_use_PLMs[relation_name] += correct_true_directly_use_PLMs
            correct_false_by_relation_directly_use_PLMs[relation_name] += correct_false_directly_use_PLMs
            total_true_by_relation[relation_name] += 1
            total_false_by_relation[relation_name] += 1
        tmp_res = {
            "correct_true_by_relation_d": dict(correct_true_by_relation_d),
            "correct_false_by_relation_d": dict(correct_false_by_relation_d),
            "correct_true_by_relation_kn": dict(correct_true_by_relation_kn),
            "correct_false_by_relation_kn": dict(correct_false_by_relation_kn),
            "correct_true_by_relation_PLMs": dict(correct_true_by_relation_directly_use_PLMs),
            "correct_false_by_relation_PLMs": dict(correct_false_by_relation_directly_use_PLMs),
            "total_true_by_relation": dict(total_true_by_relation),
            "total_false_by_relation": dict(total_false_by_relation),
        }
        os.makedirs(f'{hallucination_dir}/tmp', exist_ok=True)
        with open(f'{hallucination_dir}/tmp/NOT_split{args.local_rank}.json', 'w') as f:
            json.dump(tmp_res, f)

        # Save the results
        result_dict = {
            "correct_true_by_relation_d": dict(correct_true_by_relation_d),
            "correct_false_by_relation_d": dict(correct_false_by_relation_d),
            "correct_true_by_relation_kn": dict(correct_true_by_relation_kn),
            "correct_false_by_relation_kn": dict(correct_false_by_relation_kn),
            "correct_true_by_relation_PLMs": dict(correct_true_by_relation_directly_use_PLMs),
            "correct_false_by_relation_PLMs": dict(correct_false_by_relation_directly_use_PLMs),
            "total_true_by_relation": dict(total_true_by_relation),
            "total_false_by_relation": dict(total_false_by_relation),
        }
        with open(f'{hallucination_dir}/split{args.local_rank}.json', 'w') as f:
            json.dump(result_dict, f)


def train_test_split_check(args, test_size=0.01):
    """
    """
    if use_complex_text:
        data_grouped_by_relation, neurons_result_grouped, kn, NUM_REPLICAS, model, tokenizer= initial_data(args)
        # triple_sentence = llama_convert_complex_to_triple(model, tokenizer, "_X_, renowned for their career in sports, plays for a team or club. This association is a significant part of their professional journey, reflecting their skill and dedication in their field.")
    else:
        data_grouped_by_relation, neurons_result_grouped, kn, NUM_REPLICAS = initial_data(args)
    # Split data and count neurons for each relation_name
    filtered_dkn_by_relation = defaultdict(set)
    filtered_kn_by_relation = defaultdict(set)
    train_test_splits = {}
    for relation_name, items in data_grouped_by_relation.items():
        train_items, test_items = train_test_split(items, test_size=test_size, random_state=42)
        train_test_splits[relation_name] = (train_items, test_items)
        # For training items, count the neurons and filter out those exceeding the threshold
        degenerate_kn_counts = defaultdict(int)
        kn_counts = defaultdict(int)  # New counter for neurons data
        for item in train_items:
            uuid = item['uuid']
            # For neurons_result
            if uuid in neurons_result_grouped[relation_name] and neurons_result_grouped[relation_name][uuid] != {}:
                # x = neurons_result_grouped[relation_name][uuid]
                degenerate_kn = neurons_result_grouped[relation_name][uuid]['dkn_cluster_2']
                if args.model_name == 'gpt2': 
                    k_neurons = neurons_result_grouped[relation_name][uuid]["neurons"]
                else:
                    k_neurons = neurons_result_grouped[relation_name][uuid]["kn"]
                for neuron in [tuple(neuron_pair) for pair in degenerate_kn for neuron_pair in pair]:
                    degenerate_kn_counts[neuron] += 1
                for neuron in k_neurons:
                    kn_counts[tuple(neuron)] += 1

        # Filter out neurons based on threshold
        degenerate_kn_threshold_count = int(max(degenerate_kn_counts.values()) * args.threshold_filter_DKN)
        kn_threshold_count = int(
            max(kn_counts.values()) * args.threshold_filter_DKN) # 控制二者数目一致，才能进行对比。
        filtered_degenerate_kn = {neuron for neuron, count in degenerate_kn_counts.items() if count > degenerate_kn_threshold_count}
        filtered_kn = {neuron for neuron, count in kn_counts.items() if
                                             count > kn_threshold_count}  # New filtering for neurons data

        filtered_dkn_by_relation[relation_name].update(filtered_degenerate_kn)
        filtered_kn_by_relation[relation_name].update(filtered_kn)
    # true is true_answer, false is false_answer. correct is model has checked correctly
    correct_true_by_relation_d = defaultdict(int)
    total_true_by_relation = defaultdict(int)
    correct_false_by_relation_d = defaultdict(int)
    total_false_by_relation = defaultdict(int)
    correct_true_by_relation_kn = defaultdict(int)
    correct_false_by_relation_kn = defaultdict(int)
    correct_true_by_relation_directly_use_PLMs = defaultdict(int)
    correct_false_by_relation_directly_use_PLMs = defaultdict(int)
    with open(f'{score_dir}/score_debug.json', 'w') as f:
        pass
    for relation_name, items in data_grouped_by_relation.items():
        _, test_items = train_test_splits[relation_name]
        test_indices = list(range(len(test_items)))
        test_indices = test_indices[args.local_rank: len(test_items): NUM_REPLICAS]
        all_uuids_processed_successfully = True
        for i, test_idx in enumerate(tqdm(test_indices, position=0)):
        # for i, test_idx in enumerate(tqdm(test_indices, position=args.local_rank)):
            item = test_items[test_idx]
            if use_complex_text:
                if args.model_name == 'gpt2':
                    triple_sentence = convert_complex_to_triple(model, tokenizer, item['sentences'][0], model_type='gpt')
                else:
                    triple_sentence = convert_complex_to_triple(model, tokenizer, item['sentences'][0], model_type='llama')
                if '_X_' not in item['sentences'][0]:
                    total_true_by_relation[relation_name] += 1
                    total_false_by_relation[relation_name] += 1
                    continue
                item['sentences'][0] = triple_sentence
            degenerate_neurons = filtered_dkn_by_relation[relation_name]
            k_neurons = filtered_kn_by_relation[relation_name]  # 有些relation-name过滤的结果完全一致。
            # Perform fact-checking with the filtered neurons
            d_correct_true, d_correct_false = kn.test_detection_system(
                item=item, neurons=degenerate_neurons,
                threshold=args.threshold,
                batch_size=args.batch_size, steps=args.steps,
                baseline_vector_path=args.baseline_vector_path,
                score_path=score_dir
            )
            correct_true_by_relation_d[relation_name] += d_correct_true
            correct_false_by_relation_d[relation_name] += d_correct_false

            k_correct_true, k_correct_false = kn.test_detection_system(
                item=item, neurons=k_neurons,
                threshold=5e-6,
                batch_size=args.batch_size, steps=args.steps,
                baseline_vector_path=args.baseline_vector_path,
                score_path=score_dir
            )
            correct_true_by_relation_kn[relation_name] += k_correct_true
            correct_false_by_relation_kn[relation_name] += k_correct_false
            # # Perform fact-checking directly using PLMs
            correct_true_directly_use_PLMs, correct_false_directly_use_PLMs = kn.test_detection_system_PLMs(item=item)
            correct_true_by_relation_directly_use_PLMs[relation_name] += correct_true_directly_use_PLMs
            correct_false_by_relation_directly_use_PLMs[relation_name] += correct_false_directly_use_PLMs
            total_true_by_relation[relation_name] += 1
            total_false_by_relation[relation_name] += 1
        if all_uuids_processed_successfully:
            tmp_res = {
                "correct_true_by_relation_d": dict(correct_true_by_relation_d),
                "correct_false_by_relation_d": dict(correct_false_by_relation_d),
                "correct_true_by_relation_kn": dict(correct_true_by_relation_kn),
                "correct_false_by_relation_kn": dict(correct_false_by_relation_kn),
                "correct_true_by_relation_PLMs": dict(correct_true_by_relation_directly_use_PLMs),
                "correct_false_by_relation_PLMs": dict(correct_false_by_relation_directly_use_PLMs),
                "total_true_by_relation": dict(total_true_by_relation),
                "total_false_by_relation": dict(total_false_by_relation),
            }
            os.makedirs(f'{hallucination_dir}/tmp', exist_ok=True)
            with open(f'{hallucination_dir}/tmp/hallucination_temp_split{args.local_rank}.json', 'w') as f:
                json.dump(tmp_res, f)

    # Save the results
    result_dict = {
                "correct_true_by_relation_d": dict(correct_true_by_relation_d),
                "correct_false_by_relation_d": dict(correct_false_by_relation_d),
                "correct_true_by_relation_kn": dict(correct_true_by_relation_kn),
                "correct_false_by_relation_kn": dict(correct_false_by_relation_kn),
                "correct_true_by_relation_PLMs": dict(correct_true_by_relation_directly_use_PLMs),
                "correct_false_by_relation_PLMs": dict(correct_false_by_relation_directly_use_PLMs),
                "total_true_by_relation": dict(total_true_by_relation),
                "total_false_by_relation": dict(total_false_by_relation),
            }
    with open(f'{hallucination_dir}/hallucination_split{args.local_rank}.json', 'w') as f:
        json.dump(result_dict, f)
def aggregate_json_files_from_directory(dir_path, keyword):
    aggregated_data = {}
    for filename in os.listdir(dir_path):
        if keyword in filename:
            with open(os.path.join(dir_path, filename), 'r') as file:
                data = json.load(file)

                for key, value_dict in data.items():
                    if key not in aggregated_data:
                        aggregated_data[key] = {}

                    for sub_key, sub_value in value_dict.items():
                        aggregated_data[key].setdefault(sub_key, 0)
                        aggregated_data[key][sub_key] += sub_value

    return aggregated_data


def calculate_metrics(aggregated_data):
    metrics = {}
    overall = {'d': {'TP': 0, 'FP': 0, 'FN': 0},
               'kn': {'TP': 0, 'FP': 0, 'FN': 0},
               'PLMs': {'TP': 0, 'FP': 0, 'FN': 0}}

    for situation in ['d', 'kn', 'PLMs']:
        metrics[situation] = {}
        for relation in aggregated_data[f'correct_true_by_relation_{situation}']:
            TP = aggregated_data[f'correct_true_by_relation_{situation}'][relation]
            FP = aggregated_data['total_false_by_relation'][relation] - aggregated_data[f'correct_false_by_relation_{situation}'][relation]
            FN = aggregated_data['total_true_by_relation'][relation] - TP

            # Update overall counts for each situation
            overall[situation]['TP'] += TP
            overall[situation]['FP'] += FP
            overall[situation]['FN'] += FN

            P = TP / (TP + FP) if (TP + FP) != 0 else 0
            R = TP / (TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * (P * R) / (P + R) if (P + R) != 0 else 0

            # metrics[situation][relation] = {'P': P, 'R': R, 'F1': F1}

    # Calculate overall metrics for each situation
    for situation in overall:
        o_TP = overall[situation]['TP']
        o_FP = overall[situation]['FP']
        o_FN = overall[situation]['FN']

        o_P = o_TP / (o_TP + o_FP) if (o_TP + o_FP) != 0 else 0
        o_R = o_TP / (o_TP + o_FN) if (o_TP + o_FN) != 0 else 0
        o_F1 = 2 * (o_P * o_R) / (o_P + o_R) if (o_P + o_R) != 0 else 0

        metrics[situation]['overall'] = {'P': o_P, 'R': o_R, 'F1': o_F1}

    return metrics


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        "Use the Pararel dataset to extract knowledge neurons from a Language Model"
    )
    parser.add_argument(
        "--local-rank", help="local rank for multigpu processing", type=int, default=0
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="EleutherAI/gpt-j-6b",
        default='gpt2',
        # default='/home/chenyuheng/KN2/Llama/Llama7bChat',
        # default="meta-llama/Llama-2-70b-chat-hf",
        # default='meta-llama/Llama-2-13b-chat-hf',
    )

    parser.add_argument('--wrong_fact_dir', type=str,
                        default='/home/chenyuheng/KN2/kn2/datasets/wrong_fact_dataset/temporal/train-gpt2',
                        # default='datasets/wrong_fact_dataset/temporal/train-llama-complex',
                        )
    parser.add_argument('--neurons_result_dir', type=str,
                        default='temporal_res/1118_3',
                        # default='temporal_res/llama7b_1226/res_wo_acc',
                        )
    parser.add_argument('--data_file', type=str,
                        default='/home/chenyuheng/KN2/kn2/datasets/wrong_fact_dataset/temporal/train-gpt2/lama.jsonl',
                        )
    parser.add_argument("--batch_size", type=int, default=5,
                        help='# assert steps % batch_size == 0'
                             '# n_batches = steps // batch_size，'
                             '此外，对应gpt类，因为要re-tokenize new inputs，所以需要内存更多')
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="number of steps to run the integrated gradient calculation for",
    )

    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--use_complex_text', default=False, action='store_true')
    parser.add_argument('--baseline_vector_path', type=str,
                        default=None,
                        )
    parser.add_argument('--threshold', type=float, default=5e-7)
    parser.add_argument('--threshold_filter_DKN', type=float,default=0.5)
    parser.add_argument('--hallucination_parent_dir', type=str,
                        default='Hallucination_res/gpt2/Golden_label',
                        # default='Hallucination_res/llama/USE_complex_text_new',
                        # default='Hallucination_res/llama/NOT_USE_complex_text',
                        )
    parser.add_argument('--run_mode', type=str,
                        # default='train_test_split_check',
                        # default='not_split_check',
                        default='donot_check',
                        )


    args = parser.parse_args()
    use_complex_text = args.use_complex_text

    torch.cuda.set_device(args.local_rank)  # 这段代码保证分布式运行。
    score_dir = f'{args.neurons_result_dir}/hyper_params'
    os.makedirs(score_dir, exist_ok=True)
    hallucination_parent_dir = args.hallucination_parent_dir
    if 'gpt2' in args.model_name:
        hallucination_dir = f'{hallucination_parent_dir}/{args.threshold}_{args.threshold_filter_DKN}'
    else:
        if use_complex_text:
            hallucination_dir = f'{hallucination_parent_dir}/{args.threshold}_{args.threshold_filter_DKN}'
        else:
            hallucination_dir = f'{hallucination_parent_dir}/{args.threshold}_{args.threshold_filter_DKN}'

    os.makedirs(hallucination_dir, exist_ok=True)
    run_mode = args.run_mode
    if run_mode == 'train_test_split_check':
        train_test_split_check(args)
    elif run_mode == 'not_split_check':
        not_split_check(args)
    else:
        pass

    # usage
    # hallucination_dir = '/home/chenyuheng/KN2/kn2/Hallucination_res/llama/NOT_complex_text_130/5e-07_0.5/tmp'
    # hallucination_dir = '/home/chenyuheng/KN2/kn2/Hallucination_res/llama/USE_complex_text_GOLDEN/5e-07_0.5/tmp'
    hallucination_dir = '/home/chenyuheng/KN2/kn2/Hallucination_res/llama/USE_complex_text_GOLDEN/5e-07_0.5'
    aggregated_results = aggregate_json_files_from_directory(dir_path=hallucination_dir,
                                                             # keyword='hallucination'
                                                             # keyword='hallucination_split'
                                                             keyword='split'
                                                             )
    metrics_results = calculate_metrics(aggregated_results)

    with open(f'{hallucination_dir}/metrics_results.json', 'w') as outfile:
        json.dump(metrics_results, outfile, indent=4)
    # Example usage with your data
    # complex_sentence = "During a landmark conference on technology and innovation, experts highlighted the contributions of Tokyo in the field of robotics. Tokyo, known as the capital city of South Korea, has been pivotal in driving advancements in artificial intelligence and automation."
    # # answer = "Tampa Bay Buccaneers"
    # model, tokenizer = initiate_model_tokenizer(args.model_name)
    # query, answer = extract_answer_from_sentence(model, tokenizer, complex_sentence)
    # print(query, answer)

