# import argparse
# import json
# import os
# import pickle
# import random
# from collections import defaultdict
# from pathlib import Path
#
# from knowledge_neurons.utils import get_model_output, load_json_files_from_directory
# import torch
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
#
# from knowledge_neurons import (
#     model_type,
#     Dkn,
# )
# from knowledge_neurons.utils import initiate_model_tokenizer
#
#
# def aggregate_results(data_dir):
#     aggregated_results = defaultdict(lambda: defaultdict(int))
#     for filename in os.listdir(data_dir):
#         with open(os.path.join(data_dir, filename)) as file:
#             data = json.load(file)
#             for key in data:
#                 for relation, value in data[key].items():
#                     aggregated_results[key][relation] += value
#     return aggregated_results
#
# def initial_data(args):
#     RESULTS_DIR = Path(args.neurons_result_dir)
#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     random.seed(args.seed)
#     data_file = f'{args.wrong_fact_dir}/lama.jsonl'
#     model, tokenizer = initiate_model_tokenizer(args.model_name)
#     kn = Dkn(model, tokenizer, model_type=model_type(args.model_name))
#     NUM_REPLICAS = torch.cuda.device_count()
#     neurons_result = load_json_files_from_directory(args.neurons_result_dir,'temporal_results')
#     dataset = []
#     with open(data_file, 'r') as file:
#         for line in file:
#             data_item = json.loads(line)
#             if data_item['uuid'] in neurons_result:
#                 dataset.append(data_item)
#     # Group data by relation_name
#     if not os.path.exists(f'{args.wrong_fact_dir}/data_grouped_by_relation.pkl'):
#         data_grouped_by_relation = defaultdict(list)
#         for item in dataset:
#             relation_name = item['relation_name']
#             data_grouped_by_relation[relation_name].append(item)
#         with open(f'{args.wrong_fact_dir}/data_grouped_by_relation.pkl', 'wb') as f:
#             pickle.dump(data_grouped_by_relation, f)
#     else:
#         with open(f'{args.wrong_fact_dir}/data_grouped_by_relation.pkl', 'rb') as f:
#             data_grouped_by_relation = pickle.load(f)
#
#     if not os.path.exists(f'{args.wrong_fact_dir}/neurons_result_grouped.pkl'):
#         neurons_result_grouped = defaultdict(dict)
#         for item in dataset:
#             relation_name = item['relation_name']
#             uuid = item['uuid']
#             neurons_result_grouped[relation_name][uuid] = neurons_result.get(uuid, {})
#         with open(f'{args.wrong_fact_dir}/neurons_result_grouped.pkl', 'wb') as f:
#             pickle.dump(neurons_result_grouped, f)
#     else:
#         with open(f'{args.wrong_fact_dir}/neurons_result_grouped.pkl', 'rb') as f:
#             neurons_result_grouped = pickle.load(f)
#     return data_grouped_by_relation, neurons_result_grouped, kn, NUM_REPLICAS
#
# def train_test_split_check(args):
#     """
#     应当说明:性能不一定比基于原来的假设的效果更好，因为这是在全集上找占比最大的。原始的方法其实噪声更少，但是丢失了大量信息。在多条数据上，这个问题就被解决了。
#     """
#     data_grouped_by_relation, neurons_result_grouped, kn, NUM_REPLICAS = initial_data(args)
#     # Split data and count neurons for each relation_name
#     filtered_dkn_by_relation = defaultdict(set)
#     filtered_kn_by_relation = defaultdict(set)
#     for relation_name, items in data_grouped_by_relation.items():
#         train_items, _ = train_test_split(items, test_size=0.1, random_state=42)
#
#         # For training items, count the neurons and filter out those exceeding the threshold
#         degenerate_kn_counts = defaultdict(int)
#         kn_counts = defaultdict(int)  # New counter for neurons data
#         for item in train_items:
#             uuid = item['uuid']
#             # For neurons_result
#             if uuid in neurons_result_grouped[relation_name] and neurons_result_grouped[relation_name][uuid] != {}:
#                 # x = neurons_result_grouped[relation_name][uuid]
#                 degenerate_kn = neurons_result_grouped[relation_name][uuid]['dkn_cluster_2']
#                 k_neurons = neurons_result_grouped[relation_name][uuid]["neurons"]
#                 for neuron in [tuple(neuron_pair) for pair in degenerate_kn for neuron_pair in pair]:
#                     degenerate_kn_counts[neuron] += 1
#                 for neuron in k_neurons:
#                     kn_counts[tuple(neuron)] += 1
#
#         # Filter out neurons based on threshold
#         degenerate_kn_threshold_count = int(max(degenerate_kn_counts.values()) * args.threshold_filter_DKN)
#         kn_threshold_count = int(max(kn_counts.values()) * args.threshold_filter_DKN)
#
#         filtered_degenerate_kn = {neuron for neuron, count in degenerate_kn_counts.items() if
#                                   count > degenerate_kn_threshold_count}
#         filtered_kn = {neuron for neuron, count in kn_counts.items() if count > kn_threshold_count}
#
#         filtered_dkn_by_relation[relation_name].update(filtered_degenerate_kn)
#         filtered_kn_by_relation[relation_name].update(filtered_kn)
#     # true is true_answer, false is false_answer. correct is model has checked correctly
#     correct_true_by_relation_d = defaultdict(int)
#     total_true_by_relation = defaultdict(int)
#     correct_false_by_relation_d = defaultdict(int)
#     total_false_by_relation = defaultdict(int)
#     correct_true_by_relation_kn = defaultdict(int)
#     correct_false_by_relation_kn = defaultdict(int)
#     correct_true_by_relation_directly_use_PLMs = defaultdict(int)
#     correct_false_by_relation_directly_use_PLMs = defaultdict(int)
#     with open(f'{score_dir}/score_debug.json', 'w') as f:
#         pass
#     for relation_name, items in data_grouped_by_relation.items():
#         _, test_items = train_test_split(items, test_size=0.1, random_state=42)
#         test_indices = list(range(len(test_items)))
#         test_indices = test_indices[args.local_rank: len(test_items): NUM_REPLICAS]
#         all_uuids_processed_successfully = True
#         for i, test_idx in enumerate(tqdm(test_indices, position=args.local_rank)):
#             item = test_items[test_idx]
#             try:
#                 degenerate_neurons = filtered_dkn_by_relation[relation_name]
#                 k_neurons = filtered_kn_by_relation[relation_name]  # 有些relation-name过滤的结果完全一致。
#                 # Perform fact-checking with the filtered neurons
#                 d_correct_true, d_correct_false = kn.test_detection_system(
#                     item=item, neurons=degenerate_neurons,
#                     threshold=args.threshold,
#                     batch_size=args.batch_size, steps=args.steps,
#                     baseline_vector_path=args.baseline_vector_path,
#                     score_path=score_dir
#                 )
#                 correct_true_by_relation_d[relation_name] += d_correct_true
#                 correct_false_by_relation_d[relation_name] += d_correct_false
#
#                 k_correct_true, k_correct_false = kn.test_detection_system(
#                     item=item, neurons=k_neurons,
#                     threshold=args.threshold,
#                     batch_size=args.batch_size, steps=args.steps,
#                     baseline_vector_path=args.baseline_vector_path,
#                     score_path=score_dir
#                 )
#                 correct_true_by_relation_kn[relation_name] += k_correct_true
#                 correct_false_by_relation_kn[relation_name] += k_correct_false
#                 # # Perform fact-checking directly using PLMs
#                 # correct_true_directly_use_PLMs, correct_false_directly_use_PLMs = kn.test_detection_system_PLMs(item=item)
#                 # correct_true_by_relation_directly_use_PLMs[relation_name] += correct_true_directly_use_PLMs
#                 # correct_false_by_relation_directly_use_PLMs[relation_name] += correct_false_directly_use_PLMs
#                 total_true_by_relation[relation_name] += 1
#                 total_false_by_relation[relation_name] += 1
#             except torch.cuda.OutOfMemoryError:
#                 print(f"Skipping {uuid} due to CUDA out of memory error.")
#                 all_uuids_processed_successfully = False
#                 continue
#         if all_uuids_processed_successfully:
#             tmp_res = {
#                 "correct_true_by_relation_d": dict(correct_true_by_relation_d),
#                 "correct_false_by_relation_d": dict(correct_false_by_relation_d),
#                 "correct_true_by_relation_kn": dict(correct_true_by_relation_kn),
#                 "correct_false_by_relation_kn": dict(correct_false_by_relation_kn),
#                 "correct_true_by_relation_PLMs": dict(correct_true_by_relation_directly_use_PLMs),
#                 "correct_false_by_relation_PLMs": dict(correct_false_by_relation_directly_use_PLMs),
#                 "total_true_by_relation": dict(total_true_by_relation),
#                 "total_false_by_relation": dict(total_false_by_relation),
#             }
#             os.makedirs(f'{hallucination_dir}/temp3', exist_ok=True)
#             with open(f'{args.neurons_result_dir}/temp3/hallucination_temp_split{args.local_rank}.json', 'w') as f:
#                 json.dump(tmp_res, f)
#
#     # Save the results
#     result_dict = {
#                 "correct_true_by_relation_d": dict(correct_true_by_relation_d),
#                 "correct_false_by_relation_d": dict(correct_false_by_relation_d),
#                 "correct_true_by_relation_kn": dict(correct_true_by_relation_kn),
#                 "correct_false_by_relation_kn": dict(correct_false_by_relation_kn),
#                 "correct_true_by_relation_PLMs": dict(correct_true_by_relation_directly_use_PLMs),
#                 "correct_false_by_relation_PLMs": dict(correct_false_by_relation_directly_use_PLMs),
#                 "total_true_by_relation": dict(total_true_by_relation),
#                 "total_false_by_relation": dict(total_false_by_relation),
#             }
#     with open(f'{hallucination_dir}/hallucination_split{args.local_rank}.json', 'w') as f:
#         json.dump(result_dict, f)
#
# def aggregate_json_files_from_directory(dir_path, keyword):
#     aggregated_data = {}
#     for filename in os.listdir(dir_path):
#         if keyword in filename:
#             with open(os.path.join(dir_path, filename), 'r') as file:
#                 data = json.load(file)
#
#                 for key, value_dict in data.items():
#                     if key not in aggregated_data:
#                         aggregated_data[key] = {}
#
#                     for sub_key, sub_value in value_dict.items():
#                         aggregated_data[key].setdefault(sub_key, 0)
#                         aggregated_data[key][sub_key] += sub_value
#
#     return aggregated_data
#
#
# def calculate_metrics(aggregated_data):
#     metrics = {}
#     overall = {'d': {'TP': 0, 'FP': 0, 'FN': 0},
#                'kn': {'TP': 0, 'FP': 0, 'FN': 0},
#                'PLMs': {'TP': 0, 'FP': 0, 'FN': 0}}
#
#     for situation in ['d', 'kn', 'PLMs']:
#         metrics[situation] = {}
#         for relation in aggregated_data[f'correct_true_by_relation_{situation}']:
#             TP = aggregated_data[f'correct_true_by_relation_{situation}'][relation]
#             FP = aggregated_data['total_false_by_relation'][relation] - aggregated_data[f'correct_false_by_relation_{situation}'][relation]
#             FN = aggregated_data['total_true_by_relation'][relation] - TP
#
#             # Update overall counts for each situation
#             overall[situation]['TP'] += TP
#             overall[situation]['FP'] += FP
#             overall[situation]['FN'] += FN
#
#             P = TP / (TP + FP) if (TP + FP) != 0 else 0
#             R = TP / (TP + FN) if (TP + FN) != 0 else 0
#             F1 = 2 * (P * R) / (P + R) if (P + R) != 0 else 0
#
#             # metrics[situation][relation] = {'P': P, 'R': R, 'F1': F1}
#
#     # Calculate overall metrics for each situation
#     for situation in overall:
#         o_TP = overall[situation]['TP']
#         o_FP = overall[situation]['FP']
#         o_FN = overall[situation]['FN']
#
#         o_P = o_TP / (o_TP + o_FP) if (o_TP + o_FP) != 0 else 0
#         o_R = o_TP / (o_TP + o_FN) if (o_TP + o_FN) != 0 else 0
#         o_F1 = 2 * (o_P * o_R) / (o_P + o_R) if (o_P + o_R) != 0 else 0
#
#         metrics[situation]['overall'] = {'P': o_P, 'R': o_R, 'F1': o_F1}
#
#     return metrics
#
#
# if __name__ == "__main__":
#     # parse arguments
#     parser = argparse.ArgumentParser(
#         "Use the Pararel dataset to extract knowledge neurons from a Language Model"
#     )
#     parser.add_argument(
#         "--local-rank", help="local rank for multigpu processing", type=int, default=0
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         # default="EleutherAI/gpt-j-6b",
#         default='gpt2',
#         # default='/home/chenyuheng/KN2/Llama/Llama7bChat',
#         # default="meta-llama/Llama-2-70b-chat-hf",
#         # default='meta-llama/Llama-2-13b-chat-hf',
#     )
#
#     parser.add_argument('--wrong_fact_dir', type=str,
#                         default='datasets/wrong_fact_dataset/temporal/train',
#                         # default='datasets/correspond_dataset/wrong_fact_zh.json',
#                         )
#     parser.add_argument('--neurons_result_dir', type=str,
#                         default='temporal_res/1118_3',
#                         )
#     parser.add_argument("--batch_size", type=int, default=5,
#                         help='# assert steps % batch_size == 0'
#                              '# n_batches = steps // batch_size，'
#                              '此外，对应gpt类，因为要re-tokenize new inputs，所以需要内存更多')
#     parser.add_argument(
#         "--steps",
#         type=int,
#         default=5,
#         help="number of steps to run the integrated gradient calculation for",
#     )
#
#     parser.add_argument('--seed', type=int, default=42, help="random seed")
#     parser.add_argument('--baseline_vector_path', type=str,
#                         default=None,
#                         )
#     parser.add_argument('--threshold', type=float, default=5e-7)
#     # parser.add_argument('--threshold_KN', type=float, default=5e-6)
#     parser.add_argument('--threshold_filter_DKN', type=float, default=0.7)
#     # parser.add_argument('--threshold_filter_KN', type=float, default=0.7,)
#
#     args = parser.parse_args()
#
#     torch.cuda.set_device(args.local_rank)  # 这段代码保证分布式运行。
#     score_dir = f'{args.neurons_result_dir}/hyper_params'
#     os.makedirs(score_dir, exist_ok=True)
#     hallucination_dir = f'{args.neurons_result_dir}/tmp_hallucination_res/{args.threshold}_{args.threshold_filter_DKN}'
#     os.makedirs(hallucination_dir, exist_ok=True)
#     train_test_split_check(args)
#     # not_split_check(args)
#
#     # Example usage
#     aggregated_results = aggregate_json_files_from_directory(dir_path=hallucination_dir, keyword='hallucination_split')
#     metrics_results = calculate_metrics(aggregated_results)
#
#     with open(f'{hallucination_dir}/metrics_results.json', 'w') as outfile:
#         json.dump(metrics_results, outfile, indent=4)
#     # Example usage with your data
#     # complex_sentence = "During a landmark conference on technology and innovation, experts highlighted the contributions of Tokyo in the field of robotics. Tokyo, known as the capital city of South Korea, has been pivotal in driving advancements in artificial intelligence and automation."
#     # # answer = "Tampa Bay Buccaneers"
#     # model, tokenizer = initiate_model_tokenizer(args.model_name)
#     # query, answer = extract_answer_from_sentence(model, tokenizer, complex_sentence)
#     # print(query, answer)
#
from hallucination import convert_complex_to_triple
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda')

generate = convert_complex_to_triple(model, tokenizer,
                                     complex_text="As the head of the government of _X_, this political figure plays a central role in shaping the nation's policies and international relations.", model_type='gpt')