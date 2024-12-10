# launch with `python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE pararel_evaluate.py`
import argparse
import glob
import json
import os
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from knowledge_neurons import Dkn
from knowledge_neurons.utils import read_and_adapt_dataset, initiate_model_tokenizer, load_json_files_from_directory, \
    load_jsonl_files_from_directory


def load_existing_results(results_dir, file_suffix, split_rank):
    file_path = results_dir / f'tmp_{file_suffix}/baseline_results_{split_rank}.json'
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def process_data(uuid, args, RESULTS_DIR, baseline_res_first, main_res,
                 baseline_res_src, baseline_res_hierarchical,
                 baseline_res_dbscan, baseline_res_kmeans):
    data = baseline_res_first.get(uuid)
    query_sample = dataset[uuid]
    query = query_sample['query'].replace("_X_.", "").strip()
    answer = query_sample['answer'][0]['name']


    def is_empty_and_increment(neurons, results_dict):
        if not neurons:  # Check if data is empty
            results_dict['empty_count'] = results_dict.get('empty_count', 0) + 1
            return True
        return False

    if uuid not in existing_res_src:
        # Extract the relevant clustering data
        dkn_cluster_2_src = data.get('dkn_cluster_2_src', [])
        if not is_empty_and_increment(dkn_cluster_2_src, baseline_res_src):
            dkn_acc_drops, all_acc_drop = kn.filter_neuron_sets_second_only(prompt=query, ground_truth=answer,
                                                                            dkn_cluster_2=dkn_cluster_2_src)
            baseline_res_src[uuid] = {
                'degenerate_kn_this_uuid': dkn_acc_drops,
                'all_acc_drop': all_acc_drop,
            }
        else:
            baseline_res_src[uuid] = None
        os.makedirs(RESULTS_DIR / 'tmp_res_src', exist_ok=True)
        baseline_res_path = RESULTS_DIR / f'tmp_res_src/baseline_results_{args.split_rank}.json'
        with open(baseline_res_path, "w") as f:
            json.dump(baseline_res_src, f, indent=4)


    if uuid not in existing_res_hierarchical:
        dkn_cluster_2_hierarchical = data.get('dkn_cluster_2_hierarchical', [])
        if not is_empty_and_increment(dkn_cluster_2_hierarchical, baseline_res_hierarchical):
            dkn_acc_drops, all_acc_drop = kn.filter_neuron_sets_second_only(prompt=query, ground_truth=answer,
                                                                            dkn_cluster_2=dkn_cluster_2_hierarchical)
            baseline_res_hierarchical[uuid] = {
                'degenerate_kn_this_uuid': dkn_acc_drops,
                'all_acc_drop': all_acc_drop,
            }
        else:
            baseline_res_hierarchical[uuid] = None
        os.makedirs(RESULTS_DIR / 'tmp_res_hierarchical', exist_ok=True)
        baseline_res_path = RESULTS_DIR / f'tmp_res_hierarchical/baseline_results_{args.split_rank}.json'
        with open(baseline_res_path, "w") as f:
            json.dump(baseline_res_hierarchical, f, indent=4)


    if uuid not in existing_res_dbscan:
        dkn_cluster_2_dbscan = data.get('dkn_cluster_2_dbscan', [])
        if not is_empty_and_increment(dkn_cluster_2_dbscan, baseline_res_dbscan):
            dkn_acc_drops, all_acc_drop = kn.filter_neuron_sets_second_only(prompt=query, ground_truth=answer,
                                                                            dkn_cluster_2=dkn_cluster_2_dbscan)
            baseline_res_dbscan[uuid] = {
                'degenerate_kn_this_uuid': dkn_acc_drops,
                'all_acc_drop': all_acc_drop,
            }
        else:
            baseline_res_hierarchical[uuid] = None
        os.makedirs(RESULTS_DIR / 'tmp_res_dbscan', exist_ok=True)
        baseline_res_path = RESULTS_DIR / f'tmp_res_dbscan/baseline_results_{args.split_rank}.json'
        with open(baseline_res_path, "w") as f:
            json.dump(baseline_res_dbscan, f, indent=4)


    if uuid not in existing_res_kmeans:
        dkn_cluster_2_kmeans = data.get('dkn_cluster_2_kmeans', [])
        if not is_empty_and_increment(dkn_cluster_2_kmeans, baseline_res_kmeans):
            dkn_acc_drops, all_acc_drop = kn.filter_neuron_sets_second_only(prompt=query, ground_truth=answer,
                                                                            dkn_cluster_2=dkn_cluster_2_kmeans)
            baseline_res_kmeans[uuid] = {
                'degenerate_kn_this_uuid': dkn_acc_drops,
                'all_acc_drop': all_acc_drop,
            }
        else:
            baseline_res_hierarchical[uuid] = None
        os.makedirs(RESULTS_DIR / 'tmp_res_kmeans', exist_ok=True)
        baseline_res_path = RESULTS_DIR / f'tmp_res_kmeans/baseline_results_{args.split_rank}.json'
        with open(baseline_res_path, "w") as f:
            json.dump(baseline_res_kmeans, f, indent=4)


    if uuid not in existing_main_res:
        dkn_cluster_main = data.get('dkn_cluster_2', [])
        if not is_empty_and_increment(dkn_cluster_main, main_res):
            dkn_acc_drops, all_acc_drop = kn.filter_neuron_sets_second_only(prompt=query, ground_truth=answer,
                                                                            dkn_cluster_2=dkn_cluster_main)
            main_res[uuid] = {
                'degenerate_kn_this_uuid': dkn_acc_drops,
                'all_acc_drop': all_acc_drop,
            }
        else:
            main_res[uuid] = None
        os.makedirs(RESULTS_DIR / 'tmp_main_res', exist_ok=True)
        baseline_res_path = RESULTS_DIR / f'tmp_main_res/baseline_results_{args.split_rank}.json'
        with open(baseline_res_path, "w") as f:
            json.dump(main_res, f, indent=4)


if __name__ == "__main__":
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
        # default='gpt2',
        default='/home/chenyuheng/KN2/Llama/Llama7bChat',
        # default="meta-llama/Llama-2-70b-chat-hf",
        # default='meta-llama/Llama-2-13b-chat-hf',
    )
    parser.add_argument('--dataset_json', type=str,
                        default='Templama/train.jsonl')
    parser.add_argument('--neurons_result_dir', type=str, default='temporal_res/llama7b_1226/all_res')
    parser.add_argument('--baseline_result_first_dir', type=str, default='temporal_res/llama7b_1226/res_wo_acc')
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--split_rank', type=int, default=0)
    args = parser.parse_args()
    RESULTS_DIR = Path(args.neurons_result_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(args.seed)
    src_dataset = read_and_adapt_dataset(file_path=args.dataset_json)
    baseline_res = load_jsonl_files_from_directory(dir_path=args.baseline_result_first_dir, keyword='results')
    dataset = {uuid: data for uuid, data in src_dataset.items() if uuid in baseline_res}
    ##############################################################################
    # data parallel stuff
    # NUM_REPLICAS = torch.cuda.device_count()
    NUM_REPLICAS = 1
    INDICES = list(range(len(dataset)))
    INDICES = INDICES[args.local_rank: len(dataset): NUM_REPLICAS]
    KEYS = list(dataset.keys())
    torch.cuda.set_device(args.local_rank)
    ##############################################################################

    # initialize results dicts
    existing_main_res = load_existing_results(RESULTS_DIR, 'main_res', split_rank=args.split_rank)
    existing_res_src = load_existing_results(RESULTS_DIR, 'res_src', split_rank=args.split_rank)
    existing_res_hierarchical = load_existing_results(RESULTS_DIR, 'res_hierarchical', split_rank=args.split_rank)
    existing_res_dbscan = load_existing_results(RESULTS_DIR, 'res_dbscan', split_rank=args.split_rank)
    existing_res_kmeans = load_existing_results(RESULTS_DIR, 'res_kmeans', split_rank=args.split_rank)

    # Initialize results dictionaries and merge existing results
    main_res = {}
    main_res.update(existing_main_res)

    baseline_res_src = {}
    baseline_res_src.update(existing_res_src)

    baseline_res_hierarchical = {}
    baseline_res_hierarchical.update(existing_res_hierarchical)

    baseline_res_dbscan = {}
    baseline_res_dbscan.update(existing_res_dbscan)

    baseline_res_kmeans = {}
    baseline_res_kmeans.update(existing_res_kmeans)
    # def initiate_model_tokenizer_new(model_name):
    #     model = AutoModelForCausalLM.from_pretrained(model_name)
    #     model=model.to('cuda')
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     return model, tokenizer
    model, tokenizer = initiate_model_tokenizer(model_name=args.model_name)
    kn = Dkn(model, tokenizer, model_type='llama')

    results_folder = args.neurons_result_dir
    # processed_uuids = set()
    #
    # # Get a list of all result files
    # result_files = glob.glob(os.path.join(f'{results_folder}/tmp_res', '*results*.json'))
    #
    # # For each results file...
    # for results_file in result_files:
    #     # Open the results file
    #     with open(results_file, 'r') as f:
    #         # Load the results as a dict
    #         results = json.load(f)
    #
    #     # Add the UUIDs from this file to the set of processed UUIDs
    #     processed_uuids.update(results.keys())

    for i, idx in enumerate(tqdm(INDICES, position=args.local_rank)):
        uuid = KEYS[idx]
        process_data(uuid=uuid, RESULTS_DIR=RESULTS_DIR, args=args,
                     baseline_res_first=baseline_res,
                     main_res=main_res,
                     baseline_res_src=baseline_res_src,
                     baseline_res_hierarchical=baseline_res_hierarchical,
                     baseline_res_dbscan=baseline_res_dbscan,
                     baseline_res_kmeans=baseline_res_kmeans,
                     )
    main_res_dir = RESULTS_DIR / 'main_res'
    os.makedirs(main_res_dir, exist_ok=True)
    baseline_res_path = f'{main_res_dir}/main_res_{args.split_rank}.json'
    with open(baseline_res_path, "w") as f:
        json.dump(baseline_res_src, f, indent=4)
    main_res_dir = RESULTS_DIR / 'baseline_res_src'
    os.makedirs(main_res_dir, exist_ok=True)
    baseline_res_path = f'{main_res_dir}/baseline_results_{args.split_rank}.json'
    with open(baseline_res_path, "w") as f:
        json.dump(baseline_res_src, f, indent=4)
    main_res_dir = RESULTS_DIR / 'baseline_res_hierarchical'
    os.makedirs(main_res_dir, exist_ok=True)
    baseline_res_path = f'{main_res_dir}/baseline_results_{args.split_rank}.json'
    with open(baseline_res_path, "w") as f:
        json.dump(baseline_res_hierarchical, f, indent=4)
    main_res_dir = RESULTS_DIR / 'baseline_res_dbscan'
    os.makedirs(main_res_dir, exist_ok=True)
    baseline_res_path = f'{main_res_dir}/baseline_results_{args.split_rank}.json'
    with open(baseline_res_path, "w") as f:
        json.dump(baseline_res_dbscan, f, indent=4)
    main_res_dir = RESULTS_DIR / 'baseline_res_kmeans'
    os.makedirs(main_res_dir, exist_ok=True)
    baseline_res_path = f'{main_res_dir}/baseline_results_{args.split_rank}.json'
    with open(baseline_res_path, "w") as f:
        json.dump(baseline_res_kmeans, f, indent=4)
