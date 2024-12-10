# launch with `python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE pararel_evaluate.py`
import argparse
import glob
import json
import os
import random
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm


from knowledge_neurons import Dkn
import torch

from knowledge_neurons.utils import read_and_adapt_dataset, initiate_model_tokenizer

if __name__ == "__main__":
    # model_name = "/home/chenyuheng/KN2/Llama/Llama7bChat"
    # # model_name = 'meta-llama/Llama-2-70b-chat-hf'
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # print('load llama successfully')
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
        # default='gpt2',
        default='/home/chenyuheng/KN2/Llama/Llama7bChat',
        # default="meta-llama/Llama-2-70b-chat-hf",
        # default='meta-llama/Llama-2-13b-chat-hf',
    )
    parser.add_argument('--dataset_json', type=str,
                        default='Templama/train.jsonl')
    parser.add_argument('--neurons_result_dir', type=str, default='temporal_res/llama7b/main_res/part1')
    parser.add_argument("--batch_size", type=int, default=10,
                        help='# assert steps % batch_size == 0'
                             '# n_batches = steps // batch_size，'
                             '此外，对应gpt类，因为要re-tokenize new inputs，所以需要内存更多')
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="number of steps to run the integrated gradient calculation for",
    )

    parser.add_argument(
        "--adaptive_threshold",
        type=float,
        default=0.2,
        help="A setting used to determine the score threshold above which coarse neurons are selected "
             "- the paper uses 0.3",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.3,
        help="the threshold for the sharing percentage - we retain neurons that are shared by p% of prompts "
             "(p here is a decimal fraction, i.e between 0 and 1)",
    )
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--distance_quantile', type=float, default=0.2)
    parser.add_argument('--persistence_quantile', type=float, default=0.5)
    parser.add_argument('--fraction_of_edges', type=float, default=0.5)
    parser.add_argument('--threshold_low', type=float, default=0.05, help='没有被百分化，就是小数')
    parser.add_argument('--threshold_high', type=float, default=0.3)
    parser.add_argument('--use_predict_acc_2', type=bool, default=False)
    args = parser.parse_args()
    RESULTS_DIR = Path(args.neurons_result_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(args.seed)
    dataset = read_and_adapt_dataset(file_path=args.dataset_json)
    # fixme for baseline
    # subset_size = int(0.1 * len(dataset))
    # sampled_items = random.sample(list(dataset.items()), subset_size)
    # dataset = dict(sampled_items)
    # fixme
    ##############################################################################
    # data parallel stuff
    NUM_REPLICAS = torch.cuda.device_count()
    INDICES = list(range(len(dataset)))
    # INDICES = INDICES[args.local_rank: len(dataset): NUM_REPLICAS]  # fixme
    KEYS = list(dataset.keys())
    torch.cuda.set_device(args.local_rank)
    ##############################################################################

    # initialize results dicts
    RESULTS = {}
    baseline_results = {}
    model, tokenizer = initiate_model_tokenizer(model_name=args.model_name)
    kn = Dkn(model, tokenizer, model_type='llama')
    # kn = Dkn(model, tokenizer, model_type='gpt')
    # x=0
    # fixme 是否要修改这里:筛选出模型掌握的知识？
    # mastered_entries = {}
    #
    # for uuid, entry in dataset.items():
    #     prompt = entry['query']
    #     ground_truth = entry['answer'][0]['name']
    #     tmp = kn.get_mastered_knowledge(prompt, ground_truth)
    #     if tmp == ground_truth:
    #         mastered_entries[uuid] = entry
    # with open('Templama/mastered_knowledge.jsonl', 'w') as f:
    #     for uuid, entry in mastered_entries.items():
    #         f.write(json.dumps(entry) + '\n')

    # because we may end up getting some neurons multiple times, use lru cache to save time
    @lru_cache(maxsize=None)
    def get_neurons(_uuid):
        PROMPTS = [dataset[_uuid]["query"]]
        GROUND_TRUTH = dataset[_uuid]["answer"][0]['name']
        RELATION_NAME = dataset[_uuid]["relation"]
        neurons = kn.get_refined_neurons(
            prompts=PROMPTS,
            ground_truth=GROUND_TRUTH.lower(),
            p=args.p,
            batch_size=args.batch_size,
            steps=args.steps,
            coarse_adaptive_threshold=args.adaptive_threshold,
            quiet=True,
        )
        return neurons, dataset[_uuid]


    def get_unrelated_fact(KEYS, uuid):
        n_keys = len(KEYS)
        while True:
            random_uuid = KEYS[random.randint(0, n_keys - 1)]
            if random_uuid == uuid:
                continue
            return random_uuid


    def process_data(uuid, KEYS, args, RESULTS, RESULTS_DIR, baseline_res):
        # unrelated_uuid = get_unrelated_fact(KEYS, uuid)  # get a uuid for an unrelated fact / relation

        neurons, data = get_neurons(uuid)
        # get refined neurons
        # unrelated_neurons, unrelated_data = get_neurons(unrelated_uuid)  # get the unrelated neurons
        adjacency_matrix = kn.get_adjacency_matrix(neurons)

        # initialize a results dict
        # results_this_uuid = {
        #     "suppression": {
        #         "related": {
        #             "pct_change": [],
        #             "correct_before": [],
        #             "correct_after": [],
        #         },
        #         "unrelated": {
        #             "pct_change": [],
        #             "correct_before": [],
        #             "correct_after": [],
        #         }},
        #     "enhancement": {
        #         "related": {
        #             "pct_change": [],
        #             "correct_before": [],
        #             "correct_after": [],
        #         },
        #         "unrelated": {
        #             "pct_change": [],
        #             "correct_before": [],
        #             "correct_after": [],
        #         }}
        # }
        # results_this_uuid = {}
        # temp_record = {}
        # for PROMPT in data["sentences"]:
        PROMPT = data['query'].replace("_X_.", "").strip()
        gt =  data['answer'][0]['name']
        dkn_cluster_1 = kn.persistent_homology_clustering(adjacency_matrix=adjacency_matrix, knowledge_neurons=neurons,
                                                          distance_quantile=args.distance_quantile,
                                                          persistence_quantile=args.persistence_quantile,
                                                          fraction_of_edges=args.fraction_of_edges)
        dkn_cluster_2_src = kn.get_dkn_src(prompt=PROMPT, ground_truth=gt, all_neurons=neurons, threshold_low=args.threshold_low,
                                           threshold_high=args.threshold_high)
        dkn_1_hierarchical = kn.hierarchical_clustering(adjacency_matrix=adjacency_matrix, percentile_threshold=0.7,
                                                        knowledge_neurons=neurons)
        dkn_1_dbscan = kn.dbscan_clustering(adjacency_matrix=adjacency_matrix, percentile_eps=0.5, knowledge_neurons=neurons)
        dkn_1_kmeans = kn.kmeans_clustering(adjacency_matrix=adjacency_matrix, knowledge_neurons=neurons, variance_threshold=0.95)
        # dkn_cluster_2, dkn_acc_drops, all_acc_drop = kn.filter_neuron_sets(prompt=PROMPT, ground_truth=gt, dkn_cluster_1=dkn_cluster_1,
        #                                       threshold_low=args.threshold_low, use_predict_acc_2=args.use_predict_acc_2)
        # # # # 因为只需要聚类，所有第一步就足够。
        dkn_cluster_2 = kn.filter_neuron_sets_only_first(prompt=PROMPT, ground_truth=gt, dkn_cluster_1=dkn_cluster_1,
                                                         threshold_low=args.threshold_low, use_predict_acc_2=args.use_predict_acc_2)
        dkn_cluster_2_hierarchical = kn.filter_neuron_sets_only_first(prompt=PROMPT, ground_truth=gt, dkn_cluster_1=dkn_1_hierarchical,
                                              threshold_low=args.threshold_low, use_predict_acc_2=args.use_predict_acc_2)
        dkn_cluster_2_dbscan = kn.filter_neuron_sets_only_first(prompt=PROMPT, ground_truth=gt, dkn_cluster_1=dkn_1_dbscan,
                                              threshold_low=args.threshold_low, use_predict_acc_2=args.use_predict_acc_2)
        dkn_cluster_2_kmeans = kn.filter_neuron_sets_only_first(prompt=PROMPT, ground_truth=gt, dkn_cluster_1=dkn_1_kmeans,
                                              threshold_low=args.threshold_low, use_predict_acc_2=args.use_predict_acc_2)
        # (dkn_cluster_2_use_2,
        #  dkn_acc_drops_use_2,
        #  all_acc_drop_use_2) \
        #     = kn.filter_neuron_sets(prompt=PROMPT, ground_truth=gt, dkn_cluster_1=dkn_cluster_1,
        #                                       threshold_low=args.threshold_low, use_predict_acc_2=True)

        # really should be using a different for the suppression, but the authors didn't make their bing dataset available
        # suppression_results, _ = kn.suppress_knowledge(PROMPT, gt, neurons, quiet=True)
        # enhancement_results, _ = kn.enhance_knowledge(PROMPT, gt, neurons, quiet=True)

        # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
        # suppression_prob_diff = (suppression_results["after"]["gt_prob"] - suppression_results["before"][
        #     "gt_prob"]) / suppression_results["before"]["gt_prob"]
        # results_this_uuid["suppression"]["related"]["pct_change"].append(suppression_prob_diff)
        #
        # enhancement_prob_diff = (enhancement_results["after"]["gt_prob"] - enhancement_results["before"][
        #     "gt_prob"]) / enhancement_results["before"]["gt_prob"]
        # results_this_uuid["enhancement"]["related"]["pct_change"].append(enhancement_prob_diff)
        #
        # # check whether the answer was correct before/after suppression
        # results_this_uuid["suppression"]["related"]["correct_before"].append(
        #     suppression_results["before"]["argmax_completion"] == gt
        # )
        # results_this_uuid["suppression"]["related"]["correct_after"].append(
        #     suppression_results["after"]["argmax_completion"] == gt
        # )
        #
        # results_this_uuid["enhancement"]["related"]["correct_before"].append(
        #     enhancement_results["before"]["argmax_completion"] == gt
        # )
        # results_this_uuid["enhancement"]["related"]["correct_after"].append(
        #     enhancement_results["after"]["argmax_completion"] == gt
        # )
        # # for PROMPT in unrelated_data["sentences"]:
        #     # do the same but with unrelated facts
        #
        # un_PROMPT = unrelated_data['query'].replace("_X_.", "").strip()
        # gt = unrelated_data['answer'][0]['name'].lower()
        # unrelated_suppression_results, _ = kn.suppress_knowledge(
        #     PROMPT, gt, neurons, quiet=True
        # )
        # unrelated_enhancement_results, _ = kn.enhance_knowledge(
        #     PROMPT, gt, neurons, quiet=True
        # )
        #
        # # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
        # suppression_prob_diff = (unrelated_suppression_results["after"]["gt_prob"] -
        #                          unrelated_suppression_results["before"]["gt_prob"]) / \
        #                         unrelated_suppression_results["before"]["gt_prob"]
        # results_this_uuid["suppression"]["unrelated"]["pct_change"].append(suppression_prob_diff)
        # enhancement_prob_diff = (unrelated_enhancement_results["after"]["gt_prob"] -
        #                          unrelated_enhancement_results["before"]["gt_prob"]) / \
        #                         unrelated_enhancement_results["before"]["gt_prob"]
        # results_this_uuid["enhancement"]["unrelated"]["pct_change"].append(enhancement_prob_diff)
        #
        # # check whether the answer was correct before/after suppression
        # results_this_uuid["suppression"]["unrelated"]["correct_before"].append(
        #     unrelated_suppression_results["before"]["argmax_completion"] == gt
        # )
        # results_this_uuid["suppression"]["unrelated"]["correct_after"].append(
        #     unrelated_suppression_results["after"]["argmax_completion"] == gt
        # )
        #
        # results_this_uuid["enhancement"]["unrelated"]["correct_before"].append(
        #     unrelated_enhancement_results["before"]["argmax_completion"] == gt
        # )
        # results_this_uuid["enhancement"]["unrelated"]["correct_after"].append(
        #     unrelated_enhancement_results["after"]["argmax_completion"] == gt
        # )

        # results_this_uuid["n_refined_neurons"] = len(neurons)
        # results_this_uuid["n_unrelated_neurons"] = len(unrelated_neurons)
        # results_this_uuid["relation_name"] = data["relation"]
        # results_this_uuid['neurons'] = neurons
        # # if args.with_synergistic:
        # RESULTS[uuid] = {
        #     # **results_this_uuid,
        #     'dkn_cluster_1': dkn_cluster_1,  #
        #     'dkn_cluster_2': dkn_cluster_2,  #
        #     # 'degenerate_kn_this_uuid': dkn_acc_drops,  #使用这个的长度可以把size和accuracy对应起来
        #     # 'all_acc_drop': all_acc_drop,
        # }
        # baseline_res[uuid] = {
        #     'dkn_cluster_2_hierarchical': dkn_cluster_2_hierarchical,
        #     'dkn_cluster_2_dbscan': dkn_cluster_2_dbscan,
        #     'dkn_cluster_2_kmeans': dkn_cluster_2_kmeans,
        #     'dkn_cluster_2_src': dkn_cluster_2_src,
        # }
        result_this_uuid = {uuid: {
            'kn': neurons,
            'dkn_cluster_2': dkn_cluster_2,
            'dkn_cluster_2_hierarchical': dkn_cluster_2_hierarchical,
            'dkn_cluster_2_dbscan': dkn_cluster_2_dbscan,
            'dkn_cluster_2_kmeans': dkn_cluster_2_kmeans,
            'dkn_cluster_2_src': dkn_cluster_2_src,
        }}
        os.makedirs(RESULTS_DIR / 'jsonl_res', exist_ok=True)
        results_json_path = RESULTS_DIR / f"jsonl_res/temporal_results_{args.local_rank}.jsonl"
        with open(results_json_path, "a") as f:
            f.write(json.dumps(result_this_uuid) + '\n')


    # Define the specific uuid you want to process
    # specific_uuid = "Q241261_P54_2015"
    # if specific_uuid in KEYS:
    #     # Process the specific UUID
    #     process_data(specific_uuid, KEYS, args, RESULTS, RESULTS_DIR, baseline_results)

    results_folder = args.neurons_result_dir
    processed_uuids = set()

    # Get a list of all result files
    result_files = glob.glob(os.path.join(f'{results_folder}/jsonl_res', '*results*.jsonl'))
    # for jsonl
    for results_file in result_files:
        with open(results_file, 'r') as f:
            for line in f:
                result = json.loads(line)
                # Assuming each line is a JSON object with a UUID as its key
                uuid = next(iter(result))
                processed_uuids.add(uuid)

    # for json
    # for results_file in result_files:
    #     with open(results_file, 'r') as f:
    #         results = json.load(f)
    #     processed_uuids.update(results.keys())

    for i, idx in enumerate(tqdm(INDICES, position=args.local_rank)):
        uuid = KEYS[idx]
        if uuid not in processed_uuids:
        #     try:
            process_data(uuid, KEYS, args, RESULTS, RESULTS_DIR, baseline_results)

            # except RuntimeError as e:
            #     if "CUDA out of memory" in str(e):
            #         print(f"\n UUID: {uuid}")
            #         with open(f'{RESULTS_DIR}/skipped_uuids.txt', 'a') as log_file:
            #             log_file.write(uuid + '\n')
            #         torch.cuda.empty_cache()
            #         continue
            #     else:
            #         raise
            # # todo 之前的数据应该被保存而不是被覆盖
    # main_res_dir = RESULTS_DIR / 'main_temporal_res'
    # os.makedirs(main_res_dir, exist_ok=True)
    # results_json_path = f"{main_res_dir}/temporal_results_{args.local_rank}.json"
    # with open(results_json_path, "w") as f:
    #     json.dump(RESULTS, f, indent=4)
    # baseline_res_path = f'{main_res_dir}/baseline_results_{args.local_rank}.json'
    # with open(baseline_res_path, "w") as f:
    #     json.dump(baseline_results, f, indent=4)
    # results_json_path = RESULTS_DIR / f"temporal_results_{args.local_rank}.json"
    # with open(results_json_path, "w") as f:
    #     json.dump(RESULTS, f, indent=4)
