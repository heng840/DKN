import argparse
import json
import os
import random

import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as font_manager

# 字体文件的路径
font_path = '/home/chenyuheng/KN2/kn2/Times New Roman.ttf'  # 请确保路径正确

# 动态添加字体路径
font_manager.fontManager.addfont(font_path)

from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from knowledge_neurons import Dkn
from knowledge_neurons.utils import read_and_adapt_dataset, read_neuron_data, calculate_accuracy_drop, \
    initiate_model_tokenizer, get_model_output, load_jsonl_files_from_directory, load_json_files_from_directory

random.seed(42)

# 获得average_results以后可以画图。
# plt.rcParams['font.family'] = 'Times New Roman'
font_path = '/path/to/TimesNewRoman.ttf'  # 替换为您的 .ttf 文件的实际路径
prop = FontProperties(fname=font_path)

plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'font.family': 'Times New Roman'})
# Function for random query interference
def interfere_query(query, method):
    tokens = query.split()
    if method == "replace":
        # Replace a random token (excluding the first and last token)
        if len(tokens) > 2:
            replace_index = random.randint(1, len(tokens) - 2)
            tokens[replace_index] = "[REPLACED]"
    elif method == "add":
        # Add a token at a random position
        add_index = random.randint(1, len(tokens))
        tokens.insert(add_index, "[ADDED]")
    elif method == "delete":
        # Delete a random token (excluding the first and last token)
        if len(tokens) > 2:
            del_index = random.randint(1, len(tokens) - 2)
            del tokens[del_index]
    return ' '.join(tokens)


def run_experiment_for_enhance(args):
    data = read_and_adapt_dataset(file_path=templama_file)
    # for GPT-2
    # if args.model_name == 'gpt2':
    #     model, tokenizer = initiate_model_tokenizer(model_name=args.model_name)
    # else:
    #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #     model = GPT2LMHeadModel.from_pretrained(args.model_name)
    #     model = model.to('cuda')
    model, tokenizer = initiate_model_tokenizer(model_name=args.model_name)
    if args.model_name == 'gpt2':
        kn = Dkn(model, tokenizer, model_type='gpt')
        neurons_result = load_json_files_from_directory(args.neurons_result_dir, 'temporal_results')
    else:
        kn = Dkn(model, tokenizer, model_type='llama')
        neurons_result = load_jsonl_files_from_directory(args.neurons_result_dir, 'temporal_results')
    # with open(save_path, 'w') as f:
    #     pass
    for uuid, item in tqdm(data.items()):
        query = item['query'].replace('_X_.', '').strip()
        disturbed_query = interfere_query(query, method=args.method)
        ground_truth = item['answer'][0]['name']
        normal_query_original_accuracy = kn.get_predict_acc(query, ground_truth, [])
        disturbed_query_original_accuracy = kn.get_predict_acc(disturbed_query, ground_truth, [])
        if uuid in neurons_result:
            dkn_cluster_2 = neurons_result[uuid]['dkn_cluster_2']
            if not dkn_cluster_2:
                with open(save_path, 'a') as f:
                    f.write(json.dumps({
                        'uuid': uuid,
                        'results': {},
                    }) + '\n')
                continue
            dkn_2d = [neuron for subset in dkn_cluster_2 for neuron in subset]
            disturbed_query_accuracy_after_mode = kn.get_predict_acc(disturbed_query, ground_truth, dkn_2d, mode=args.mode)
            normal_query_accuracy_after_mode = kn.get_predict_acc(query, ground_truth, dkn_2d, mode=args.mode)

            disturbed_query_drop_neurons = calculate_accuracy_drop(disturbed_query_original_accuracy,
                                                           disturbed_query_accuracy_after_mode)
            normal_query_drop_neurons = calculate_accuracy_drop(normal_query_original_accuracy,
                                                        normal_query_accuracy_after_mode)
            normal2disturb_drop_neurons = calculate_accuracy_drop(disturbed_query_accuracy_after_mode,
                                                          normal_query_accuracy_after_mode)
            kn.change_weights(dkn_2d, value=args.change_weight_value)

            normal_query_acc_after_destruction = kn.get_acc_without_mode(query, ground_truth)
            disturbed_query_acc_after_destruction = kn.get_acc_without_mode(disturbed_query, ground_truth)

            # Restore the model weights to their original state
            kn.restore_weights()

            # Calculate accuracy drops
            normal_query_drop_weights = calculate_accuracy_drop(normal_query_original_accuracy, normal_query_acc_after_destruction)
            disturbed_query_drop_weights = calculate_accuracy_drop(disturbed_query_original_accuracy, disturbed_query_acc_after_destruction)
            normal2disturb_drop_weights = calculate_accuracy_drop(normal_query_acc_after_destruction,
                                                              disturbed_query_acc_after_destruction)
        else:
            raise NotImplementedError
        # fixme 如果改变acc的函数，应该重新运行，因为要和main得到的结果对应。
        results = {
            "normal_query_original_accuracy": normal_query_original_accuracy,
            "disturbed_query_original_accuracy": disturbed_query_original_accuracy,
            "normal_query_drop_suppress_neurons": normal_query_drop_neurons,  # 抑制神经元
            'normal2disturb_drop_suppress_neurons': normal2disturb_drop_neurons,
            "disturbed_query_drop_suppress_neurons": disturbed_query_drop_neurons,
            'disturbed_query_accuracy_after_mode':disturbed_query_accuracy_after_mode,
            'normal_query_accuracy_after_mode':normal_query_accuracy_after_mode,
            "normal_query_drop_zero_weights": normal_query_drop_weights,  # 破坏结构
            "disturbed_query_drop_zero_weights": disturbed_query_drop_weights,
            'normal2disturb_drop_zero_weights': normal2disturb_drop_weights,
            'normal_query_acc_after_destruction':normal_query_acc_after_destruction,
            'disturbed_query_acc_after_destruction':disturbed_query_acc_after_destruction
        }

        with open(save_path, 'a') as f:
            f.write(json.dumps({
                'uuid': uuid,
                'results': results,
            }) + '\n')
        # save results

def average_results(file_path):
    # Initialize a dictionary to store the total sums and counts for each key
    sums = {
        # "normal_query_original_accuracy": 0,
        # "disturbed_query_original_accuracy": 0,  # 希望：和normal接近
        # "normal_query_drop_suppress_neurons": 0,  # 希望下降不多, 1代表了acc的方法1
        # 'normal2disturb_drop_suppress_neurons':0,
        # "disturbed_query_drop_suppress_neurons": 0,  # 下降多
        'disturbed_query_accuracy_after_mode':0,
        'normal_query_accuracy_after_mode':0,
        # "normal_query_drop_zero_weights": 0,
        # "disturbed_query_drop_zero_weights": 0,
        # 'normal2disturb_drop_zero_weights':0,
        'normal_query_acc_after_destruction':0,
        'disturbed_query_acc_after_destruction':0,
    }
    counts = {key: 0 for key in sums.keys()}

    # Read the file and accumulate the sums and counts
    # fixme None值表示：原始准确率和编辑后（抑制神经元或者破坏结构）准确率很接近。这代表了模型这对模型预测没有影响。跳过/置0？
    with open(file_path, 'r') as f:
        max_values = {key: float('-inf') for key in sums.keys()}
        for line in f:
            sample = json.loads(line)
            results = sample['results']
            for key in sums.keys():
                if not results:
                    continue
                if not results[key]:
                    counts[key] += 1
                else:

                    if results[key] > max_values[key]:
                        max_values[key] = results[key]
                    if key == 'disturbed_query_accuracy_after_mode' or key == 'normal_query_accuracy_after_mode':
                        if 'gpt2' in args.model_name:
                            if abs(results[key]) < 1e-05:
                                sums[key] += results[key]
                                counts[key] += 1
                        else:
                            sums[key] += results[key]
                            counts[key] += 1
                    elif key == 'disturbed_query_acc_after_destruction' or key == 'normal_query_acc_after_destruction':
                        if 'gpt2' in args.model_name:
                            if abs(results[key]) < 1e-6:
                                sums[key] += results[key]
                                counts[key] += 1
                        else:
                            if abs(results[key]) < 2e-10:
                                sums[key] += results[key]
                                counts[key] += 1
                    else:
                        sums[key] += results[key]
                        counts[key] += 1

    # Calculate the averages
    averages = {key: sums[key] / counts[key] for key in sums.keys()}
    # print(averages)
    return averages

#TODO 加入背景色。两个情况，分别用两种背景色。
def plot_and_save_graph_from_jsonl(jsonl_file_path, output_file_path):
    # Initialize data structure
    data = {
        'Attenuate Values': {'Add': {}, 'Replace': {}, 'Delete': {}, 'Average': {}},
        'Attenuate Weights': {'Add': {}, 'Replace': {}, 'Delete': {}, 'Average': {}}
    }
    # data = {
    #     'Attenuate Values': {'Add': {}, 'Rpl': {}, 'Del': {}, 'Ovr': {}},
    #     'Attenuate Weights': {'Add': {}, 'Rpl': {}, 'Del': {}, 'Ovr': {}}
    # }

    # Read JSONL file and extract data
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            for situation, situation_data in json_data.items():
                data['Attenuate Values'][situation] = {
                    'Normal-Q': situation_data['normal_query_accuracy_after_mode'],
                    'Perturb-Q': situation_data['disturbed_query_accuracy_after_mode'],
                    'Reduction %': 100 * (situation_data['disturbed_query_accuracy_after_mode'] - situation_data['normal_query_accuracy_after_mode']) / situation_data['normal_query_accuracy_after_mode']
                }
                data['Attenuate Weights'][situation] = {
                    'Normal-Q': situation_data['normal_query_acc_after_destruction'],
                    'Perturb-Q': situation_data['disturbed_query_acc_after_destruction'],
                    'Reduction %': 100 * (situation_data['disturbed_query_acc_after_destruction'] - situation_data['normal_query_acc_after_destruction']) / situation_data['normal_query_acc_after_destruction']
                }


    # situations = ['Add', 'Rpl', 'Del', 'Ovr']
    situations = ['Add', 'Replace', 'Delete', 'Average']
    data_types = ['Attenuate Values', 'Attenuate Weights']
    bar_width = 0.35 / 2
    group_spacing = 0.1  # Space between each group of situations
    data_types_x = np.arange(len(data_types)) * 2  # Adjusted for overall spacing between data types

    fig, ax = plt.subplots(figsize=(12, 6))
    color_normal = '#00bcd4'  # Soft green
    color_perturb = '#ff69b4'  # Gentle pink
    alpha_normal = 0.6
    alpha_perturb = 0.6
    # Draw rectangles for background colors
    left_bg_color = '#e6faff'  # Light blue for 'Attenuate Values'
    right_bg_color = '#fff5e6'  # Light orange for 'Attenuate Weights'
    ax.add_patch(plt.Rectangle((0, 0), 0.5, 1, transform=ax.transAxes, color=left_bg_color, zorder=-1))
    ax.add_patch(plt.Rectangle((0.5, 0), 0.5, 1, transform=ax.transAxes, color=right_bg_color, zorder=-1))
    # ax.add_patch(plt.Rectangle((0, -0.2), 0.5, 0.2, transform=ax.transAxes, color=left_bg_color, zorder=-1))
    # ax.add_patch(plt.Rectangle((0.5, -0.2), 0.5, 0.2, transform=ax.transAxes, color=right_bg_color, zorder=-1))
    ax2 = ax.twinx()

    for i, data_type in enumerate(data_types):
        data_type_position = data_types_x[i]
        for j, situation in enumerate(situations):
            normal_q = data[data_type][situation]['Normal-Q']
            perturb_q = data[data_type][situation]['Perturb-Q']
            reduction_percentage = data[data_type][situation]['Reduction %']

            bar_position_normal = data_type_position + (j * (2 * bar_width + group_spacing))
            bar_position_perturb = bar_position_normal + bar_width

            axis = ax if i == 0 else ax2
            axis.bar(bar_position_normal, normal_q, bar_width, color=color_normal, alpha=alpha_normal)
            axis.bar(bar_position_perturb, perturb_q, bar_width, color=color_perturb, alpha=alpha_perturb)
            if situation == 'Add':
                top_of_bar = perturb_q
            else:
                top_of_bar = perturb_q
            text_x = bar_position_perturb
            text_y = top_of_bar
            if reduction_percentage <= 0:
                reduction_percentage = -reduction_percentage
                axis.annotate(f'{reduction_percentage:.0f}%↓',
                              xy=(text_x, text_y),
                              textcoords="offset points",
                              ha='center', va='bottom')
            else:
                axis.annotate(f'{reduction_percentage:.0f}↑',
                              xy=(text_x, text_y),
                              textcoords="offset points",
                              ha='center', va='bottom')


    # Adding legends for Normal-Q and Perturb-Q
    custom_xticks = []
    custom_xticklabels = []
    data_type_centers = []

    for i, data_type in enumerate(data_types):
        data_type_position = data_types_x[i]
        situation_centers = []

        for j, situation in enumerate(situations):
            center_position = data_type_position + (j * (2 * bar_width + group_spacing)) + bar_width / 2
            custom_xticks.append(center_position)
            custom_xticklabels.append(situation)
            situation_centers.append(center_position)

        # Calculate the average of the center positions for the data type label
        data_type_centers.append(sum(situation_centers) / len(situations))

    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xticklabels, rotation=3, fontsize=25)

    for i, data_type_center in enumerate(data_type_centers):
        bg_color = left_bg_color if i == 0 else right_bg_color
        ax.text(data_type_center, -0.18, data_types[i], ha='center', va='top',
                transform=ax.get_xaxis_transform(),
                bbox=dict(facecolor=bg_color, edgecolor='none', boxstyle='round,pad=0.3'))
        # rect_start = 0 if i == 0 else 0.5
        # ax.text(data_type_center, -0.18, data_types[i], ha='center', va='top',
        #         transform=ax.get_xaxis_transform())

    if 'gpt2' in args.model_name:
        ax.set_title('GPT-2', bbox=dict(facecolor="#ffcccc", edgecolor='none', boxstyle='round,pad=0.1'), zorder=100)
    else:
        ax.set_title('Llama2', bbox=dict(facecolor="#cce6ff", edgecolor='none', boxstyle='round,pad=0.1'), zorder=100)

    max_value_Values = 0
    max_value_weights = 0
    for key_i, data_type in data.items():
        for situation in data_type.values():
            if key_i == 'Attenuate Values':
                max_value_Values = max(max_value_Values, situation['Normal-Q'], situation['Perturb-Q'])
            else:
                max_value_weights = max(max_value_weights, situation['Normal-Q'], situation['Perturb-Q'])

    # Add y-axis labels
    ax.set_ylabel('Prob',
                  # fontsize=14
                  )  # Caption for the primary y-axis
    # ax.set_ylabel('Prob. for Attenuate Values',
    #               # fontsize=14
    #               )  # Caption for the primary y-axis
    # ax2.set_ylabel('Prob. for Attenuate Weights',
    #                # fontsize=14
    #                )  # Caption for the secondary y-axis
    # Set y-axis limits to a bit more than the max value
    ax.set_ylim(0, 1.6 * max_value_Values)  # 10% more than the max value
    ax2.set_ylim(0, 1.6 * max_value_weights)  # Adjust the secondary axis similarly

    ax.grid(True, linestyle='-', linewidth=1, color='gray', alpha=0.5)

    ax.set_axisbelow(True)  # Ensure grid lines are below the bars
    normal_patch = mpatches.Patch(color=color_normal, label=r'$Q$')
    perturb_patch = mpatches.Patch(color=color_perturb, label=r'$Q^*$')
    legend = ax.legend(handles=[normal_patch, perturb_patch], loc=(0, 0.69), fontsize=22)
    plt.tight_layout()

    # Save the plot to the specified file
    plt.savefig(output_file_path)
    plt.show()

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Run Degeneracy Experiment")
    parser.add_argument("--method", type=str, choices=["replace", "add", "delete"], default='add')

    parser.add_argument(
        "--model_name",
        type=str,
        # default="EleutherAI/gpt-j-6b",
        default='gpt2',
        # default='/home/chenyuheng/KN2/kn2/saved_models/epoch100/model_direct',
        # default='/home/chenyuheng/KN2/Llama/Llama7bChat',
        # default="meta-llama/Llama-2-70b-chat-hf",
        # default='meta-llama/Llama-2-13b-chat-hf',
    )
    # parser.add_argument('--input_file_dir', type=str,
    #                     default='Templama')
    parser.add_argument('--neurons_result_dir', type=str,
                        default='/home/chenyuheng/KN2/kn2/temporal_res/1118_3'
                        # default='temporal_res/llama7b_1226/res_wo_acc'
                        )
    parser.add_argument('--mode', type=str, default='suppress')
    parser.add_argument('--change_weight_value', type=float, default=0.0)
    parser.add_argument('--save_dir', type=str,
                        default='Disturb_or_Enhance/gpt2/suppress_res'
                        # default='Disturb_or_Enhance/llama/suppress_res'
                        )
    parser.add_argument('--run_experiment', action='store_true', default=False)
    # Parse arguments
    args = parser.parse_args()

    method = args.method
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{method}.jsonl'
    # input_file_dir = args.input_file_dir
    if args.mode == 'enhance':
        templama_file = f'Templama/enhance_value_or_weight/filtered_train_enhance_{method}.jsonl'
    elif args.mode == 'suppress':
        templama_file = f'Templama/train.jsonl'
    else:
        raise NotImplementedError
    # Run the experiment
    if args.run_experiment:
        # run_experiment(args)
        run_experiment_for_enhance(args)
    average_add = average_results(f'{save_dir}/add.jsonl')
    average_delete = average_results(f'{save_dir}/delete.jsonl')
    average_replace = average_results(f'{save_dir}/replace.jsonl')
    #
    overall_average = {}
    for key in average_add.keys():
        overall_average[key] = (average_add[key] + average_delete[key] + average_replace[key]) / 3
    # Writing the results to a file
    with open(f'{save_dir}/average_res.jsonl', 'w') as f:
        # f.write(json.dumps({'Add': average_add}) + "\n")
        # f.write(json.dumps({'Del': average_delete}) + "\n")
        # f.write(json.dumps({'Rpl': average_replace}) + "\n")
        # f.write(json.dumps({'Ovr': overall_average}) + "\n")
        f.write(json.dumps({'Add': average_add}) + "\n")
        f.write(json.dumps({'Delete': average_delete}) + "\n")
        f.write(json.dumps({'Replace': average_replace}) + "\n")
        f.write(json.dumps({'Average': overall_average}) + "\n")
    #
    plot_and_save_graph_from_jsonl(jsonl_file_path=f'{save_dir}/average_res.jsonl', output_file_path=f'{save_dir}/fig-dkn-suppress_2.png')
