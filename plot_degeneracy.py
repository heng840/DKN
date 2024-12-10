import argparse
import json
import os

from matplotlib import ticker

from knowledge_neurons.utils import load_json_files_from_directory
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as font_manager

# 字体文件的路径

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# Set font properties
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 33})
def calculate_average(lst):
    return sum(lst) / len(lst) if lst else 0
def count_empty_lists(neurons_data, log_dir):
    empty_count = 0
    for key, value in neurons_data.items():
        # Check if the value is a dictionary and has the key 'dkn_cluster_2'
        if not value.get('dkn_cluster_2'):
            empty_count += 1
    with open(f'{log_dir}/empty_counts.json', 'w') as log_file:
        json.dump({'empty_count': empty_count}, log_file)
    return empty_count



def plot_degeneracy(T=4000):
    os.makedirs(f'{data_dir}/plot_degeneracy', exist_ok=True)
    output_json = f'{data_dir}/plot_degeneracy/acc_drop{T}.json'
    output_png = f'{data_dir}/plot_degeneracy/acc_drop{T}.png'
    data = load_json_files_from_directory(dir_path=data_dir, keyword='temporal_results')
    # Initialize variables for aggregation
    total_drops = []
    size_based_drops = {}
    total_all_neurons_drop = []

    # Aggregate data
    for uuid, results in data.items():
        # Aggregate data for degenerate neuron sets
        degenerate_data = results.get('degenerate_kn_this_uuid', {})
        if degenerate_data == {}:
            continue
        for size, subsets in degenerate_data.items():
            for subset_info in subsets:
                accuracy_drop = -subset_info['accuracy_drop']  # 计算acc写错了，所以加负号。
                if abs(accuracy_drop) < T:
                    total_drops.append(accuracy_drop)
                    size_based_drops.setdefault(size, []).append(accuracy_drop)

        # Aggregate data for full suppression
        # if 1000<abs(results['all_acc_drop']) < T:
        #     print(-results['all_acc_drop'])
        if abs(results['all_acc_drop']) < T:
            total_all_neurons_drop.append(-results['all_acc_drop'])
    # Calculate average drops
    avg_drop = sum(total_drops) / len(total_drops) if total_drops else 0
    avg_drops_by_size = {size: sum(drops) / len(drops) for size, drops in size_based_drops.items()}
    avg_all_neurons_drop = sum(total_all_neurons_drop) / len(
        total_all_neurons_drop) if total_all_neurons_drop else 0

    # Save numerical results to JSON and CSV
    results_to_save = {
        'average_drop': avg_drop,
        'average_drops_by_size': avg_drops_by_size,
        'average_drop_all_neurons': avg_all_neurons_drop
    }
    with open(output_json, 'w') as json_file:
        json.dump(results_to_save, json_file)

    # Plotting
    plt.figure()
    sizes = [int(size) for size in avg_drops_by_size.keys()]

    # Append the size for the 'all neurons' case, which is one more than the largest size
    sizes.append(max(sizes) + 1)

    avg_drops = list(avg_drops_by_size.values())

    # Append the average drop for all neurons
    avg_drops.append(avg_all_neurons_drop)

    plt.plot(sizes, avg_drops, marker='o', label='Avg Drop by Subset Size')
    plt.axhline(y=avg_drop, color='r', linestyle='--', label='Avg Drop Neurons')  # 这是虚线

    plt.title('Average Accuracy Drop by Subset Size')
    plt.xlabel('Subset Size (Including All Neurons)')
    plt.ylabel('Average Accuracy Drop')
    plt.xticks(sizes)  # Ensure all sizes are marked on x-axis
    plt.legend()
    plt.grid(True)
    plt.savefig(output_png)


def aggregate_data_by_size(data):
    size_based_aggregation = {}
    for uuid, results in data.items():
        if len(results['dkn_cluster_2']) > 1:
            size = str(len(results['dkn_cluster_2']))
            if size not in size_based_aggregation:
                size_based_aggregation[size] = []
            size_based_aggregation[size].append(results)

    return size_based_aggregation

def aggregate_data_by_size_baseline(data):
    size_based_aggregation = {}
    for uuid, results in data.items():
        if uuid == 'empty_count' or not results:
            continue
        degenerate_data = results['degenerate_kn_this_uuid']
        if not degenerate_data:
            continue
        max_key = max(int(k) for k in degenerate_data.keys())
        size = max_key + 1
        if size not in size_based_aggregation:
            size_based_aggregation[size] = []
        size_based_aggregation[size].append(results)

    return size_based_aggregation


def record_overall_data_for_baseline(src_data, T):
    fig_dir = 'Acquisition-of-DKN'
    os.makedirs(fig_dir, exist_ok=True)
    total_all_neurons_drop = []
    total_drops = []
    empty = 0
    for key, item in src_data.items():
        if key == 'empty_count':
            continue
        else:
            if not item:
                empty += 1
                continue
            degenerate_data = item.get('degenerate_kn_this_uuid', {})

            if degenerate_data == {}:
                continue

            for subset_size, subsets in degenerate_data.items():
                for subset_info in subsets:
                    accuracy_drop = -subset_info['accuracy_drop']
                    if abs(accuracy_drop) < T:
                        total_drops.append(accuracy_drop)

            if abs(item['all_acc_drop']) < T:
                total_all_neurons_drop.append(-item['all_acc_drop'])
    average_partial_suppression_drop = calculate_average(total_drops)
    average_full_suppression_drop = calculate_average(total_all_neurons_drop)

    # Save the averages to a JSON file
    averages = {
        "average_partial_suppression_drop": average_partial_suppression_drop,
        "average_full_suppression_drop": average_full_suppression_drop,
        'empty_count': empty
    }

    with open(f'{fig_dir}/averages.json', 'a') as outfile:
        json.dump(averages, outfile, indent=4)
def record_overall_data(src_data, T):
    fig_dir = 'Acquisition-of-DKN'
    os.makedirs(fig_dir, exist_ok=True)
    total_all_neurons_drop = []
    total_drops = []
    for key, item in src_data.items():
        if key == 'empty_count':
            continue
        degenerate_data = item.get('degenerate_kn_this_uuid', {})

        if degenerate_data == {}:
            continue

        for subset_size, subsets in degenerate_data.items():
            for subset_info in subsets:
                accuracy_drop = -subset_info['accuracy_drop']
                if abs(accuracy_drop) < T/2:
                    total_drops.append(accuracy_drop)

        if abs(item['all_acc_drop']) < T*2:
            total_all_neurons_drop.append(-item['all_acc_drop'])
    average_partial_suppression_drop = calculate_average(total_drops)
    average_full_suppression_drop = calculate_average(total_all_neurons_drop)

    # Save the averages to a JSON file
    averages = {
        "average_partial_suppression_drop": average_partial_suppression_drop,
        "average_full_suppression_drop": average_full_suppression_drop
    }

    with open(f'{fig_dir}/averages.json', 'a') as outfile:
        json.dump(averages, outfile, indent=4)


def record_data_for_each_size(size_based_aggregation, T, data_dir):
    fig_dir = f'/home/chenyuheng/KN2/kn2/Acquisition-of-DKN/{data_dir}'
    os.makedirs(fig_dir, exist_ok=True)

    size_based_results = {}
    overall_total_drops = []
    overall_total_all_neurons_drop = []

    for size, results_list in size_based_aggregation.items():
        total_drops = []
        total_all_neurons_drop = []

        for results in results_list:
            # Aggregate data for degenerate neuron sets
            degenerate_data = results.get('degenerate_kn_this_uuid', {})
            for subsets in degenerate_data.values():
                for subset_info in subsets:
                    accuracy_drop = -subset_info['accuracy_drop']
                    if abs(accuracy_drop) < T/2:
                        total_drops.append(accuracy_drop)
                        overall_total_drops.append(accuracy_drop)

            all_acc_drop = -results['all_acc_drop']
            if abs(all_acc_drop) < T*2:
                total_all_neurons_drop.append(all_acc_drop)
                overall_total_all_neurons_drop.append(all_acc_drop)

        # Calculate averages for each size
        avg_accuracy_drop = sum(total_drops) / len(total_drops) if total_drops else 0
        avg_all_neurons_drop = sum(total_all_neurons_drop) / len(total_all_neurons_drop) if total_all_neurons_drop else 0

        size_based_results[size] = {
            'average_accuracy_drop': avg_accuracy_drop,
            'average_all_neurons_drop': avg_all_neurons_drop
        }

    # Calculate overall averages
    overall_avg_accuracy_drop = sum(overall_total_drops) / len(overall_total_drops) if overall_total_drops else 0
    overall_avg_all_neurons_drop = sum(overall_total_all_neurons_drop) / len(overall_total_all_neurons_drop) if overall_total_all_neurons_drop else 0

    size_based_results['overall'] = {
        'average_accuracy_drop': overall_avg_accuracy_drop,
        'average_all_neurons_drop': overall_avg_all_neurons_drop
    }

    # Save the results in a JSON file
    with open(f'{fig_dir}/size_based_results_{T}.json', 'w') as f:
        json.dump(size_based_results, f, indent=4)

    return size_based_results

def record_data_for_each_size_baseline(size_based_aggregation,  data_dir, T=10000,):
    fig_dir = f'/home/chenyuheng/KN2/kn2/Acquisition-of-DKN/{data_dir}'
    os.makedirs(fig_dir, exist_ok=True)

    size_based_results = {}
    overall_total_drops = []
    overall_total_all_neurons_drop = []

    for size, results_list in size_based_aggregation.items():
        total_drops = []
        total_all_neurons_drop = []

        for results in results_list:
            # Aggregate data for degenerate neuron sets
            degenerate_data = results.get('degenerate_kn_this_uuid', {})
            for subsets in degenerate_data.values():
                for subset_info in subsets:
                    accuracy_drop = -subset_info['accuracy_drop']
                    if abs(accuracy_drop) < T:
                        total_drops.append(accuracy_drop)
                        overall_total_drops.append(accuracy_drop)

            all_acc_drop = -results['all_acc_drop']
            if abs(all_acc_drop) < T:
                total_all_neurons_drop.append(all_acc_drop)
                overall_total_all_neurons_drop.append(all_acc_drop)

        # Calculate averages for each size
        avg_accuracy_drop = sum(total_drops) / len(total_drops) if total_drops else 0
        avg_all_neurons_drop = sum(total_all_neurons_drop) / len(total_all_neurons_drop) if total_all_neurons_drop else 0

        size_based_results[size] = {
            'average_accuracy_drop': avg_accuracy_drop,
            'average_all_neurons_drop': avg_all_neurons_drop
        }

    # Calculate overall averages
    overall_avg_accuracy_drop = sum(overall_total_drops) / len(overall_total_drops) if overall_total_drops else 0
    overall_avg_all_neurons_drop = sum(overall_total_all_neurons_drop) / len(overall_total_all_neurons_drop) if overall_total_all_neurons_drop else 0

    size_based_results['overall'] = {
        'average_accuracy_drop': overall_avg_accuracy_drop,
        'average_all_neurons_drop': overall_avg_all_neurons_drop
    }

    # Save the results in a JSON file
    with open(f'{fig_dir}/size_based_results_{T}.json', 'w') as f:
        json.dump(size_based_results, f, indent=4)

    return size_based_results
def merge_figures_for_2model(size_based_aggregation1, size_based_aggregation2, T=4000, data_dir='main'):
    fig_dir = f'Acquisition-of-DKN/{data_dir}'
    os.makedirs(fig_dir, exist_ok=True)

    # sizes = [2, 3, 8, 11, 14, 17]
    sizes = [2, 3, 4,5,6,7]
    fig, axs = plt.subplots(1, len(sizes), figsize=(36, 6))

    handles, labels = [], []  # Lists to collect legend handles and labels

    for i, size in enumerate(sizes):
        if str(size) in ['2', '3']:
            axs[i].set_xticks([0, 1, 2, 3])
        # elif str(size) == '8':
        #     axs[i].set_xticks([0, 2, 4, 6, 8])
        # elif str(size) == '11':
        #     axs[i].set_xticks([0, 3, 6, 9, 11])
        # elif str(size) == '14':
        #     axs[i].set_xticks([0, 4, 8, 11, 14])
        else:
            axs[i].set_xticks([0, 2, 4,6,8])

        if size in size_based_aggregation1:
            size_based_drops1 = {}
            total_drops1 = []
            total_all_neurons_drop1 = []
            for results in size_based_aggregation1[size]:
                degenerate_data = results.get('degenerate_kn_this_uuid', {})
                for subset_size, subsets in degenerate_data.items():
                    for subset_info in subsets:
                        accuracy_drop = -subset_info['accuracy_drop']
                        if abs(accuracy_drop) < T/2:
                            total_drops1.append(accuracy_drop)
                            size_based_drops1.setdefault(subset_size, []).append(accuracy_drop)
                if abs(results['all_acc_drop']) < T*2:
                    total_all_neurons_drop1.append(-results['all_acc_drop'])

            avg_drop1 = sum(total_drops1) / len(total_drops1) if total_drops1 else 0
            avg_drops_by_size1 = {size: sum(drops) / len(drops) for size, drops in size_based_drops1.items()}
            avg_all_neurons_drop1 = sum(total_all_neurons_drop1) / len(
                total_all_neurons_drop1) if total_all_neurons_drop1 else 0
            sizes1 = [int(size) for size in avg_drops_by_size1.keys()]
            sizes1.append(max(sizes1) + 1)
            avg_drops1 = list(avg_drops_by_size1.values())
            avg_drops1.append(avg_all_neurons_drop1)
            line1, = axs[i].plot(sizes1, avg_drops1, '-', markersize=15, markerfacecolor='none', color='#1f77b4', linewidth=1, zorder=2, label='GPT-2')
            axs[i].plot(sizes1, avg_drops1, '-', color='#1f77b4', linewidth=15, alpha=0.3, zorder=1)

        if size in size_based_aggregation2:
            size_based_drops2 = {}
            total_drops2 = []
            total_all_neurons_drop2 = []
            for results in size_based_aggregation2[size]:
                degenerate_data = results.get('degenerate_kn_this_uuid', {})
                for subset_size, subsets in degenerate_data.items():
                    for subset_info in subsets:
                        accuracy_drop = -subset_info['accuracy_drop']
                        if abs(accuracy_drop) < T/2:
                            total_drops2.append(accuracy_drop)
                            size_based_drops2.setdefault(subset_size, []).append(accuracy_drop)
                if abs(results['all_acc_drop']) < T*2:
                    total_all_neurons_drop2.append(-results['all_acc_drop'])

            avg_drop2 = sum(total_drops2) / len(total_drops2) if total_drops2 else 0
            avg_drops_by_size2 = {size: sum(drops) / len(drops) for size, drops in size_based_drops2.items()}
            avg_all_neurons_drop2 = sum(total_all_neurons_drop2) / len(
                total_all_neurons_drop2) if total_all_neurons_drop2 else 0
            sizes2 = [int(size) for size in avg_drops_by_size2.keys()]
            sizes2.append(max(sizes2) + 1)
            avg_drops2 = list(avg_drops_by_size2.values())
            avg_drops2.append(avg_all_neurons_drop2)
            line2, = axs[i].plot(sizes2, avg_drops2, '-', markersize=15, color='#ff7f0e', linewidth=1, zorder=2, label='Llama2')
            axs[i].plot(sizes2, avg_drops2, '-', color='#ff7f0e', linewidth=15, alpha=0.3, zorder=1)

        if i == 0:  # Collect once from the first subplot to avoid duplication
            handles.extend([line1, line2])
            labels.extend(['GPT-2', 'Llama2'])

        axs[i].grid(True)

    fig.suptitle('GPT-2 vs Llama2', fontsize=40)
    axs[0].set_ylabel('Prob Drop(%)', fontsize=32)
    fig.text(0.5, 0.05, 'Number of suppressed BDCs', ha='center', va='center', fontsize=35)
    fig.legend(handles, labels, loc='upper right', fontsize=30, bbox_to_anchor=(0.99, 0.98))

    plt.tight_layout()
    output_png = f'{fig_dir}/merged_figures.pdf'
    plt.savefig(output_png, bbox_inches='tight')
    plt.show()
def process_and_plot_data_for_each_size_for_2model(size_based_aggregation1, size_based_aggregation2, T=4000, data_dir='main', ):
    fig_dir = f'Acquisition-of-DKN/{data_dir}'
    os.makedirs(fig_dir, exist_ok=True)

    # Union of keys from both dictionaries
    all_sizes = set(size_based_aggregation1.keys()).union(set(size_based_aggregation2.keys()))

    for size in all_sizes:
        output_png = f'{fig_dir}/acc_drop_size_{size}.pdf'
        # fig, ax = plt.subplots()  # Create a figure and an axes
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and an axes

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        color_blue = (0.7, 0.8, 1.0, 0.3)  # light blue with increased transparency
        color_green = (0.8, 1.0, 0.8, 0.3)  # light green with increased transparency

        if str(size) == '5':
            # ax.set_facecolor(color_blue)  # Set the background color of the plot area
            ax.set_xticks([0, 1, 2, 3, 4, 5])
        elif str(size) == '12':
            # ax.set_facecolor(color_green)
            ax.set_xticks([0, 3, 6, 9, 12])
        elif size >10:

            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        # Process and plot data for size_based_aggregation1
        if size in size_based_aggregation1:
            size_based_drops1 = {}
            total_drops1 = []
            total_all_neurons_drop1 = []
            for results in size_based_aggregation1[size]:
                degenerate_data = results.get('degenerate_kn_this_uuid', {})
                for subset_size, subsets in degenerate_data.items():
                    for subset_info in subsets:
                        accuracy_drop = -subset_info['accuracy_drop']
                        if abs(accuracy_drop) < T/2:
                            total_drops1.append(accuracy_drop)
                            size_based_drops1.setdefault(subset_size, []).append(accuracy_drop)
                if abs(results['all_acc_drop']) < T*2:
                    total_all_neurons_drop1.append(-results['all_acc_drop'])

            avg_drop1 = sum(total_drops1) / len(total_drops1) if total_drops1 else 0
            avg_drops_by_size1= {size: sum(drops) / len(drops) for size, drops in size_based_drops1.items()}
            avg_all_neurons_drop1 = sum(total_all_neurons_drop1) / len(
                total_all_neurons_drop1) if total_all_neurons_drop1 else 0
            sizes1 = [int(size) for size in avg_drops_by_size1.keys()]

            # Append the size for the 'all neurons' case, which is one more than the largest size
            sizes1.append(max(sizes1) + 1)

            avg_drops1 = list(avg_drops_by_size1.values())

            # Append the average drop for all neurons
            avg_drops1.append(avg_all_neurons_drop1)
            # plt.plot(sizes1, avg_drops1, 'o-', markersize=15, markerfacecolor='none', label='GPT-2', color='#1f77b4', linewidth=2)
            plt.plot(sizes1, avg_drops1, '-', markersize=15, markerfacecolor='none', label='GPT-2', color='#1f77b4',
                     linewidth=1, zorder=2)
            plt.plot(sizes1, avg_drops1, '-', color='#1f77b4', linewidth=15, alpha=0.3, zorder=1)

            # plt.axhline(y=avg_drop1, color='r', linestyle='--', label='Avg Drop')

        # Process and plot data for size_based_aggregation2
        if size in size_based_aggregation2:
            size_based_drops2 = {}
            total_drops2 = []
            total_all_neurons_drop2 = []
            for results in size_based_aggregation2[size]:
                degenerate_data = results.get('degenerate_kn_this_uuid', {})
                for subset_size, subsets in degenerate_data.items():
                    for subset_info in subsets:
                        accuracy_drop = -subset_info['accuracy_drop']
                        if abs(accuracy_drop) < T/2:
                            total_drops2.append(accuracy_drop)
                            size_based_drops2.setdefault(subset_size, []).append(accuracy_drop)
                if abs(results['all_acc_drop']) < T*2:
                    total_all_neurons_drop2.append(-results['all_acc_drop'])

            avg_drop2 = sum(total_drops2) / len(total_drops2) if total_drops2 else 0
            avg_drops_by_size2= {size: sum(drops) / len(drops) for size, drops in size_based_drops2.items()}
            avg_all_neurons_drop2 = sum(total_all_neurons_drop2) / len(
                total_all_neurons_drop2) if total_all_neurons_drop2 else 0
            sizes2 = [int(size) for size in avg_drops_by_size2.keys()]

            # Append the size for the 'all neurons' case, which is one more than the largest size
            sizes2.append(max(sizes2) + 1)

            avg_drops2 = list(avg_drops_by_size2.values())

            # Append the average drop for all neurons
            avg_drops2.append(avg_all_neurons_drop2)
            # plt.plot(sizes2, avg_drops2, 'x-', markersize=15, label='Llama2', color='#ff7f0e', linewidth=2)
            plt.plot(sizes2, avg_drops2, '-', markersize=15, label='Llama2', color='#ff7f0e', linewidth=1, zorder=2)
            plt.plot(sizes2, avg_drops2, '-', color='#ff7f0e', linewidth=15, alpha=0.3, zorder=1)
            # plt.axhline(y=avg_drop2, color='r', linestyle='--', label='Avg Drop')

        plt.title(f'GPT-2 vs Llama2')
        plt.ylabel('Prob Drop', fontsize=32, fontweight='bold')
        plt.xlabel('Number of attenuated BDCs', fontsize=35)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_png)
        plt.show()
def process_and_plot_data_for_each_size_for_2model_baseline(size_based_aggregation1, size_based_aggregation2, T=4000, data_dir='main', ):
    fig_dir = f'Acquisition-of-DKN/{data_dir}'
    os.makedirs(fig_dir, exist_ok=True)

    # Union of keys from both dictionaries
    all_sizes = set(size_based_aggregation1.keys()).union(set(size_based_aggregation2.keys()))

    for size in all_sizes:
        output_png = f'{fig_dir}/acc_drop_size_{size}.pdf'
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and an axes

        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        color_blue = (0.7, 0.8, 1.0, 0.3)  # light blue with increased transparency
        color_green = (0.8, 1.0, 0.8, 0.3)  # light green with increased transparency

        if size < 10:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        # Process and plot data for size_based_aggregation1
        if size in size_based_aggregation1:
            size_based_drops1 = {}
            total_drops1 = []
            total_all_neurons_drop1 = []
            for results in size_based_aggregation1[size]:
                degenerate_data = results.get('degenerate_kn_this_uuid', {})
                for subset_size, subsets in degenerate_data.items():
                    for subset_info in subsets:
                        accuracy_drop = -subset_info['accuracy_drop']
                        if abs(accuracy_drop) < T/2:
                            total_drops1.append(accuracy_drop)
                            size_based_drops1.setdefault(subset_size, []).append(accuracy_drop)
                if abs(results['all_acc_drop']) < T*2:
                    total_all_neurons_drop1.append(-results['all_acc_drop'])

            avg_drop1 = sum(total_drops1) / len(total_drops1) if total_drops1 else 0
            avg_drops_by_size1= {size: sum(drops) / len(drops) for size, drops in size_based_drops1.items()}
            avg_all_neurons_drop1 = sum(total_all_neurons_drop1) / len(
                total_all_neurons_drop1) if total_all_neurons_drop1 else 0
            sizes1 = [int(size) for size in avg_drops_by_size1.keys()]

            # Append the size for the 'all neurons' case, which is one more than the largest size
            sizes1.append(max(sizes1) + 1)

            avg_drops1 = list(avg_drops_by_size1.values())

            # Append the average drop for all neurons
            avg_drops1.append(avg_all_neurons_drop1)
            plt.plot(sizes1, avg_drops1, '-', markersize=15, markerfacecolor='none', label='GPT-2', color='#1f77b4',
                     linewidth=1, zorder=2)
            plt.plot(sizes1, avg_drops1, '-', color='#1f77b4', linewidth=15, alpha=0.3, zorder=1)

            # plt.axhline(y=avg_drop1, color='r', linestyle='--', label='Avg Drop')

        # Process and plot data for size_based_aggregation2
        if size in size_based_aggregation2:
            size_based_drops2 = {}
            total_drops2 = []
            total_all_neurons_drop2 = []
            for results in size_based_aggregation2[size]:
                degenerate_data = results.get('degenerate_kn_this_uuid', {})
                for subset_size, subsets in degenerate_data.items():
                    for subset_info in subsets:
                        accuracy_drop = -subset_info['accuracy_drop']
                        if abs(accuracy_drop) < T/2:
                            total_drops2.append(accuracy_drop)
                            size_based_drops2.setdefault(subset_size, []).append(accuracy_drop)
                if abs(results['all_acc_drop']) < T*2:
                    total_all_neurons_drop2.append(-results['all_acc_drop'])

            avg_drop2 = sum(total_drops2) / len(total_drops2) if total_drops2 else 0
            avg_drops_by_size2= {size: sum(drops) / len(drops) for size, drops in size_based_drops2.items()}
            avg_all_neurons_drop2 = sum(total_all_neurons_drop2) / len(
                total_all_neurons_drop2) if total_all_neurons_drop2 else 0
            sizes2 = [int(size) for size in avg_drops_by_size2.keys()]

            # Append the size for the 'all neurons' case, which is one more than the largest size
            sizes2.append(max(sizes2) + 1)

            avg_drops2 = list(avg_drops_by_size2.values())

            # Append the average drop for all neurons
            avg_drops2.append(avg_all_neurons_drop2)
            plt.plot(sizes2, avg_drops2, '-', markersize=15, label='Llama2', color='#ff7f0e', linewidth=1, zorder=2)
            plt.plot(sizes2, avg_drops2, '-', color='#ff7f0e', linewidth=15, alpha=0.3, zorder=1)

        plt.title(f'GPT-2 vs Llama2')
        plt.ylabel('Prob Drop')
        plt.xlabel('Attenuated BDCs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_png)
        # plt.show()

def process_and_plot_data_for_each_size(size_based_aggregation, T=4000, data_dir='main'):
    #TODO 把GPT2和llama放在一张图里，用大圆圈和叉。可以考虑去掉虚线，也可以不去掉虚线。
    fig_dir = f'Acquisition-of-DKN/{data_dir}'
    os.makedirs(fig_dir, exist_ok=True)
    for size, results_list in size_based_aggregation.items():
        size_based_drops = {}
        total_all_neurons_drop = []
        total_drops = []
        # Aggregate data for degenerate neuron sets

        for results in results_list:
            # Aggregate data for degenerate neuron sets
            degenerate_data = results.get('degenerate_kn_this_uuid', {})
            if degenerate_data == {}:
                continue
            for subset_size, subsets in degenerate_data.items():
                for subset_info in subsets:
                    accuracy_drop = -subset_info['accuracy_drop']
                    if abs(accuracy_drop) < T/2:
                        total_drops.append(accuracy_drop)
                        size_based_drops.setdefault(subset_size, []).append(accuracy_drop)

            if abs(results['all_acc_drop']) < T*2:
                total_all_neurons_drop.append(-results['all_acc_drop'])
        # Calculate average drop
        avg_drop = sum(total_drops) / len(total_drops) if total_drops else 0
        avg_drops_by_size = {size: sum(drops) / len(drops) for size, drops in size_based_drops.items()}
        avg_all_neurons_drop = sum(total_all_neurons_drop) / len(
            total_all_neurons_drop) if total_all_neurons_drop else 0
        # print(avg_all_neurons_drop)
        # Plotting
        output_png = f'{fig_dir}/acc_drop_size_{size}.png'
        plt.figure()
        sizes = [int(size) for size in avg_drops_by_size.keys()]

        # Append the size for the 'all neurons' case, which is one more than the largest size
        sizes.append(max(sizes) + 1)

        avg_drops = list(avg_drops_by_size.values())

        # Append the average drop for all neurons
        avg_drops.append(avg_all_neurons_drop)

        plt.plot(sizes, avg_drops, marker='o', label='Prob Drop')
        plt.axhline(y=avg_drop, color='r', linestyle='--', label='Avg Drop')  # 这是虚线

        plt.title('GPT-2')
        # plt.title('LLaMa2-7B')

        # plt.xlabel('Subset Size (Including All Neurons)')
        plt.ylabel('Accuracy Drop')
        plt.xticks(sizes)  # Ensure all sizes are marked on x-axis
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_png)
def process_and_plot_data_for_each_size_baseline(size_based_aggregation, T=4000, data_dir='kmeans'):
    fig_dir = f'Acquisition-of-DKN/{data_dir}'
    os.makedirs(fig_dir, exist_ok=True)
    for size, results_list in size_based_aggregation.items():
        size_based_drops = {}
        total_all_neurons_drop = []
        total_drops = []
        # Aggregate data for degenerate neuron sets

        for results in results_list:
            # Aggregate data for degenerate neuron sets
            degenerate_data = results.get('degenerate_kn_this_uuid', {})
            if degenerate_data == {}:
                continue
            for subset_size, subsets in degenerate_data.items():
                for subset_info in subsets:
                    accuracy_drop = -subset_info['accuracy_drop']
                    if abs(accuracy_drop) < T:
                        total_drops.append(accuracy_drop)
                        size_based_drops.setdefault(subset_size, []).append(accuracy_drop)

            if abs(results['all_acc_drop']) < T:
                total_all_neurons_drop.append(-results['all_acc_drop'])
        # Calculate average drop
        avg_drop = sum(total_drops) / len(total_drops) if total_drops else 0
        avg_drops_by_size = {size: sum(drops) / len(drops) for size, drops in size_based_drops.items()}
        avg_all_neurons_drop = sum(total_all_neurons_drop) / len(
            total_all_neurons_drop) if total_all_neurons_drop else 0
        # print(avg_all_neurons_drop)
        # Plotting
        output_png = f'{fig_dir}/acc_drop_size_{size}.png'
        plt.figure()
        sizes = [int(size) for size in avg_drops_by_size.keys()]

        # Append the size for the 'all neurons' case, which is one more than the largest size
        sizes.append(max(sizes) + 1)

        avg_drops = list(avg_drops_by_size.values())

        # Append the average drop for all neurons
        avg_drops.append(avg_all_neurons_drop)

        plt.plot(sizes, avg_drops, marker='o', label='Suppress subset')
        plt.axhline(y=avg_drop, color='r', linestyle='--', label='Avg proper subset')  # 这是虚线

        plt.title('GPT-2')
        # plt.title('LLaMa2-7B')

        # plt.xlabel('Subset Size (Including All Neurons)')
        plt.ylabel('Accuracy Drop')
        plt.xticks(sizes)  # Ensure all sizes are marked on x-axis
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_png)


def count_total_empty(dir_path, output_file, keyword):
    total_empty_count = 0
    dir = os.path.join(dir_path, keyword)
    for filename in os.listdir(dir):
        if filename.startswith('baseline_results'):
            with open(os.path.join(dir, filename)) as file:
                data = json.load(file)
                total_empty_count += data.get('empty_count', 0)

    with open(output_file, 'a') as file:
        json.dump({f'{keyword}_total_empty_count': total_empty_count}, file)

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Plot Degeneracy Analysis")

    # Define arguments
    parser.add_argument("--data_dir", type=str,
                        default='temporal_res/llama7b_1226/all_res/tmp_main_res'
                        # default='temporal_res/llama7b/2/main_temporal_res'
                        )

    # Parse arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    # Call the function with parsed arguments

    # for i in range(100, 1000, 100):
    #     plot_degeneracy(T=i)
    # plot_degeneracy(T=600)
    # count_total_empty(dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline',
    #                   output_file='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/empty_counts.json',
    #                   keyword='baseline_res_dbscan')
    # count_total_empty(dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline',
    #                   output_file='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/empty_counts.json',
    #                   keyword='baseline_res_hierarchical')
    # count_total_empty(dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline',
    #                   output_file='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/empty_counts.json',
    #                   keyword='baseline_res_kmeans')
    # count_total_empty(dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline',
    #                   output_file='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/empty_counts.json',
    #                   keyword='baseline_res_src')
    # record_overall_data(data, T=1300)


    # data = load_json_files_from_directory(dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/baseline_res_src',
    #                                       keyword='baseline_results',
    #                                       )
    # record_overall_data_for_baseline(data, T=1300)
    # x=9
    # count_empty_lists(data, data_dir)

    '''记录表格'''

    # data_llama = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='temporal_res/llama7b_1226/all_res/tmp_main_res',
    #     keyword='results',
    # )
    # size_based_aggregation_llama = aggregate_data_by_size_baseline(data_llama)
    # record_data_for_each_size(size_based_aggregation_llama, T=900, data_dir='llama/main')
    # print('main')
    #
    # data = load_json_files_from_directory(
    #     # dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/baseline_res_dbscan',
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_res_dbscan',
    #     keyword='results',
    # )
    # size_based_aggregation = aggregate_data_by_size_baseline(data)
    # record_data_for_each_size_baseline(size_based_aggregation, T=900, data_dir='llama/dbscan')
    # print('dbscan')
    #
    # data = load_json_files_from_directory(
    #     # dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/baseline_res_hierarchical',
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_res_hierarchical',
    #     keyword='results',
    #     )
    # size_based_aggregation = aggregate_data_by_size_baseline(data)
    # record_data_for_each_size_baseline(size_based_aggregation, T=900, data_dir='llama/hierarchical')
    # print('baseline_res_hierarchical')
    #
    # data = load_json_files_from_directory(
    #     # dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/baseline_res_kmeans',
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_res_kmeans',
    #     keyword='results',
    #     )
    # size_based_aggregation = aggregate_data_by_size_baseline(data)
    # record_data_for_each_size_baseline(size_based_aggregation, T=900, data_dir='llama/kmeans')
    # print('baseline_res_kmeans')
    #
    # data = load_json_files_from_directory(
    #     # dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/baseline_res_src',
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_res_src',
    #     keyword='results',
    #     )
    # size_based_aggregation = aggregate_data_by_size_baseline(data)
    # record_data_for_each_size_baseline(size_based_aggregation, T=30000, data_dir='llama/src')
    # record_data_for_each_size_baseline(size_based_aggregation, data_dir='src', T=30000)
    # print('baseline_res_src')
    # process_and_plot_data_for_each_size(size_based_aggregation, T=6000)

    '''画图'''
    # data_gpt = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1118_3',
    #     keyword='temporal_results',
    # )
    # size_based_aggregation_gpt = aggregate_data_by_size_baseline(data_gpt)
    # data_llama = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_main_res',
    #     keyword='results',
    # )
    # size_based_aggregation_llama = aggregate_data_by_size_baseline(data_llama)
    #
    # process_and_plot_data_for_each_size_for_2model(size_based_aggregation1=size_based_aggregation_gpt,
    #                                                size_based_aggregation2=size_based_aggregation_llama,
    #                                                T=900, data_dir='All/main')
    # print('main')

    data_gpt = load_json_files_from_directory(
        # dir_path=data_dir,
        dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1118_3',
        keyword='temporal_results',
    )
    size_based_aggregation_gpt = aggregate_data_by_size_baseline(data_gpt)
    data_llama = load_json_files_from_directory(
        # dir_path=data_dir,
        dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_main_res',
        keyword='results',
    )
    size_based_aggregation_llama = aggregate_data_by_size_baseline(data_llama)

    # process_and_plot_data_for_each_size_for_2model(size_based_aggregation1=size_based_aggregation_gpt,
    #                                                size_based_aggregation2=size_based_aggregation_llama,
    #                                                T=900, data_dir='All_2/main_appendix')
    merge_figures_for_2model(size_based_aggregation1=size_based_aggregation_gpt,
                             size_based_aggregation2=size_based_aggregation_llama,
                             T=900, data_dir='All_2/main'
                             )
    print('main')
    #
    #
    # data_gpt = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/baseline_res_dbscan',
    #     keyword='results',
    # )
    # size_based_aggregation_gpt = aggregate_data_by_size_baseline(data_gpt)
    # data_llama = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_res_dbscan',
    #     keyword='results',
    # )
    # size_based_aggregation_llama = aggregate_data_by_size_baseline(data_llama)
    #
    # process_and_plot_data_for_each_size_for_2model_baseline(size_based_aggregation1=size_based_aggregation_gpt,
    #                                                size_based_aggregation2=size_based_aggregation_llama,
    #                                                T=900, data_dir='All_2/dbscan')
    # print('dbscan')
    #
    #
    # data_gpt = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/baseline_res_kmeans',
    #     keyword='results',
    # )
    # size_based_aggregation_gpt = aggregate_data_by_size_baseline(data_gpt)
    # data_llama = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_res_kmeans',
    #     keyword='results',
    # )
    # size_based_aggregation_llama = aggregate_data_by_size_baseline(data_llama)
    #
    # process_and_plot_data_for_each_size_for_2model_baseline(size_based_aggregation1=size_based_aggregation_gpt,
    #                                                size_based_aggregation2=size_based_aggregation_llama,
    #                                                T=900, data_dir='All_2/kmeans')
    # print('kmeans')


    # data_gpt = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/baseline_res_hierarchical',
    #     keyword='results',
    # )
    # size_based_aggregation_gpt = aggregate_data_by_size_baseline(data_gpt)
    # data_llama = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_res_hierarchical',
    #     keyword='results',
    # )
    # size_based_aggregation_llama = aggregate_data_by_size_baseline(data_llama)
    #
    # process_and_plot_data_for_each_size_for_2model_baseline(size_based_aggregation1=size_based_aggregation_gpt,
    #                                                size_based_aggregation2=size_based_aggregation_llama,
    #                                                T=900, data_dir='All_2/hierarchical')
    # print('hierarchical')
    #
    #
    # data_gpt = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/1205/all_baseline/baseline_res_src',
    #     keyword='results',
    # )
    # size_based_aggregation_gpt = aggregate_data_by_size_baseline(data_gpt)
    # data_llama = load_json_files_from_directory(
    #     # dir_path=data_dir,
    #     dir_path='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/all_res/tmp_res_src',
    #     keyword='results',
    # )
    # size_based_aggregation_llama = aggregate_data_by_size_baseline(data_llama)
    #
    # process_and_plot_data_for_each_size_for_2model_baseline(size_based_aggregation1=size_based_aggregation_gpt,
    #                                                size_based_aggregation2=size_based_aggregation_llama,
    #                                                T=900, data_dir='All_2/src')
    # print('src')