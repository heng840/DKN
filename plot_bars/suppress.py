import os

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)



# Set font properties
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# Define data for each subplot
data_gpt2_values_sup = {
    'replace': [47, 32, 11, 2.6],  # [D, N, Rnd, Empty]
    'add': [30, 22, 3, 5],  # [D, N, Rnd, Empty]
    'delete': [39, 33, 8, -2],  # [D, N, Rnd, Empty]
}

data_gpt2_weights_sup = {
    'replace': [38, 22, 2, 3],  # [D, N, Rnd, Empty]
    'add': [19, 18, 5, 4],  # [D, N, Rnd, Empty]
    'delete': [34, 29, 4, 4],  # [D, N, Rnd, Empty]
}

data_llama_values_sup = {
    'replace': [77, 44, 5, 1.76],  # [D, N, Rnd, Empty]
    'add': [12, 16, 3, 0.88],  # [D, N, Rnd, Empty]
    'delete': [65, 50, 4, 3.5],  # [D, N, Rnd, Empty]
}

data_llama_weights_sup = {
    'replace': [45, 22, 2, 3],  # [D, N, Rnd, Empty]
    'add': [18, 11, 3.9, 2],  # [D, N, Rnd, Empty]
    'delete': [77, 33, 3,8, 4.4],  # [D, N, Rnd, Empty]
}

def plot_bars(data_gpt2_values,data_gpt2_weights,data_llama_values,data_llama_weights,
              save_path='/home/chenyuheng/chenyuheng/LIKN/kn2/Disturb_or_Enhance/suppress.pdf'):
    # Calculate average for each dataset
    for data in [data_gpt2_values, data_gpt2_weights, data_llama_values, data_llama_weights]:
        data['average'] = [sum(x) / len(x) for x in zip(data['replace'], data['add'], data['delete'])]

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(30, 6))
    plt.subplots_adjust(wspace=0.2)

    handles, labels = [], []  # Lists to collect legend handles and labels

    subplot_data = [
        (data_gpt2_values, 'GPT-2: Values'),
        (data_gpt2_weights, 'GPT-2: Weights'),
        (data_llama_values, 'LLaMA2: Values'),
        (data_llama_weights, 'LLaMA2: Weights')
    ]

    methods = [r'$\mathcal{D}$', r'$\mathcal{N}$', r'$Rnd$', r'$\emptyset$']
    colors = ['lightpink', 'darkblue', 'darkgreen', 'darkviolet']

    hatches = ['', '///', '///', '///',]  # Applied to all bars for consistency
    for i, (data, title) in enumerate(subplot_data):
        ax = axes[i]
        ax.set_title(title, fontsize=30)
        ax.set_facecolor('floralwhite')
        x = np.arange(len(data['replace']))
        width = 0.2

        for j, method in enumerate(methods):
            values = [data[op][j] for op in ['replace', 'add', 'delete', 'average']]
            if j ==0:
                bars = ax.bar(x + j * width, values, width, color=colors[j], label=method)
            else:
                bars = ax.bar(x + j * width, values, width, color='white',edgecolor=colors[j], label=method,hatch=hatches[j])

            if i == 0:  # Collect handles and labels from the first subplot
                handles.append(bars)
                labels.append(method)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(['Replace', 'Add', 'Delete', 'Average'], fontsize=26)
        ax.tick_params(axis='y', labelsize=25)

        if i == 0:
            ax.set_ylabel(r'$\Delta$ Prob (%)', fontsize=30)

    # Add legend to the figure
    fig.legend(handles, labels, loc="upper left", fontsize=25, bbox_to_anchor=(0.71, 1))

    plt.suptitle('Suppressing DKNs', fontsize=30, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig(save_path)
    plt.show()

plot_bars(data_gpt2_values=data_gpt2_values_sup,data_gpt2_weights=data_gpt2_weights_sup,data_llama_values=data_llama_values_sup,
          data_llama_weights=data_llama_weights_sup, save_path='/home/chenyuheng/chenyuheng/LIKN/kn2/Disturb_or_Enhance/suppress.pdf')


# Define data for each subplot
data_gpt2_values_enh = {
    'replace': [11.8, 1.2, 1.33],  # [D, N, Rnd, Empty]
    'add': [10.2, 3.8, 3.0],  # [D, N, Rnd, Empty]
    'delete': [2.11, 0.97, 0.72,],  # [D, N, Rnd, Empty]
}

data_gpt2_weights_enh = {
    'replace': [14.7, 12.6, 2.1],  # [D, N, Rnd, Empty]
    'add': [12, 9.22, 0.11],  # [D, N, Rnd, Empty]
    'delete': [10.2, 3.55, 1.66],  # [D, N, Rnd, Empty]
}

data_llama_values_enh = {
    'replace': [13.4, 3.55, 2.33],  # [D, N, Rnd, Empty]
    'add': [18.5, 11.22, 0.98],  # [D, N, Rnd, Empty]
    'delete': [19.1, 18.0, 4.3],  # [D, N, Rnd, Empty]
}

data_llama_weights_enh = {
    'replace': [13.1, 12.2, 2.02],  # [D, N, Rnd, ]
    'add': [19.6, 11, 3.9],  # [D, N, Rnd, ]
    'delete': [18.8, 3.8, 4.4],  # [D, N, Rnd, ]
}

def plot_bars_enh(data_gpt2_values, data_gpt2_weights, data_llama_values, data_llama_weights,
              save_path='/home/chenyuheng/chenyuheng/LIKN/kn2/Disturb_or_Enhance/suppress.pdf'):
    # Calculate average for each dataset
    for data in [data_gpt2_values, data_gpt2_weights, data_llama_values, data_llama_weights]:
        data['average'] = [sum(x) / len(x) for x in zip(data['replace'], data['add'], data['delete'])]

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(30, 6))
    plt.subplots_adjust(wspace=0.2)

    handles, labels = [], []  # Lists to collect legend handles and labels

    subplot_data = [
        (data_gpt2_values, 'GPT-2: Values'),
        (data_gpt2_weights, 'GPT-2: Weights'),
        (data_llama_values, 'LLaMA2: Values'),
        (data_llama_weights, 'LLaMA2: Weights')
    ]

    methods = [r'$\mathcal{D}$', r'$\mathcal{N}$', r'$Rnd$']
    colors = ['lightpink', 'darkblue', 'darkgreen']

    hatches = ['', '///', '///']  # Applied to all bars for consistency
    for i, (data, title) in enumerate(subplot_data):
        ax = axes[i]
        ax.set_title(title, fontsize=30)
        ax.set_facecolor('floralwhite')
        x = np.arange(4)
        width = 0.2

        for j, method in enumerate(methods):
            values = [data[op][j] for op in ['replace', 'add', 'delete', 'average']]
            if j == 0:
                bars = ax.bar(x + j * width, values, width, color=colors[j], label=method)
            else:
                bars = ax.bar(x + j * width, values, width, color='white', edgecolor=colors[j], label=method, hatch=hatches[j])

            if i == 0:  # Collect handles and labels from the first subplot
                handles.append(bars)
                labels.append(method)

        ax.set_xticks(x + width)
        ax.set_xticklabels(['Replace', 'Add', 'Delete', 'Average'], fontsize=26)
        ax.tick_params(axis='y', labelsize=25)

        if i == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=30)

    # Add legend to the figure
    fig.legend(handles, labels, loc="upper left", fontsize=25, bbox_to_anchor=(0.71, 1))

    plt.suptitle('Enhancing DKNs', fontsize=30, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()


plot_bars_enh(data_gpt2_values=data_gpt2_values_enh,data_gpt2_weights=data_gpt2_weights_enh,data_llama_values=data_llama_values_enh,
          data_llama_weights=data_llama_weights_enh, save_path='/home/chenyuheng/chenyuheng/LIKN/kn2/Disturb_or_Enhance/enhance.pdf')
