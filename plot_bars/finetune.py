import os

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)



# Set font properties
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# Define data for each subplot
data_gpt2 = {
    r'$Q_{new}$': [88, 83, 43, 93],  # [D, N, Rnd, All]
    r'$Q_{old}$': [83, 86, 71, 49],  # [D, N, Rnd, All]
    r'$Q_{au}$': [86, 82, 38, 91],  # [D, N, Rnd, All]
}

data_llama = {
    r'$Q_{new}$': [98, 95, 78, 99],  # [D, N, Rnd, All]
    r'$Q_{old}$': [93, 91, 88, 66],  # [D, N, Rnd, All]
    r'$Q_{au}$': [97, 92, 71, 99],  # [D, N, Rnd, All]
}

def plot_bars(data_gpt2,data_llama,
              save_path='/home/chenyuheng/chenyuheng/LIKN/kn2/Disturb_or_Enhance/suppress.pdf'):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    # Calculate average for each method
    for data in [data_gpt2, data_llama]:
        data['Average'] = [sum(x) / len(x) for x in
                           zip(*[data[q] for q in [r'$Q_{new}$', r'$Q_{old}$', r'$Q_{au}$']])]
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(24, 5))
    plt.subplots_adjust(wspace=0.2)

    handles, labels = [], []  # Lists to collect legend handles and labels

    subplot_data = [
        (data_gpt2, 'GPT-2'),
        (data_llama, 'LLaMA2'),
    ]

    methods = [r'$\Theta(\mathcal{D})$', r'$\Theta(\mathcal{N})$', r'$\Theta(Rnd)$', r'$\Theta(All)$']
    colors = ['lightpink', 'darkblue', 'darkgreen', 'darkviolet']

    hatches = ['', '///', '///', '///',]  # Applied to all bars for consistency
    for i, (data, title) in enumerate(subplot_data):
        ax = axes[i]
        ax.set_title(title, fontsize=30)
        ax.set_facecolor('floralwhite')
        x = np.arange(4)
        width = 0.2

        for j, method in enumerate(methods):
            values = [data[op][j] for op in [r'$Q_{new}$', r'$Q_{old}$', r'$Q_{au}$', 'Average']]
            if j ==0:
                bars = ax.bar(x + j * width, values, width, color=colors[j], label=method)
            else:
                bars = ax.bar(x + j * width, values, width, color='white',edgecolor=colors[j], label=method,hatch=hatches[j])

            if i == 0:  # Collect handles and labels from the first subplot
                handles.append(bars)
                labels.append(method)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels([r'$Q_{new}$', r'$Q_{old}$', r'$Q_{au}$', 'Average'], fontsize=26)
        ax.tick_params(axis='y', labelsize=25)

        if i == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=30)

    # Add legend to the figure
    fig.legend(handles, labels, loc="upper left", fontsize=22, bbox_to_anchor=(0.9, 1.04))

    plt.suptitle('Freeze Fine-tuning', fontsize=30, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.7)
    plt.savefig(save_path)
    plt.show()

plot_bars(data_gpt2,data_llama, save_path='/home/chenyuheng/chenyuheng/LIKN/kn2/Fine-tuning/acc.pdf')


