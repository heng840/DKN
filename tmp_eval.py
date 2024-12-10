import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_gpt2_neuron_overlap_modified(neurons_1, neurons_2, layers=12, neurons_per_layer=3072, save_path='gpt2_overlap.png'):
    # Dimensions for each small rectangle
    rect_height = 1.0 / layers  # Length (height) of each small rectangle
    rect_width = 2.0 / neurons_per_layer  # Width of each small rectangle

    # Adjust figure size to prevent overly elongated image
    fig_width = layers * rect_width  # Total width of the figure
    fig_height = rect_height  # Total height of the figure

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim([0, layers * rect_width])
    ax.set_ylim([0, rect_height])

    # Set x-axis and y-axis ticks
    ax.set_xticks([l * rect_width for l in range(layers)])
    ax.set_xticklabels([str(l) for l in range(layers)])
    ax.set_yticks([n * rect_height for n in range(layers)])
    ax.set_yticklabels([str(n) for n in range(layers)])

    # Fill in the base color for the entire area
    ax.add_patch(patches.Rectangle((0, 0), fig_width, fig_height, color=(1, 1, 1, 0.3), edgecolor=None))

    # Plot neurons from neurons_1
    for layer, neuron in neurons_1:
        ax.add_patch(patches.Rectangle((layer * rect_width, neuron * rect_height), rect_width, rect_height, color='blue', edgecolor=None))

    # Calculate and plot overlapping regions
    overlap_neurons = set(tuple(neuron) for neuron in neurons_1) & set(tuple(neuron) for neuron in neurons_2)
    for layer, neuron in overlap_neurons:
        ax.add_patch(patches.Rectangle((layer * rect_width, neuron * rect_height), rect_width, rect_height, color='red', edgecolor=None))

    # Add titles and labels
    ax.set_title("Overlap of Parameter Changes in GPT-2")
    ax.set_xlabel("Layer (0 to 11)")
    ax.set_ylabel("Neuron Index (0 to 3071)")
    ax.set_aspect('auto')

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
# Modified code with user's updated requirements: 100:10 aspect ratio, and rect_width and rect_height set to 1
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_rectangles(data_batch1, data_batch2, width=10, height=100, filename=None):
    fig, ax = plt.subplots(figsize=(10, 10))  # Making the plot square
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    # Set small rectangle size
    rect_width = 1
    rect_height = 1

    # Draw all small rectangles to show the grid
    for i in range(width):
        for j in range(height):
            small_rect = patches.Rectangle((i, j), rect_width, rect_height, edgecolor='lightgray', facecolor='white')
            ax.add_patch(small_rect)

    # Function to add colored small rectangles
    def add_small_rectangles(data_batch, color):
        for i, j in data_batch:
            rect = patches.Rectangle((i, j), rect_width, rect_height, edgecolor=color, facecolor=color)
            ax.add_patch(rect)

    # Add small rectangles for each data batch
    add_small_rectangles(data_batch1, 'red')
    add_small_rectangles(data_batch2, 'blue')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Neuron-position')
    ax.set_title('Overlap of Parameter Changes')

    # Save the figure
    plt.savefig(filename)

# User's updated data with new aspect ratio
neurons_1 = [(layer, neuron) for layer in range(12) for neuron in range(0, 3072, 50)]  # Neurons at every 10th position in each layer
neurons_2 = [(layer, neuron) for layer in range(12) for neuron in range(25, 3072, 50)]   # Neurons at every 10th position starting from 5 in each layer

draw_rectangles(neurons_1, neurons_2, width=12, height=3072)

"""
类似的，给出llama部分的表格。\textbf{NTC (Ours)}对应数据（注意要上色）：{
    "5": {
        "average_accuracy_drop": 8.374441412030619,
        "average_all_neurons_drop": 49.88744459525469
    },
    "2": {
        "average_accuracy_drop": 2.8108330052697954,
        "average_all_neurons_drop": 15.566176870745286
    },
    "6": {
        "average_accuracy_drop": 9.812632858810773,
        "average_all_neurons_drop": 29.593045073736928
    },
    "8": {
        "average_accuracy_drop": 7.843292339391411,
        "average_all_neurons_drop": 50.13947440310643
    },
    "10": {
        "average_accuracy_drop": 9.111411190510042,
        "average_all_neurons_drop": 31.460820411065097
    },
    "12": {
        "average_accuracy_drop": 15.157311703287924,
        "average_all_neurons_drop": 125.46061645704134
    },
    "9": {
        "average_accuracy_drop": 7.640410367679147,
        "average_all_neurons_drop": 21.549983301464685
    },
    "3": {
        "average_accuracy_drop": 4.33707254451994,
        "average_all_neurons_drop": 19.05935506241466
    },
    "4": {
        "average_accuracy_drop": 6.435972328579575,
        "average_all_neurons_drop": 37.41504143850229
    },
    "11": {
        "average_accuracy_drop": 4.5793305592831075,
        "average_all_neurons_drop": 25.66930013594468
    },
    "7": {
        "average_accuracy_drop": 9.106409588187212,
        "average_all_neurons_drop": 27.568528362215808
    },
    "14": {
        "average_accuracy_drop": 13.553088344493933,
        "average_all_neurons_drop": 37.713820406771035
    },
    "13": {
        "average_accuracy_drop": 15.80391611387849,
        "average_all_neurons_drop": 183.67566134712968
    },
    "15": {
        "average_accuracy_drop": 3.1424651723466357,
        "average_all_neurons_drop": 17.22557208783447
    },
    "17": {
        "average_accuracy_drop": 31.192376646172132,
        "average_all_neurons_drop": 134.90290122174815
    },
    "16": {
        "average_accuracy_drop": 0.9884344530253327,
        "average_all_neurons_drop": 0.4797809187016241
    },
    "overall": {
        "average_accuracy_drop": 14.106107893627838,
        "average_all_neurons_drop": 28.776432890149124
    }
}       DBSCAN对应数据：{
    "6": {
        "average_accuracy_drop": 17.60569401733246,
        "average_all_neurons_drop": 34.19945859792888
    },
    "2": {
        "average_accuracy_drop": 7.141062510524715,
        "average_all_neurons_drop": 15.4498137630158
    },
    "5": {
        "average_accuracy_drop": 8.87907630715434,
        "average_all_neurons_drop": 14.573422965207918
    },
    "9": {
        "average_accuracy_drop": 12.394545846436262,
        "average_all_neurons_drop": 28.581182240010985
    },
    "11": {
        "average_accuracy_drop": 7.184471885572738,
        "average_all_neurons_drop": 25.26159772767521
    },
    "4": {
        "average_accuracy_drop": 11.168619961234482,
        "average_all_neurons_drop": 20.982979616362122
    },
    "3": {
        "average_accuracy_drop": 8.709409857743493,
        "average_all_neurons_drop": 15.215339322477146
    },
    "7": {
        "average_accuracy_drop": 7.731958234578644,
        "average_all_neurons_drop": 18.021222427593376
    },
    "8": {
        "average_accuracy_drop": 7.695987360945098,
        "average_all_neurons_drop": 16.321925272077685
    },
    "10": {
        "average_accuracy_drop": 12.172220912825027,
        "average_all_neurons_drop": 32.13813230350819
    },
    "13": {
        "average_accuracy_drop": 92.49147931861967,
        "average_all_neurons_drop": 29.360171066658097
    },
    "12": {
        "average_accuracy_drop": 53.63039672969435,
        "average_all_neurons_drop": 125.62072477643419
    },
    "16": {
        "average_accuracy_drop": 0.178679925755477,
        "average_all_neurons_drop": 0
    },
    "overall": {
        "average_accuracy_drop": 22.257947484308687,
        "average_all_neurons_drop": 18.236953545996922
    }
}Hierarchical 对应数据：{
    "2": {
        "average_accuracy_drop": 27.5873536343497,
        "average_all_neurons_drop": 44.30165867709551
    },
    "3": {
        "average_accuracy_drop": 22.331834611013058,
        "average_all_neurons_drop": 6.501896095457458
    },
    "overall": {
        "average_accuracy_drop": 27.500157994445168,
        "average_all_neurons_drop": 44.040970659291105
    }
}K-means对应数据：{
    "5": {
        "average_accuracy_drop": 25.782581608579303,
        "average_all_neurons_drop": 41.4091950186739
    },
    "2": {
        "average_accuracy_drop": 14.733109309592011,
        "average_all_neurons_drop": 34.58034831871355
    },
    "3": {
        "average_accuracy_drop": 19.231806070917113,
        "average_all_neurons_drop": 39.137569011287006
    },
    "6": {
        "average_accuracy_drop": 14.193922658324512,
        "average_all_neurons_drop": 34.2188494530007
    },
    "11": {
        "average_accuracy_drop": 50.70075159097938,
        "average_all_neurons_drop": 110.4839431140876
    },
    "4": {
        "average_accuracy_drop": 22.200817971084305,
        "average_all_neurons_drop": 45.40794656708482
    },
    "10": {
        "average_accuracy_drop": 47.924933473198124,
        "average_all_neurons_drop": 87.79717317196368
    },
    "8": {
        "average_accuracy_drop": 37.91250718896755,
        "average_all_neurons_drop": 97.05469662900015
    },
    "12": {
        "average_accuracy_drop": 13.549145066961513,
        "average_all_neurons_drop": 23.17625257306286
    },
    "7": {
        "average_accuracy_drop": 28.58128433522617,
        "average_all_neurons_drop": 50.812947395646184
    },
    "15": {
        "average_accuracy_drop": 68.80270827265008,
        "average_all_neurons_drop": 375.0327180147892
    },
    "14": {
        "average_accuracy_drop": 36.31084393700899,
        "average_all_neurons_drop": 0
    },
    "9": {
        "average_accuracy_drop": 29.381452688085858,
        "average_all_neurons_drop": 74.86966721572684
    },
    "17": {
        "average_accuracy_drop": 4.921002106960342,
        "average_all_neurons_drop": 17.92566304900011
    },
    "13": {
        "average_accuracy_drop": 9.720969506488164,
        "average_all_neurons_drop": 9.100807675173245
    },
    "16": {
        "average_accuracy_drop": 3.328354331281301,
        "average_all_neurons_drop": 11.128475299443384
    },
    "overall": {
        "average_accuracy_drop": 20.07728007791249,
        "average_all_neurons_drop": 39.23741522926859
    }
}AMIG 对应数据不变。"""