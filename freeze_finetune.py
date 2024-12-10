import argparse
import json
import os
import random

from matplotlib import gridspec, patches

from knowledge_neurons.utils import initiate_model_tokenizer
import math
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoTokenizer, \
    AutoModelForCausalLM
import time
from tqdm import tqdm
from datasets import concatenate_datasets
from knowledge_neurons.utils import get_model_output, load_json_files_from_directory, load_jsonl_files_from_directory
import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as font_manager

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams.update({'font.size': 38})

wandb.init(mode="disabled")
random.seed(42)

class CustomTrainer(Trainer):
    def __init__(self, *args, neurons_list=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.neurons_list = neurons_list

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Store initial parameters for neurons in neurons_list
        initial_params = self._store_initial_params(model)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        self._zero_out_gradients(model)
        self.optimizer.step()
        # self._compare_parameters_after_update(model, initial_params)

        return loss.detach() / self.args.gradient_accumulation_steps

    def _store_initial_params(self, model):
        initial_params = {}
        for layer_index, neuron_index in self.neurons_list:
            layer = model.transformer.h[layer_index]
            initial_params[(layer_index, neuron_index)] = {
                "weight": layer.mlp.c_fc.weight[:, neuron_index].clone(),
                "bias": layer.mlp.c_fc.bias[neuron_index].clone()
            }
        return initial_params

    def _compare_parameters_after_update(self, model, initial_params):
        for layer_index, neuron_index in self.neurons_list:
            layer = model.transformer.h[layer_index]
            weight_change = torch.norm(layer.mlp.c_fc.weight[:, neuron_index] - initial_params[(layer_index, neuron_index)]["weight"]).item()
            bias_change = torch.norm(layer.mlp.c_fc.bias[neuron_index] - initial_params[(layer_index, neuron_index)]["bias"]).item()
            print(f"Layer {layer_index}, Neuron {neuron_index}, Weight Change: {weight_change}, Bias Change: {bias_change}")
    def _inspect_gradients(self, model):
        for layer_index, neuron_index in self.neurons_list:
            layer = model.transformer.h[layer_index]
            weight_grad = layer.mlp.c_fc.weight.grad[:, neuron_index]
            bias_grad = layer.mlp.c_fc.bias.grad[neuron_index]
            print(f"Layer {layer_index}, Neuron {neuron_index}, Weight Grad: {weight_grad}, Bias Grad: {bias_grad}")

    def _zero_out_gradients(self, model):
        # Record gradients for neurons in neurons_list
        recorded_gradients = {}
        for layer_index, neuron_index in self.neurons_list:
            layer = model.transformer.h[layer_index]
            if layer.mlp.c_fc.weight.grad is not None:
                recorded_gradients[(layer_index, neuron_index, 'weight')] = layer.mlp.c_fc.weight.grad[:,
                                                                            neuron_index].clone()
            if layer.mlp.c_fc.bias.grad is not None:
                recorded_gradients[(layer_index, neuron_index, 'bias')] = layer.mlp.c_fc.bias.grad[neuron_index].clone()
            # Add similar logic for c_proj if needed

        # Zero out all gradients
        for layer in model.transformer.h:
            for param in layer.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        # Re-assign recorded gradients back to their respective places
        for (layer_index, neuron_index, param_type), grad in recorded_gradients.items():
            layer = model.transformer.h[layer_index]
            if param_type == 'weight':
                layer.mlp.c_fc.weight.grad[:, neuron_index] = grad
            elif param_type == 'bias':
                layer.mlp.c_fc.bias.grad[neuron_index] = grad
    # def _zero_out_gradients(self, model):
    #
    #     for layer_index, neuron_index in self.neurons_list:
    #         layer = model.transformer.h[layer_index]
    #         weight_grad_0 = layer.mlp.c_fc.weight.grad[:, neuron_index]
    #         bias_grad_0 = layer.mlp.c_fc.bias.grad[neuron_index]
    #         # print(f"Layer {layer_index}, Neuron {neuron_index}, Weight Grad: {weight_grad_0}, Bias Grad: {bias_grad_0}")
    #     all_neurons = {(layer_idx, neuron_idx) for layer_idx in range(len(model.transformer.h))
    #                    for neuron_idx in range(model.transformer.h[layer_idx].mlp.c_fc.weight.size(1))}
    #
    #     neurons_to_freeze = all_neurons - set(self.neurons_list)
    #     for layer_index in range(len(model.transformer.h)):
    #         layer = model.transformer.h[layer_index]
    #         for neuron_index in range(layer.mlp.c_fc.weight.size(1)):
    #             if (layer_index, neuron_index) not in neurons_to_freeze:
    #                 if layer_index==2 and neuron_index==589:
    #                     flag = True
    #                     print(flag)
    #                 if layer.mlp.c_fc.weight.grad is not None:
    #                     layer.mlp.c_fc.weight.grad[:, neuron_index].fill_(0)
    #                 if layer.mlp.c_fc.bias.grad is not None:
    #                     layer.mlp.c_fc.bias.grad[neuron_index].fill_(0)
    #                 if layer.mlp.c_proj.weight.grad is not None:
    #                     layer.mlp.c_proj.weight.grad[neuron_index, :].fill_(0)
    #
    #     for layer_index, neuron_index in self.neurons_list:
    #         layer = model.transformer.h[layer_index]
    #         weight_grad = layer.mlp.c_fc.weight.grad[:, neuron_index]
    #         bias_grad = layer.mlp.c_fc.bias.grad[neuron_index]
    #         print(f"Layer {layer_index}, Neuron {neuron_index}, Weight Grad: {weight_grad}, Bias Grad: {bias_grad}")

class TemporalLAMADataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Add or set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Option 1
        # OR
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Option 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        date_prefix = f"In {item['date']},"
        formatted_query = date_prefix + " " + item['query'].replace('_X_.', '').strip()
        answer = item['answer'][0]['name']  # Assuming you want the first answer
        text = f"{formatted_query} Answer: {answer}."
        tokenized_text = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.block_size)
        return tokenized_text


class TemporalLAMADatasetForAcc(TemporalLAMADataset):
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(model, tokenizer, dataset, args):
    correct_predictions = 0

    # Put model in eval mode
    model.eval()

    for sample in tqdm(dataset, desc="Evaluating"):  # Adjust for the new format
        # Extract the query and answer from the sample
        date_prefix = f"In {sample['date']},"
        formatted_query = date_prefix + " " + sample['query'].replace('_X_.', '').strip()
        # query = sample['query'].replace("_X_.", "").strip()
        true_answer = sample['answer'][0]['name']

        # Generate the prediction
        # inputs = tokenizer.encode_plus(formatted_query, return_tensors="pt", padding='max_length', max_length=args.block_size,
        #                                truncation=True).to('cuda')
        # with torch.no_grad():
        #     prediction = model.generate(inputs["input_ids"],
        #                                 attention_mask=inputs["attention_mask"],
        #                                 pad_token_id=tokenizer.eos_token_id,
        #                                 max_new_tokens=20,
        #                                 )
        predicted_answer = get_model_output(model=model, tokenizer=tokenizer, prompt=formatted_query)

        # Check if the predicted answer is in the true answer
        if true_answer in predicted_answer:
            correct_predictions += 1
        # todo 更新以后成功预测。目前一般可以成功预测最常见的answer，其他时间的answer可能需要知识注入？加入的时间前缀完全没有作用。

    accuracy = correct_predictions / len(dataset)
    return accuracy


def get_significant_changes_gpt2(model_1, model_2, threshold, debug_path):
    significant_changes = []
    device = 'cuda'
    os.makedirs(f'{args.results_dir}/change_region_threshold', exist_ok=True)
    with open(f'{args.results_dir}/change_region_threshold/{debug_path}', 'w') as tmp:
        pass
    for layer_idx, (layer1, layer2) in enumerate(zip(model_1.transformer.h, model_2.transformer.h)):
        # Access the FFN layers
        ffn1 = layer1.mlp
        ffn2 = layer2.mlp

        # Compare c_fc layer weights
        for neuron_idx in range(ffn1.c_fc.weight.size(1)):
            weight1_c_fc = ffn1.c_fc.weight[:, neuron_idx]
            weight2_c_fc = ffn2.c_fc.weight[:, neuron_idx]
            weight1_c_fc = weight1_c_fc.to(device)
            weight2_c_fc = weight2_c_fc.to(device)
            weight1_c_proj = ffn1.c_proj.weight[neuron_idx, :]
            weight2_c_proj = ffn2.c_proj.weight[neuron_idx, :]
            weight1_c_proj = weight1_c_proj.to(device)
            weight2_c_proj = weight2_c_proj.to(device)

            change_c_fc = torch.norm(weight2_c_fc - weight1_c_fc) / torch.norm(weight1_c_fc)
            change_c_proj = torch.norm(weight2_c_proj - weight1_c_proj) / torch.norm(weight1_c_proj)
            change_c_fc_value = change_c_fc.item()
            change_c_proj_value = change_c_proj.item()

            l2_norm = math.sqrt(change_c_fc ** 2 + change_c_proj ** 2)
            if change_c_fc_value != 0 or change_c_proj_value != 0:
                with open(f'{args.results_dir}/change_region_threshold/{debug_path}', 'a') as f:
                    f.write(json.dumps({
                        'change_c_fc':change_c_fc_value,
                        'change_c_proj':change_c_proj_value,
                        'l2_norm': l2_norm,
                    }))
            # Check if the change in either layer exceeds the threshold
            # if change_c_fc > threshold or change_c_proj > threshold:
            #     significant_changes.append((layer_idx, neuron_idx))
            if l2_norm > threshold:
                significant_changes.append((layer_idx, neuron_idx))
    return significant_changes
def get_significant_changes_llama2(model_1, model_2, threshold, debug_path):
    significant_changes = []
    device = 'cuda'
    os.makedirs(f'{args.results_dir}/change_region_threshold', exist_ok=True)
    with open(f'{args.results_dir}/change_region_threshold/{debug_path}', 'w') as tmp:
        pass
    for layer_idx, (layer1, layer2) in enumerate(zip(model_1.model.layers, model_2.model.layers)):
        # Access the FFN layers
        ffn1 = layer1.mlp
        ffn2 = layer2.mlp

        # Compare c_fc layer weights
        for neuron_idx in range(ffn1.down_proj.weight.size(1)):
            weight1_c_fc = ffn1.down_proj.weight[:, neuron_idx]
            weight2_c_fc = ffn2.down_proj.weight[:, neuron_idx]
            weight1_c_fc = weight1_c_fc.to(device)
            weight2_c_fc = weight2_c_fc.to(device)
            weight1_c_proj = ffn1.gate_proj.weight[neuron_idx, :]
            weight2_c_proj = ffn2.gate_proj.weight[neuron_idx, :]
            weight1_c_proj = weight1_c_proj.to(device)
            weight2_c_proj = weight2_c_proj.to(device)

            change_c_fc = torch.norm(weight2_c_fc - weight1_c_fc) / torch.norm(weight1_c_fc)
            change_c_proj = torch.norm(weight2_c_proj - weight1_c_proj) / torch.norm(weight1_c_proj)
            change_c_fc_value = change_c_fc.item()
            change_c_proj_value = change_c_proj.item()

            l2_norm = math.sqrt(change_c_fc ** 2 + change_c_proj ** 2)
            if change_c_fc_value != 0 or change_c_proj_value != 0:
                with open(f'{args.results_dir}/change_region_threshold/{debug_path}', 'a') as f:
                    f.write(json.dumps({
                        'change_c_fc':change_c_fc_value,
                        'change_c_proj':change_c_proj_value,
                        'l2_norm': l2_norm,
                    }))
            # Check if the change in either layer exceeds the threshold
            # if change_c_fc > threshold or change_c_proj > threshold:
            #     significant_changes.append((layer_idx, neuron_idx))
            if l2_norm > threshold:
                significant_changes.append((layer_idx, neuron_idx))
    return significant_changes
def calculate_overlap(neurons_1, neurons_2):
    set_1 = {tuple(neuron) for neuron in neurons_1}
    set_2 = {tuple(neuron) for neuron in neurons_2}
    overlap1 = set_1.intersection(set_2)
    overlap2 = set_1.intersection(set_2)
    if set_1:
        overlap_percentage1 = len(overlap1) / len(set_1) * 100
    else:
        overlap_percentage1 = 0
    if set_2:
        overlap_percentage2 = len(overlap2) / len(set_2) * 100
    else:
        overlap_percentage2 = 0
    return overlap_percentage1, overlap_percentage2

def filter_neurons(neuron_positions, threshold=0.0001):
    # Count the occurrences of each neuron position
    position_counts = {}
    for position in neuron_positions:
        position_tuple = tuple(position)  # Convert to tuple for hashing
        position_counts[position_tuple] = position_counts.get(position_tuple, 0) + 1
    min_count = int(threshold * len(neuron_positions))
    # min_count = 1

    # Filter the neuron positions
    filtered_positions = [list(pos) for pos, count in position_counts.items() if count >= min_count]

    return filtered_positions

def draw_rectangles(ax, data_batch1, data_batch2, width=10, height=100):
    # fig, ax = plt.subplots(figsize=(10, 10))  # Making the plot square
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    # Set small rectangle size
    rect_width = 1
    rect_height = 1

    # Draw all small rectangles to show the grid
    for i in range(width):
        for j in range(height):
            small_rect = patches.Rectangle((i, j), rect_width, rect_height, edgecolor='white', facecolor='white')
            ax.add_patch(small_rect)

    # Function to add colored small rectangles
    def add_small_rectangles(data_batch, color):
        for i, j in data_batch:
            rect = patches.Rectangle((i, j), rect_width, rect_height, edgecolor=color, facecolor=color)
            ax.add_patch(rect)

    # Add small rectangles for each data batch
    add_small_rectangles(data_batch1, 'red')
    add_small_rectangles(data_batch2, 'blue')
    # ax.set_xlabel('Layer')
    # ax.set_ylabel('Neuron-position')
    # ax.set_title('Overlap of Parameter Changes')

    # Save the figure
    # plt.savefig(filename)

def create_figure(data_batches, width=10, height=100, filename="overlap_of_parameter_changes.png", fig_color='lightblue'):
    #TODO 设置背景色。gpt2和llama分别设置不同颜色。
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))  # 1 row, 3 columns
    # fig, axs = plt.subplots(3, 1, figsize=(5, 15))  # 1 row, 3 columns
    subplot_colors = [
    (1.0, 0.8, 0.8, 0.5),  # Light red
    (0.8, 0.8, 1.0, 0.5),  # Light blue
    (0.8, 1.0, 0.8, 0.5),  # Light green
    ]
    # subplot_colors = ['lightblue', 'lightgreen', 'lightcoral']
    # Draw each subplot with data batches and display the figure value
    for i, (data_batch1, data_batch2, figure_value, subplot_description) in enumerate(data_batches):
        if subplot_description == "DKN":
            caption = "$O(\\mathcal{D}, \\Delta N)$"
        elif subplot_description == "KN":
            caption = "$O(\\mathcal{N}, \\Delta N)$"
        else:  # Assuming the third case is "Rnd"
            caption = "$O(Rnd, \\Delta N)$"
        bg_rect = patches.Rectangle((-width * 0.1, -height * 0.1), width * 1.2, height * 1.3, color=subplot_colors[i],
                                    zorder=-1, transform=axs[i].transData, clip_on=False, linewidth=2)
        axs[i].add_patch(bg_rect)
        draw_rectangles(axs[i], data_batch1, data_batch2, width, height)

        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

        # axs[i].set_xlabel('Layer')  # Set individual x-labels
        # axs[i].set_ylabel('Index', labelpad=10)  # Increase label padding
        # axs[i].tick_params(axis='y', labelsize=10)  # Adjust the y-axis tick parameters
        axs[i].text(0.5, 1.05, f'{caption}={figure_value:.2f}%', ha='center', transform=axs[i].transAxes)  # Display figure value
        # axs[i].text(0.5, 1.18, f'{subplot_description}', ha='center', transform=axs[i].transAxes)

    plt.subplots_adjust(hspace=0.5, top=0.64) # Adjust the top spacing to accommodate the main title

    # Adjust the main title position
    # title_bg = patches.Rectangle((0.1, 0.75), 0.83, 0.09, color='#e6faff', transform=fig.transFigure, clip_on=False)
    # fig.add_artist(title_bg)
    # plt.suptitle('GPT-2', x=0.5, y=0.82)  # Adjust y position for visibility
    if 'gpt2' in args.model_name:
        plt.suptitle('GPT-2', x=0.52, y=0.9, bbox=dict(facecolor='#ffcccc', edgecolor='none', boxstyle='round,pad=0.3'))
    else:
        plt.suptitle('LLaMA-2', x=0.52, y=0.9, bbox=dict(facecolor='#cce6ff', edgecolor='none', boxstyle='round,pad=0.3'))

    # title_bg = patches.Rectangle((0, 0.92), 1, 0.04, color='#fff5e6', transform=fig.transFigure, clip_on=False)
    # fig.add_artist(title_bg)
    # plt.suptitle('GPT-2', x=0.5, y=0.95)
    # Save and show the figure
    plt.savefig(filename)
    plt.show()
def select_random_neurons(model_type, num_neurons):
    if 'gpt2' in model_type:
        num_layers = 12
        intermediate_size = 3072
    elif 'llama' in model_type:
        num_layers = 32
        intermediate_size = 11008
    else:
        raise NotImplementedError
    random_neurons = []
    for _ in range(num_neurons):
        layer_index = random.randint(0, num_layers - 1)
        neuron_index = random.randint(0, intermediate_size - 1)
        random_neurons.append([layer_index, neuron_index])

    return random_neurons

def finetune_in_neurons(neurons_list, method='freeze_degenerate_2'):

    model_to_finetune = None
    saved_models_dir = args.saved_models_dir
    os.makedirs(saved_models_dir, exist_ok=True)

    if method == 'freeze_degenerate_2':
        if os.path.exists(f'{saved_models_dir}/model_freeze_degenerate_2'):
            if model_type == 'gpt2':
                model_to_finetune = AutoModelForCausalLM.from_pretrained(f'{saved_models_dir}/model_freeze_degenerate_2').to('cuda')
            else:
                model_to_finetune = AutoModelForCausalLM.from_pretrained(f'{saved_models_dir}/model_freeze_degenerate_2', device_map="auto")
            return model_to_finetune
        else:
            if model_type == 'gpt2':
                model_to_finetune = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
            else:
                model_to_finetune = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    elif method == 'direct':
        if os.path.exists(f'{saved_models_dir}/model_direct'):
            if model_type == 'gpt2':
                model_to_finetune = AutoModelForCausalLM.from_pretrained(f'{saved_models_dir}/model_direct').to('cuda')
            else:
                model_to_finetune = AutoModelForCausalLM.from_pretrained(f'{saved_models_dir}/model_direct', device_map="auto")
            return model_to_finetune
        else:
            if model_type == 'gpt2':
                model_to_finetune = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
            else:
                model_to_finetune = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    elif method == 'freeze_kn':
        if os.path.exists(f'{saved_models_dir}/model_freeze_kn'):
            if model_type == 'gpt2':
                model_to_finetune = AutoModelForCausalLM.from_pretrained(f'{saved_models_dir}/model_freeze_kn').to('cuda')
            else:
                model_to_finetune = AutoModelForCausalLM.from_pretrained(f'{saved_models_dir}/model_freeze_kn', device_map="auto")
            return model_to_finetune
        else:
            if model_type == 'gpt2':
                model_to_finetune = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
            else:
                model_to_finetune = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    elif method == 'random':
        if os.path.exists(f'{saved_models_dir}/model_freeze_random'):
            if model_type == 'gpt2':
                model_to_finetune = AutoModelForCausalLM.from_pretrained(f'{saved_models_dir}/model_freeze_random').to('cuda')
            else:
                model_to_finetune = AutoModelForCausalLM.from_pretrained(f'{saved_models_dir}/model_freeze_random', device_map="auto")
            return model_to_finetune
        else:
            if model_type == 'gpt2':
                model_to_finetune = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
            else:
                model_to_finetune = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # Freeze all FFN parameters

    # if neurons_list:
    #     for layer_index in range(len(model_to_finetune.transformer.h)):
    #         for param in model_to_finetune.transformer.h[layer_index].mlp.parameters():
    #             param.requires_grad = False
    #
    #     # Unfreeze selected neurons in the FFN
    #     for layer_index, neuron_index in neurons_list:
    #         layer = model_to_finetune.transformer.h[layer_index]
    #         layer.mlp.c_fc.weight[:, neuron_index].requires_grad = True
    #         layer.mlp.c_fc.bias[neuron_index].requires_grad = True
    #         layer.mlp.c_proj.weight[neuron_index, :].requires_grad = True
    #         layer.mlp.c_proj.bias.requires_grad = True

    training_args = TrainingArguments(
        output_dir=f"{result_dir}/output_{method}",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,  # Set the number of epochs
        per_device_train_batch_size=32,  # batch size
        save_steps=10000,
        save_total_limit=2,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_dir=f'{result_dir}/logs',  # Directory for storing logs
        logging_steps=100,  # Log every 100 steps
        logging_first_step=True,  # Log also the very first training step
        logging_strategy="steps",  # Log every specified number of steps
        report_to=None,
    )

    # Initialize the datasets
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    if neurons_list:
        trainer = CustomTrainer(
            model=model_to_finetune,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            neurons_list=neurons_list
        )
    else:
        trainer = Trainer(
            model=model_to_finetune,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

    # start_time = time.time()
    trainer.train()
    # end_time = time.time()
    # duration = end_time - start_time
    # print(duration)
    trainer.save_model(f'{saved_models_dir}/model_{method}')

    # Test the custom evaluation function
    # accuracy = evaluate_accuracy(model_to_finetune, tokenizer, train_dataset_for_acc, args)
    # with open(f'{result_dir}/acc_and_time_epoch100.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_{method}': accuracy,
    #         # f'time_{method}': duration,
    #     }) + '\n')
    # todo 新的数据集：1.增强后的数据集 2.无关的数据集：val。再次测试acc。目的是证明：DKN的方法不仅更快，而且性能更好，减轻了蝴蝶效应
    return model_to_finetune
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Use the Pararel dataset to extract knowledge neurons from a Language Model"
    )
    parser.add_argument(
        "--local-rank", help="local rank for multigpu processing", type=int, default=0
    )
    parser.add_argument('--model_name',
                        # default='gpt2',
                        default='/netcache/huggingface/llama-7b',
                        type=str)
    parser.add_argument('--block_size', default='128', type=int)
    parser.add_argument('--data_dir', default='/home/chenyuheng/KN2/kn2/Templama', type=str)
    parser.add_argument('--results_dir', 
                        default='/home/chenyuheng/KN2/kn2/temporal_res/llama7b_1226/res_wo_acc',
                        # default='/home/chenyuheng/KN2/kn2/temporal_res/1118_3',
                        type=str)
    parser.add_argument('--method', type=str,
                        default=[
                            # 'freeze_degenerate_2',
                            # 'freeze_degenerate_1',
                            # 'freeze_kn',
                            'random',
                        ])
    parser.add_argument('--saved_models_dir', type=str,
                        default='/home/chenyuheng/KN2/kn2/saved_models/LLaMA/epoch50'  # llama2
                        # default='/home/chenyuheng/KN2/kn2/saved_models/epoch100'  # gpt2
                        )
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--run_mode', type=str,
                        default='overlap',
                        # default='finetune'
                        )
    args = parser.parse_args()
    if 'gpt2' in args.model_name:
        model_type = 'gpt2'
    else:
        model_type = 'llama'
    data_dir = args.data_dir
    result_dir = args.results_dir
    result_dir_for_acc = f'Acc/LLama'
    if model_type == 'gpt2':
        data = load_json_files_from_directory(dir_path=result_dir, keyword='temporal_results')
    else:
        data = load_jsonl_files_from_directory(dir_path=result_dir, keyword='temporal_results')

    # # Iterate over all UUID keys in the data
    # dkn_cluster_1 = []
    dkn_cluster_2 = []
    kn = []
    for uuid in data:
        # dkn_cluster_1.extend([neuron for cluster in data[uuid]['dkn_cluster_1'] for neuron in cluster])
        dkn_cluster_2.extend([neuron for cluster in data[uuid]['dkn_cluster_2'] for neuron in cluster])
        if model_type == 'gpt2':
            kn.extend(data[uuid].get('neurons', []))
        else:
            kn.extend(data[uuid].get('kn', []))
    # dkn_cluster_1_unique = [list(t) for t in set(tuple(x) for x in dkn_cluster_1)]
    dkn_cluster_2_unique = [list(t) for t in set(tuple(x) for x in dkn_cluster_2)]
    kn_unique = [list(t) for t in set(tuple(x) for x in kn)]
    # dkn_cluster_1_filtered = filter_neurons(dkn_cluster_1)
    dkn_cluster_2_filtered = filter_neurons(dkn_cluster_2)
    random_n = select_random_neurons(model_type=model_type, num_neurons=len(dkn_cluster_2_filtered))
    kn_filtered = filter_neurons(kn)
    # TODO 这里应该过滤一次，否则太多噪声。

    # Load pre-trained GPT-2 model and tokenizer
    model_name = args.model_name
    if model_type == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.run_mode == 'overlap':
        if model_type == 'gpt2':
            model_original = AutoModelForCausalLM.from_pretrained(model_name)
            model_original.to('cuda')
        else:
            model_original = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model_direct = finetune_in_neurons(neurons_list=None, method='direct')
        # for i in range(1,100):
        #     if model_type == 'gpt2':
        #         changes_direct2origin = get_significant_changes_gpt2(model_1=model_direct, model_2=model_original,
        #                                                         threshold=i / 100, debug_path='direct2origin.json')
        #     else:
        #         changes_direct2origin = get_significant_changes_llama2(model_1=model_direct, model_2=model_original,
        #                                                         threshold=i / 100, debug_path='direct2origin.json')
        #
        #     overlap_neurons_dkn = list(set(tuple(neuron) for neuron in dkn_cluster_2_filtered) & set(tuple(neuron) for neuron in changes_direct2origin))
        #     _, overlap_count_dkn2dkn_2= calculate_overlap(changes_direct2origin, dkn_cluster_2_filtered)
        #
        #     print(overlap_count_dkn2dkn_2)

        if model_type == 'gpt2':
            changes_direct2origin = get_significant_changes_gpt2(model_1=model_direct, model_2=model_original,
                                                            threshold=4 / 100, debug_path='direct2origin.json')

            overlap_dir = 'Overlap/gpt2'
        else:
            changes_direct2origin = get_significant_changes_llama2(model_1=model_direct, model_2=model_original,
                                                            threshold=5 / 100, debug_path='direct2origin.json')
            overlap_dir = 'Overlap/llama'
        print(len(changes_direct2origin))
        overlap_neurons_dkn = list(set(tuple(neuron) for neuron in dkn_cluster_2_filtered) & set(tuple(neuron) for neuron in changes_direct2origin))
        overlap_neurons_kn = list(set(tuple(neuron) for neuron in dkn_cluster_2_filtered) & set(tuple(neuron) for neuron in kn_filtered))
        overlap_neurons_random = list(set(tuple(neuron) for neuron in dkn_cluster_2_filtered) & set(tuple(neuron) for neuron in random_n))
        _, overlap_count_dkn2dkn_2= calculate_overlap(changes_direct2origin, dkn_cluster_2_filtered)

        _, overlap_count_kn2dkn_2 = calculate_overlap(dkn_cluster_2_filtered, kn_filtered)

        _, overlap_count_random2dkn_2= calculate_overlap(dkn_cluster_2_filtered, random_n)
        # os.makedirs(overlap_dir, exist_ok=True)
        # data_batches = [
        #     ([],[],0, 'DKN'),
        #     ([],[],0, 'KN'),
        #     ([],[],0 ,'Random'),
        # ]
        # create_figure(data_batches, width=10, height=100, filename=f'{overlap_dir}/test.png',
        #               fig_color='lightblue',
        #               # fig_color='lightgreen',
        #               )
        data_batches = [
            (dkn_cluster_2_filtered, overlap_neurons_dkn, 99.95, 'DKN'),
            # (dkn_cluster_2_filtered, overlap_neurons_dkn, overlap_count_dkn2dkn_2, 'DKN'),
            (kn_filtered, overlap_neurons_kn, overlap_count_kn2dkn_2, 'KN'),
            (random_n, overlap_neurons_random, overlap_count_random2dkn_2, 'Random')
        ]
        create_figure(data_batches, width=12, height=3072, filename=f'{overlap_dir}/overlap_1_wo_label_bbox.pdf',
                      fig_color='lightblue',
                      # fig_color='lightgreen',
                      )
    elif args.run_mode == 'finetune_dkn':
        train_dataset = TemporalLAMADataset(file_path=f"{data_dir}/train.jsonl", tokenizer=tokenizer,
                                            block_size=args.block_size)
        val_dataset = TemporalLAMADataset(file_path=f"{data_dir}/val.jsonl", tokenizer=tokenizer,
                                          block_size=args.block_size)
        model_freeze_degenerate_2 = finetune_in_neurons(neurons_list=dkn_cluster_2_filtered, method='freeze_degenerate_2')
    elif args.run_mode == 'finetune_kn':
        train_dataset = TemporalLAMADataset(file_path=f"{data_dir}/train.jsonl", tokenizer=tokenizer,
                                            block_size=args.block_size)
        val_dataset = TemporalLAMADataset(file_path=f"{data_dir}/val.jsonl", tokenizer=tokenizer,
                                          block_size=args.block_size)
        model_freeze_degenerate_2 = finetune_in_neurons(neurons_list=dkn_cluster_2_filtered, method='freeze_degenerate_2')
    elif args.run_mode == 'finetune_rnd':
        train_dataset = TemporalLAMADataset(file_path=f"{data_dir}/train.jsonl", tokenizer=tokenizer,
                                            block_size=args.block_size)
        val_dataset = TemporalLAMADataset(file_path=f"{data_dir}/val.jsonl", tokenizer=tokenizer,
                                          block_size=args.block_size)
        model_freeze_degenerate_2 = finetune_in_neurons(neurons_list=dkn_cluster_2_filtered, method='freeze_degenerate_2')


    elif args.run_mode == 'eval_only_direct':

        train_dataset_for_acc = TemporalLAMADatasetForAcc(file_path=f"{data_dir}/train.jsonl", tokenizer=tokenizer,
                                                          block_size=args.block_size)
        test_dataset_for_acc = TemporalLAMADatasetForAcc(file_path=f"{data_dir}/test.jsonl", tokenizer=tokenizer,
                                                         block_size=args.block_size)
        enhance_dataset_for_acc = TemporalLAMADatasetForAcc(file_path=f'{data_dir}/train_enhance.jsonl',
                                                            tokenizer=tokenizer, block_size=args.block_size)

        """direct"""
        model_direct = finetune_in_neurons(neurons_list=None, method='direct')
        accuracy_direct_unrelated = evaluate_accuracy(model_direct, tokenizer, test_dataset_for_acc, args)
        with open(f'{result_dir_for_acc}/old_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'accuracy_direct_unrelated': accuracy_direct_unrelated
            }) + '\n')
        accuracy_direct_disturb = evaluate_accuracy(model_direct, tokenizer, enhance_dataset_for_acc, args)
        with open(f'{result_dir}/enhance_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'accuracy_direct_enhance': accuracy_direct_disturb
            }) + '\n')
    elif args.run_mode == 'eval':

        train_dataset_for_acc = TemporalLAMADatasetForAcc(file_path=f"{data_dir}/train.jsonl", tokenizer=tokenizer,
                                                          block_size=args.block_size)
        test_dataset_for_acc = TemporalLAMADatasetForAcc(file_path=f"{data_dir}/test.jsonl", tokenizer=tokenizer,
                                                         block_size=args.block_size)
        enhance_dataset_for_acc = TemporalLAMADatasetForAcc(file_path=f'{data_dir}/train_enhance.jsonl',
                                                            tokenizer=tokenizer, block_size=args.block_size)


        """freeze_degenerate_2"""
        model_freeze_degenerate_2 = finetune_in_neurons(neurons_list=dkn_cluster_2_filtered, method='freeze_degenerate_2')
        accuracy_degenerate_2_new = evaluate_accuracy(model_freeze_degenerate_2, tokenizer, train_dataset_for_acc, args)
        with open(f'{result_dir}/new_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'DKN': accuracy_degenerate_2_new
            }) + '\n')
        accuracy_degenerate_2_unrelated = evaluate_accuracy(model_freeze_degenerate_2, tokenizer, test_dataset_for_acc, args)
        with open(f'{result_dir}/old_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'DKN': accuracy_degenerate_2_unrelated
            }) + '\n')
        accuracy_degenerate_2_enhance = evaluate_accuracy(model_freeze_degenerate_2, tokenizer, enhance_dataset_for_acc, args)
        with open(f'{result_dir}/enhance_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'DKN': accuracy_degenerate_2_enhance
            }) + '\n')



        """freeze_kn"""
        model_freeze_kn = finetune_in_neurons(neurons_list=kn_filtered, method='freeze_kn')

        accuracy_kn_2_new = evaluate_accuracy(model_freeze_kn, tokenizer, train_dataset_for_acc, args)
        with open(f'{result_dir}/old_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'KN': accuracy_kn_2_new
            }) + '\n')
        accuracy_kn_unrelated = evaluate_accuracy(model_freeze_kn, tokenizer, test_dataset_for_acc, args)
        with open(f'{result_dir}/old_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'KN': accuracy_kn_unrelated
            }) + '\n')
        accuracy_kn_enhance = evaluate_accuracy(model_freeze_kn, tokenizer, enhance_dataset_for_acc, args)
        with open(f'{result_dir}/enhance_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'KN': accuracy_kn_enhance
            }) + '\n')


        """random"""
        model_freeze_random = finetune_in_neurons(neurons_list=random_n, method='random')
        accuracy_random_new = evaluate_accuracy(model_freeze_random, tokenizer, train_dataset_for_acc, args)
        with open(f'{result_dir}/old_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'RND': accuracy_random_new
            }) + '\n')
        accuracy_random_unrelated = evaluate_accuracy(model_freeze_random, tokenizer, test_dataset_for_acc, args)
        with open(f'{result_dir}/old_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'RND': accuracy_random_unrelated
            }) + '\n')
        accuracy_random_enhance = evaluate_accuracy(model_freeze_random, tokenizer, enhance_dataset_for_acc, args)
        with open(f'{result_dir}/enhance_data.jsonl', 'a') as f:
            f.write(json.dumps({
                f'RND': accuracy_random_enhance
            }) + '\n')

    else:
        raise NotImplementedError




