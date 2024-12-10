import argparse
import json
import os
import random

import math
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer, logging
import time
from tqdm import tqdm
from knowledge_neurons.utils import get_model_output, load_json_files_from_directory
import wandb

wandb.init(mode="disabled")
random.seed(42)

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

    for sample in tqdm(dataset, desc="Evaluating"):
        date_prefix = f"In {sample['date']},"
        formatted_query = date_prefix + " " + sample['query'].replace('_X_.', '').strip()
        true_answer = sample['answer'][0]['name']

        predicted_answer = get_model_output(model=model, tokenizer=tokenizer, prompt=formatted_query)
        if true_answer in predicted_answer:
            correct_predictions += 1

    accuracy = correct_predictions / len(dataset)
    return accuracy


def finetune_in_neurons(method='freeze_degenerate_2'):
    model_to_finetune = None
    saved_models_dir = args.saved_models_dir
    os.makedirs(saved_models_dir, exist_ok=True)

    if method == 'freeze_degenerate_2':
        if os.path.exists(f'{saved_models_dir}/model_freeze_degenerate_2'):
            model_to_finetune = GPT2LMHeadModel.from_pretrained(f'{saved_models_dir}/model_freeze_degenerate_2')
            return model_to_finetune.to('cuda')
        else:
            model_to_finetune = GPT2LMHeadModel.from_pretrained(model_name)

    elif method == 'direct':
        if os.path.exists(f'{saved_models_dir}/model_direct'):
            model_to_finetune = GPT2LMHeadModel.from_pretrained(f'{saved_models_dir}/model_direct')
            return model_to_finetune.to('cuda')
        else:
            model_to_finetune = GPT2LMHeadModel.from_pretrained(model_name)

    elif method == 'freeze_kn':
        if os.path.exists(f'{saved_models_dir}/model_freeze_kn'):
            model_to_finetune = GPT2LMHeadModel.from_pretrained(f'{saved_models_dir}/model_freeze_kn')
            return model_to_finetune.to('cuda')
        else:
            model_to_finetune = GPT2LMHeadModel.from_pretrained(model_name)
    elif method == 'random':
        if os.path.exists(f'{saved_models_dir}/model_freeze_random'):
            model_to_finetune = GPT2LMHeadModel.from_pretrained(f'{saved_models_dir}/model_freeze_random')
            return model_to_finetune.to('cuda')
        else:
            model_to_finetune = GPT2LMHeadModel.from_pretrained(model_name)

    model_to_finetune.to('cuda')
    return model_to_finetune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Use the Pararel dataset to extract knowledge neurons from a Language Model"
    )
    parser.add_argument(
        "--local-rank", help="local rank for multigpu processing", type=int, default=0
    )
    parser.add_argument('--model_name', default='gpt2', type=str)
    parser.add_argument('--block_size', default='128', type=int)
    parser.add_argument('--data_dir', default='Templama', type=str)
    parser.add_argument('--results_dir', default='temporal_res/1118_3', type=str)
    parser.add_argument('--method', type=str,
                        default=[
                            # 'freeze_degenerate_2',
                            # 'freeze_degenerate_1',
                            # 'freeze_kn',
                            'random',
                        ])
    parser.add_argument('--saved_models_dir', type=str, default='saved_models/epoch100')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--unrelated_dataset', type=str,
                        default='Templama/filtered_test.jsonl'
                        # default='Templama/test.jsonl'  # 测试了原始的test。但是这里性能上升可能是微调的结果。
                        )
    parser.add_argument('--enhanced_dataset', type=str, default='Templama/train_enhance2.jsonl')
    args = parser.parse_args()

    data_dir = args.data_dir
    result_dir = f'{args.results_dir}/acc_for_other_data'
    data = load_json_files_from_directory(dir_path=result_dir, keyword='temporal_results')

    # Load pre-trained GPT-2 model and tokenizer
    model_name = args.model_name
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    test_dataset_for_acc = TemporalLAMADatasetForAcc(file_path=args.unrelated_dataset, tokenizer=tokenizer, block_size=args.block_size)
    # filtered_model_can_answer_test_degenerate.jsonl: DKN能够回答的答案。实验：测试不同微调方法对旧知识的影响。（即微调前就能回答对的问题），实际测试时用：filtered_test.jsonl
    #
    # 测试干扰的数据。希望比原始模型更好。
    # enhance_dataset_for_acc = TemporalLAMADatasetForAcc(file_path=args.enhanced_dataset, tokenizer=tokenizer, block_size=args.block_size)
    # train_enhance2.jsonl：用于测试增强后数据。DKN能够回答。
    # 测试时,修改了数据集.筛选了train_enhance_direct_cannot_answer.jsonl.然后获得一个新的数据集.train_enhance3.jsonl
    # Initialize the datasets
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Test the custom evaluation function

    model_original = GPT2LMHeadModel.from_pretrained(model_name)
    model_original.to('cuda')
    model_direct = finetune_in_neurons(method='direct')
    model_freeze_degenerate_2 = finetune_in_neurons(method='freeze_degenerate_2')
    model_freeze_kn = finetune_in_neurons(method='freeze_kn')
    model_freeze_random = finetune_in_neurons(method='random')
    accuracy_src_unrelated = evaluate_accuracy(model_original, tokenizer, test_dataset_for_acc, args)
    with open(f'{result_dir}/filtered_test_data.jsonl', 'a') as f:
        f.write(json.dumps({
            f'accuracy_src_unrelated': accuracy_src_unrelated
        }) + '\n')
    accuracy_direct_unrelated = evaluate_accuracy(model_direct, tokenizer, test_dataset_for_acc, args)
    with open(f'{result_dir}/filtered_test_data.jsonl', 'a') as f:
        f.write(json.dumps({
            f'accuracy_direct_unrelated': accuracy_direct_unrelated
        }) + '\n')

    accuracy_degenerate_2_unrelated = evaluate_accuracy(model_freeze_degenerate_2, tokenizer, test_dataset_for_acc, args)
    with open(f'{result_dir}/filtered_test_data.jsonl', 'a') as f:
        f.write(json.dumps({
            f'accuracy_degenerate_2_unrelated': accuracy_degenerate_2_unrelated
        }) + '\n')
    accuracy_kn_unrelated = evaluate_accuracy(model_freeze_kn, tokenizer, test_dataset_for_acc, args)
    with open(f'{result_dir}/filtered_test_data.jsonl', 'a') as f:
        f.write(json.dumps({
            f'accuracy_kn_unrelated': accuracy_kn_unrelated
        }) + '\n')
    accuracy_random_unrelated = evaluate_accuracy(model_freeze_random, tokenizer, test_dataset_for_acc, args)
    with open(f'{result_dir}/filtered_test_data.jsonl', 'a') as f:
        f.write(json.dumps({
            f'accuracy_random_unrelated': accuracy_random_unrelated
        }) + '\n')
    # with open(f'{result_dir}/all_test_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_src_unrelated': accuracy_src_unrelated
    #     }) + '\n')
    # accuracy_direct_unrelated = evaluate_accuracy(model_direct, tokenizer, test_dataset_for_acc, args)
    # with open(f'{result_dir}/all_test_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_direct_unrelated': accuracy_direct_unrelated
    #     }) + '\n')
    #
    # accuracy_degenerate_2_unrelated = evaluate_accuracy(model_freeze_degenerate_2, tokenizer, test_dataset_for_acc, args)
    # with open(f'{result_dir}/all_test_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_degenerate_2_unrelated': accuracy_degenerate_2_unrelated
    #     }) + '\n')
    # accuracy_kn_unrelated = evaluate_accuracy(model_freeze_kn, tokenizer, test_dataset_for_acc, args)
    # with open(f'{result_dir}/all_test_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_kn_unrelated': accuracy_kn_unrelated
    #     }) + '\n')
    # accuracy_random_unrelated = evaluate_accuracy(model_freeze_random, tokenizer, test_dataset_for_acc, args)
    # with open(f'{result_dir}/all_test_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_random_unrelated': accuracy_random_unrelated
    #     }) + '\n')
    # accuracy_src_enhance = evaluate_accuracy(model_original, tokenizer, enhance_dataset_for_acc, args)
    # with open(f'{result_dir}/acc_for_other_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_src_enhance': accuracy_src_enhance
    #     }) + '\n')
    #
    # accuracy_direct_enhance = evaluate_accuracy(model_direct, tokenizer, enhance_dataset_for_acc, args)
    # with open(f'{result_dir}/acc_for_other_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_direct_enhance': accuracy_direct_enhance
    #     }) + '\n')
    #
    # accuracy_degenerate_2_enhance = evaluate_accuracy(model_freeze_degenerate_2, tokenizer, enhance_dataset_for_acc, args)
    # with open(f'{result_dir}/acc_for_other_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_degenerate_enhance': accuracy_degenerate_2_enhance
    #     }) + '\n')
    #
    # accuracy_kn_enhance = evaluate_accuracy(model_freeze_kn, tokenizer, enhance_dataset_for_acc, args)
    # with open(f'{result_dir}/acc_for_other_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_kn_enhance': accuracy_kn_enhance
    #     }) + '\n')
    # accuracy_random_enhance = evaluate_accuracy(model_freeze_random, tokenizer, enhance_dataset_for_acc, args)
    # with open(f'{result_dir}/acc_for_other_data.jsonl', 'a') as f:
    #     f.write(json.dumps({
    #         f'accuracy_random_enhance': accuracy_random_enhance
    #     }) + '\n')
    #
    #
