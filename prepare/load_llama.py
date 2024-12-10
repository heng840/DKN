import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        "Use the Pararel dataset to extract knowledge neurons from a Language Model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="EleutherAI/gpt-j-6b",
        # default='gpt2',
        # default='/home/chenyuheng/KN2/Llama/Llama7bChat', help='meta-llama/Llama-2-7b-chat-hf'
        # default="meta-llama/Llama-2-70b-chat-hf",
        default='meta-llama/Llama-2-13b-chat-hf',
    )
    args = parser.parse_args()
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)

