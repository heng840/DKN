from transformers import BertTokenizer, BertLMHeadModel, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, \
    BartTokenizer, MBartForConditionalGeneration, BartForConditionalGeneration, AutoModel

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .kn import KnowledgeNeurons
from .dkn import Dkn

BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-cased", "bert-base-cased"]
GPT2_MODELS = ["gpt2", "ai-forever/mGPT"]
GPT_NEO_MODELS = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]
bart_models = ["facebook/mbart-large-50", 'facebook/bart-large']
ALL_MODELS = BERT_MODELS + GPT2_MODELS + GPT_NEO_MODELS + bart_models


def model_type(model_name: str):
    if model_name in BERT_MODELS:
        return "bert"
    elif model_name in GPT2_MODELS:
        return "gpt"
    elif model_name in GPT_NEO_MODELS:
        return "gpt_neo"
    elif model_name in bart_models:
        return 'bart'
    else:
        raise ValueError("Model {model_name} not supported")