import torch
from tqdm import tqdm
import json
import argparse
from itertools import chain
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoModelForMultipleChoice, AutoTokenizer, AutoConfig, default_data_collator
to_cuda = 'cuda:0'
parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
parser.add_argument(
        "--multichoice_output_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
parser.add_argument(
    "--config_name",
    type=str,
    default=None,
    help="Pretrained config name or path if not the same as model_name",
)
parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )

parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
args = parser.parse_args()


model_path = args.model_name_or_path 
model = AutoModelForMultipleChoice.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
padding = "max_length" if args.pad_to_max_length else False 
data_collator = default_data_collator

ending_names = [f"paragraphs{i}" for i in range(4)]
context_name = "question"
question_header_name = "question"
with open(args.test_file, "r", encoding="utf-8") as f:
    data = json.load(f)

progress_bar = tqdm(total=len(data), desc="Processing Multichoice prediction", unit="sample")
for item in data:
    question = item["question"]
    paragraphs = [item[f"paragraphs{i}"] for i in range(4)]
    
    inputs = tokenizer(
        [question] * len(paragraphs), 
        paragraphs,                    
        return_tensors="pt",          
        padding="max_length",                  
        truncation=True,           
        max_length=args.max_seq_length              
    )

    input_ids = inputs["input_ids"].unsqueeze(0) 
    attention_mask = inputs["attention_mask"].unsqueeze(0)  

    model = model.to(to_cuda)
    input_ids = input_ids.to(to_cuda)
    attention_mask = attention_mask.to(to_cuda)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    predicted_choice = torch.argmax(logits, dim=-1).item()
    item['label'] = predicted_choice
    item['relevant'] = paragraphs[predicted_choice]  
    progress_bar.update(1)
    

with open(args.multichoice_output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Processing completed. Results saved to {args.multichoice_output_file}")