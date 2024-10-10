import json
import argparse

parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")

parser.add_argument(
    "--test_file",
    type=str,
    # default="./data/test.json",
    help='test.json'
)
parser.add_argument(
    "--test_file_transform",
    type=str,
    # default="./data/test_transformed.json",
    help='test.json'
)
parser.add_argument(
    "--context_file",
    type=str,
    # default="./data/context.json",
    help='context.json'
)

args = parser.parse_args()
    
with open(args.test_file, 'r', encoding='utf-8') as f:
    data_test = json.load(f)

with open(args.context_file, 'r', encoding='utf-8') as f:
    context = json.load(f)

for item in data_test:
    paragraphs = item.pop('paragraphs', [])
    label = None  
    for idx, paragraph in enumerate(paragraphs):
        item[f'paragraphs{idx}'] = context[paragraph]
    item['label'] = None
    item['relevant'] = ""
    item['answer'] = {}
    item['answer']['text'] = [""]
    item['answer']['answer_start'] = None
        

with open(args.test_file_transform, 'w', encoding='utf-8') as f:
    json.dump(data_test, f, ensure_ascii=False, indent=2)
