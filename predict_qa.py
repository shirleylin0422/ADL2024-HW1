from tqdm import tqdm
import json
import csv
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import argparse
import evaluate
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from utils_qa import postprocess_qa_predictions
to_cuda = 'cuda:0'
logger = get_logger(__name__)
parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")

parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
parser.add_argument("--do_predict", action="store_true", help="To do prediction on the question answering model")
parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
parser.add_argument(
        "--qa_output_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
parser.add_argument(
        "--csv_output_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
parser.add_argument(
        "--preprocessing_num_workers", type=int, default=1, help="A csv or a json file containing the training data."
    )
parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=False,
    )
parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )

args = parser.parse_args()
data_collator = default_data_collator
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=True, trust_remote_code=args.trust_remote_code
        )
pad_on_right = tokenizer.padding_side == "right"
config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat
def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=args.version_2_with_negative,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=args.null_score_diff_threshold,
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


data_files = {}
data_files["test"] = args.test_file
extension = args.test_file.split(".")[-1]
raw_datasets = load_dataset(extension, data_files=data_files)

column_names = raw_datasets["test"].column_names
question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "relevant" if "relevant" in column_names else column_names[1]
answer_column_name = "answer" if "answer" in column_names else column_names[2]

if args.do_predict:
    predict_examples = raw_datasets["test"]
    if args.max_predict_samples is not None:
        # We will select sample from whole data
        predict_examples = predict_examples.select(range(args.max_predict_samples))
    # Predict Feature Creation
    with accelerator.main_process_first():
        predict_dataset = predict_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
        if args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(args.max_predict_samples))
    
    predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
    predict_dataloader = DataLoader(
        predict_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    logger.info("***** Running Prediction *****")
    logger.info(f"  Num examples = {len(predict_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    all_start_logits = []
    all_end_logits = []

    model = model.to(to_cuda)
    model.eval()
    progress_bar = tqdm(total=len(predict_dataloader), desc="Processing QA prediction")
    for step, batch in enumerate(predict_dataloader):
        batch = {k: v.to(to_cuda) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())
        progress_bar.update(1)
    progress_bar.close()

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)


outputs_numpy = (start_logits_concat, end_logits_concat)
prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
start_index = np.argmax(start_logits_concat, axis=1)  # 找到每個樣本最可能的起始位置
end_index = np.argmax(end_logits_concat, axis=1)  

with open(args.test_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for i, example in enumerate(data):
    example['answer']['answer_start'] = int(start_index[i])
    example['answer']['answer_end'] = int(end_index[i])

    predicted_answer = prediction.predictions[i]
    example['answer']['text'] = predicted_answer['prediction_text']
    example['id'] = predicted_answer['id'] 

with open(args.qa_output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

csv_file = args.csv_output_file
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'answer'])
    for example in data:
        writer.writerow([example['id'], example['answer']['text']])

print(f"Processing completed. Results saved to {csv_file}")