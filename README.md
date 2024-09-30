
---
title: ADL2024-HW1
https://www.csie.ntu.edu.tw/~miulab/f113-adl/

---

# ADL2024-HW1 
## Chinese Extractive Question Answering

### Task Description
In this homework, we need to train two natrual language understanding models. One is for **Paragraph Selection**, and another is for **Span selection**.

https://docs.google.com/presentation/d/184NEz-Oz9E-nryYuC5wfDNvafXE1eIxWR8vrt9UvMRA/edit?usp=sharing

### Environment

- python `3.12.6`
- pytorch `2.3.0`
- transformers `4.45.0.dev0`

### How to reproduce training models
#### Command for training Paragraph Selection

```shell
bash ./train_qa.sh
```
or
```shell
python train_multichoice.py \
  --model_name_or_path hfl/chinese-lert-large\
  --train_file ./data/train_transformed.json \
  --validation_file ./data/valid_transformed.json \
  --max_seq_length 512 \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --output_dir ./saved_train_model/train_multichoice/
```
#### Command for training Span selection
```shell
bash ./train_multichoice.sh
```
or
```shell
python train_qa.py \
  --model_name_or_path hfl/chinese-lert-large \
  --train_file ./data/train_transformed.json \
  --validation_file ./data/valid_transformed.json \
  --max_seq_length 512 \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1.5e-5 \
  --num_train_epochs 3 \
  --n_best_size  50 \
  --output_dir ./saved_train_model/train_qa/
```

### How to use these model for prediction
```shell=
bash ./run.sh ./data/context.json  ./data/test.json ./data/test_output.csv
```
