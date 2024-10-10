# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# accelerate launch run_swag_no_trainer.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name swag \
#   --output_dir /tmp/test-swag-no-trainer \
#   --pad_to_max_length

# python run_swag_no_trainer.py \
#   --model_name_or_path bert-base-chinese \
#   --dataset_name swag \
#   --max_seq_length 512 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./saved/$DATASET_NAME/



  # python data_format_transform.py \
  # --train_file ./data/train.json \
  # --valid_file ./data/valid.json \
  # --test_file ./data/test.json \
  # --context_file        ./data/context.json \

#0.9538
# Train multi choice
  # python train_multichoice_hw1.py \
  # --model_name_or_path bert-base-chinese \
  # --train_file ./data/train_transformed.json \
  # --validation_file ./data/valid_transformed.json \
  # --max_seq_length 512 \
  # --gradient_accumulation_steps 2 \
  # --per_device_train_batch_size 1 \
  # --per_device_eval_batch_size 1 \
  # --learning_rate 3e-5 \
  # --num_train_epochs 3 \
  # --output_dir ./saved/train_hw1/

#0.9528
# Train multi choice
  # python train_multichoice_hw1.py \
  # --model_name_or_path hfl/chinese-bert-wwm-ext \
  # --train_file ./data/train_transformed.json \
  # --validation_file ./data/valid_transformed.json \
  # --max_seq_length 512 \
  # --gradient_accumulation_steps 2 \
  # --per_device_train_batch_size 1 \
  # --per_device_eval_batch_size 1 \
  # --learning_rate 2e-5 \
  # --num_train_epochs 3 \
  # --output_dir ./saved/train_hw1/learningr_2e5_chinese-bert-wwm-ext


#0.94
  # python train_multichoice_hw1.py \
  # --model_name_or_path bert-base-chinese \
  # --train_file ./data/train_transformed.json \
  # --validation_file ./data/valid_transformed.json \
  # --max_seq_length 512 \
  # --gradient_accumulation_steps 2 \
  # --per_device_train_batch_size 2 \
  # --per_device_eval_batch_size 1 \
  # --learning_rate 6e-5 \
  # --num_train_epochs 1 \
  # --output_dir ./saved/train_hw1/epoch_1/lr_6e

# learning a 1e5 太低 loss甚至=0 感覺是"模仿"而不是學習  0.956
  # python train_multichoice_hw1.py \
  # --model_name_or_path bert-base-chinese \
  # --train_file ./data/train_transformed.json \
  # --validation_file ./data/valid_transformed.json \
  # --max_seq_length 512 \
  # --gradient_accumulation_steps 1 \
  # --per_device_train_batch_size 1 \
  # --per_device_eval_batch_size 1 \
  # --learning_rate 1e-5 \
  # --num_train_epochs 2 \
  # --output_dir ./saved/train_hw1/epoch_2/lr_1e

#0.958 epoch=2反而比較好?
  #   python train_multichoice_hw1.py \
  # --model_name_or_path bert-base-chinese \
  # --train_file ./data/train_transformed.json \
  # --validation_file ./data/valid_transformed.json \
  # --max_seq_length 512 \
  # --gradient_accumulation_steps 2 \
  # --per_device_train_batch_size 1 \
  # --per_device_eval_batch_size 1 \
  # --learning_rate 2e-5 \
  # --num_train_epochs 2 \
  # --output_dir ./saved/train_hw1/epoch_2

# # batch_16 0.963
# python train_multichoice_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base\
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 16 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 2 \
#   --output_dir ./saved/train_hw1/chinese-lert/epoch_2/batch_16

# # batch_4 0.963
# python train_multichoice_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base\
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 4 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 4e-5 \
#   --num_train_epochs 2 \
#   --output_dir ./saved/train_hw1/chinese-lert/epoch_2/batch_4


# 43428*0.05=2200 lr scheduler type to constant_with_warmup 
#0.956
# python train_multichoice_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base\
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 2 \
#   --lr_scheduler_type constant_with_warmup \
#   --num_warmup_steps 2200 \
#   --output_dir ./saved/train_hw1/chinese-lert/epoch_2/num_warmup_steps


#epoch 3-10
#0.96
# python train_multichoice_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base\
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 10 \
#   --output_dir ./saved/train_hw1/chinese-lert/epoch_10

#macbert 0.962
# python train_multichoice_hw1.py \
#   --model_name_or_path hfl/chinese-macbert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 2 \
#   --output_dir ./saved/train_hw1/macbert

  # hfl/chinese-lert
# 0.965
# python train_multichoice_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base\
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 2 \
#   --output_dir ./saved/train_hw1/chinese-lert/epoch_2

#lert-large 感覺太久
# python train_multichoice.py \
#   --model_name_or_path hfl/chinese-lert-large\
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 2 \
#   --output_dir ./saved/train_hw1/chinese-lert-large/epoch_2

#0.955
# python train_multichoice.py \
#   --model_name_or_path hfl/chinese-roberta-wwm-ext \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 1.5e-5 \
#   --num_train_epochs 2 \
#   --output_dir ./saved/train_hw1/roberta


#lr 1.5
# 0.967
python train_multichoice.py \
  --model_name_or_path hfl/chinese-lert-base \
  --train_file ./data/train_transformed.json \
  --validation_file ./data/valid_transformed.json \
  --max_seq_length 512 \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1.5e-5 \
  --num_train_epochs 2 \
  --output_dir ./MULTICHOICE_model

