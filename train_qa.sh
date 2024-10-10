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
  # --original_train_file ./data/train.json \
  # --original_valid_file ./data/valid.json \
  # --context_file        ./data/context.json \



# Train QA
#79.16
#  python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-bert-wwm-ext \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./saved/train_qa_hw1/chinese-bert-wwm-ext/

#80.22
#  python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-bert-wwm-ext \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./saved/train_qa_hw1/chinese-bert-wwm-ext/learningr_2e5

#77.4
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-bert-wwm-ext \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 2 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 6e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./saved/train_qa_hw1/chinese-bert-wwm-ext/learningr_6e5

# chinese-bert-wwm 80
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-bert-wwm \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./saved/train_qa_hw1/chinese-bert-wwm/learningr_2e5/
  

# increase learning rate to 4 , 82.88 
  # python train_qa_hw1.py \
  # --model_name_or_path hfl/chinese-lert-base \
  # --train_file ./data/train_transformed.json \
  # --validation_file ./data/valid_transformed.json \
  # --max_seq_length 512 \
  # --gradient_accumulation_steps 2 \
  # --per_device_train_batch_size 2 \
  # --per_device_eval_batch_size 2 \
  # --learning_rate 4e-5 \
  # --num_train_epochs 3 \
  # --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_4e5/


# use lert model epoch 6: 83.01
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 6 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/epoch_6

#epoch 2 83.15
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 2 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/epoch_2


# batch size 1
#doc_stride
#n_best_size
#82.7
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 1 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --doc_stride 256 \
#   --n_best_size  50 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/batchs_1


# lr scheduler type to constant_with_warmup 
# num_warmup_steps = 41514*0.05~0.1
#3000=4000: 81
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --doc_stride 256 \
#   --lr_scheduler_type constant_with_warmup \
#   --num_warmup_steps 4000 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/constant_with_warmup


# # doc_stride 256
# # 82.95
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --doc_stride 256 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/doc_stride_256

# # doc_stride  83.28
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --doc_stride 64 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/doc_stride_64

##############################################################3

# batch 4  82.4
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 4 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 4e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_4e5/batch_4

# #83.28
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 16 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 5.6e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/batch_16


# 83.38 lert
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/


#83.48
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 4 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --n_best_size  50 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/batch_4

# 82
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 10 \
#   --n_best_size  50 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/final!!!/epoch_10



#79.8
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 8 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 4e-5 \
#   --num_train_epochs 6 \
#   --n_best_size  50 \
#   --lr_scheduler_type constant_with_warmup \
#   --num_warmup_steps 3500 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_4e5/epoch_6



#81.688
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 8 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 6 \
#   --n_best_size  50 \
#   --lr_scheduler_type constant_with_warmup \
#   --num_warmup_steps 3500 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/epoch_6

#macbert 82.35
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-macbert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --n_best_size  50 \
#   --output_dir ./saved/train_qa_hw1/macbert/learningr_2e5/

  # FacebookAI/roberta-base 82.68
  # python train_qa.py \
  # --model_name_or_path hfl/chinese-roberta-wwm-ext \
  # --train_file ./data/train_transformed.json \
  # --validation_file ./data/valid_transformed.json \
  # --max_seq_length 512 \
  # --gradient_accumulation_steps 2 \
  # --per_device_train_batch_size 1 \
  # --per_device_eval_batch_size 1 \
  # --learning_rate 1.5e-5 \
  # --num_train_epochs 3 \
  # --n_best_size  50 \
  # --output_dir ./saved/train_qa_hw1/roberta

#83.848
# n_best_size  50 = 70
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-base \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --n_best_size  50 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert/learningr_2e5/n_best_size_50

# lert-large 84.44
# python train_qa_hw1.py \
#   --model_name_or_path hfl/chinese-lert-large \
#   --train_file ./data/train_transformed.json \
#   --validation_file ./data/valid_transformed.json \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 2 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --n_best_size  50 \
#   --output_dir ./saved/train_qa_hw1/chinese-lert-large/learningr_2e5/



#lr =1.5e
# 84.77
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
  --output_dir ./QA_model

