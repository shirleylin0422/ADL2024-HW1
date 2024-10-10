
python train_qa.py \
  --config_name ./from_scrach_config.json \
  --tokenizer_name ./scratch_tokenizer/ \
  --train_file ./data/train_transformed.json \
  --validation_file ./data/valid_transformed.json \
  --max_seq_length 512 \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1.5e-5 \
  --num_train_epochs 3 \
  --n_best_size  50 \
  --output_dir ./QA_model/scratch

