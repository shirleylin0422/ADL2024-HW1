# Download Multi choice model
mkdir -p ./MULTICHOICE_model
gdown --fuzzy https://drive.google.com/file/d/1jMK0GWx1JFm1FU61tVCY-zyQ_SZjuwMs/view?usp=sharing -O ./MULTICHOICE_model/config.json
gdown --fuzzy https://drive.google.com/file/d/1baBUA0WJgsQcVvlB7O9Af-lYR3UqaYmL/view?usp=sharing -O ./MULTICHOICE_model/model.safetensors
gdown --fuzzy https://drive.google.com/file/d/1OymJJWDbjcv9f7MBKhPwuCsG99niJpHN/view?usp=sharing -O ./MULTICHOICE_model/tokenizer_config.json
gdown --fuzzy https://drive.google.com/file/d/16cIDz7F-3qR-GP7SK1k9t2tlFZiCiQPv/view?usp=sharing -O ./MULTICHOICE_model/tokenizer.json
gdown --fuzzy https://drive.google.com/file/d/1WILX0AzxoyYEn-vOIR14NhsEBKmgo-Vy/view?usp=sharing -O ./MULTICHOICE_model/vocab.txt
gdown --fuzzy https://drive.google.com/file/d/1l4NRB_PpqIqQ3qUSTiW7DAzGQ2cSwbgX/view?usp=sharing -O ./MULTICHOICE_model/special_tokens_map.json

# Download QA model
mkdir -p ./QA_model
gdown --fuzzy https://drive.google.com/file/d/1ttajjizTS4fVkbdA7aOUKx6IjY1lnmtb/view?usp=sharing -O ./QA_model/config.json
gdown --fuzzy https://drive.google.com/file/d/18NLRtJ1mjme5_tKpLfUQFP-WLXp6wV8r/view?usp=sharing -O ./QA_model/model.safetensors
gdown --fuzzy https://drive.google.com/file/d/1rOtWEo0nfpzZR0ogAZoPAcXTHeTpK6_x/view?usp=sharing -O ./QA_model/tokenizer_config.json
gdown --fuzzy https://drive.google.com/file/d/1woINrmdapjMufrYuwIoI9PxljYRmZyGd/view?usp=sharing -O ./QA_model/tokenizer.json
gdown --fuzzy https://drive.google.com/file/d/159JqFkAHrUEu0b7YY9hmeA-FdLwXOO5K/view?usp=sharing -O ./QA_model/vocab.txt
gdown --fuzzy https://drive.google.com/file/d/1FTv-PZ8YL-aEY5796S0kas7GT3T1In2O/view?usp=sharing -O ./QA_model/special_tokens_map.json

# Download data
mkdir -p ./data
gdown --fuzzy https://drive.google.com/file/d/18RgGWzFKGcrPVxcH7wDjs_7aBoyOEaKR/view?usp=sharing -O ./data/context.json
gdown --fuzzy https://drive.google.com/file/d/1SdGCuZGc5jeetEdZuFDvh9uaQ6_J_CSr/view?usp=sharing -O ./data/test.json
gdown --fuzzy https://drive.google.com/file/d/1Vtc9damr-v4ukcGsAk6ubcslhjsuXYDs/view?usp=sharing -O ./data/train.json
gdown --fuzzy https://drive.google.com/file/d/1cstyzexupzChtqim28bD9888KUkZO3EL/view?usp=sharing -O ./data/valid.json
gdown --fuzzy https://drive.google.com/file/d/1xC7PcyPUBbUeQ8okca0zztIlRxNYcGaz/view?usp=sharing -O ./data/train_transformed.json
gdown --fuzzy https://drive.google.com/file/d/1Kpu00FcHI5iJXvAMR90-uPQep_kWZfcq/view?usp=sharing -O ./data/valid_transformed.json

mkdir -p ./data/predict_output