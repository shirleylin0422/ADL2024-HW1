# bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
# bash ./run.sh C:/Users/Shirley/Desktop/hw_test/context.json  C:/Users/Shirley/Desktop/hw_test/test.json C:/Users/Shirley/Desktop/hw_test/test_output.csv
context_file="$1"
test_file="$2"
output_csv_name="$3"

MULTICHOICE_MODEL="./MULTICHOICE_model"
MULTICHOICE_TEST_FILE="./data/predict_output/test_transformed.json"
MULTICHOICE_OUTPUT_FILE="./data/predict_output/test_multichoice_output.json"

QA_MODEL="./QA_model"
QA_TEST_FILE=$MULTICHOICE_OUTPUT_FILE
QA_OUTPUT_FILE="./data/predict_output/test_qa_output.json"
QA_OUTPUT_CSV=$output_csv_name

python data_format_transform.py \
  --test_file                $test_file \
  --test_file_transform      $MULTICHOICE_TEST_FILE \
  --context_file             $context_file \

python predict_multichoice.py \
  --model_name_or_path $MULTICHOICE_MODEL  \
  --tokenizer_name $MULTICHOICE_MODEL \
  --config_name $MULTICHOICE_MODEL \
  --max_seq_length 512 \
  --pad_to_max_length \
  --test_file  $MULTICHOICE_TEST_FILE \
  --multichoice_output_file $MULTICHOICE_OUTPUT_FILE \

python predict_qa.py \
  --model_name_or_path $QA_MODEL \
  --tokenizer_name $QA_MODEL \
  --config_name $QA_MODEL \
  --test_file $QA_TEST_FILE \
  --qa_output_file $QA_OUTPUT_FILE \
  --csv_output_file $QA_OUTPUT_CSV \
  --pad_to_max_length \
  --do_predict \
  --max_seq_length 512 \

