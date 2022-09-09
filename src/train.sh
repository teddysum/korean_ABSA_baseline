
python sentiment_analysis.py \
  --train_data ../data/sample.jsonl \
  --dev_data ../data/sample.jsonl \
  --base_model xlm-roberta-base \
  --do_train \
  --do_eval \
  --learning_rate 3e-6 \
  --eps 1e-8 \
  --num_train_epochs 20 \
  --entity_property_model_path ../saved_model/category_extraction/ \
  --polarity_model_path ../saved_model/polarity_classification/ \
  --batch_size 8 \
  --max_len 256