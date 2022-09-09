
python sentiment_analysis.py \
  --test_data ../data/sample.jsonl \
  --base_model xlm-roberta-base \
  --do_test \
  --entity_property_model_path ../saved_model/category_extraction/saved_model_epoch_8.pt \
  --polarity_model_path ../saved_model/polarity_classification/saved_model_epoch_10.pt \
  --batch_size 8 \
  --max_len 256