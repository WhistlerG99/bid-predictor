S3_BUCKET_DATA="s3://amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8/dzd_4dt0rvdnr1hoiv/5vt5uv9jpcqmxz/"
DATA_PATH="${S3_BUCKET_DATA}/data/air_canada_and_lot/bid_data_enriched_new.csv"
OUT_PATH="${S3_BUCKET_DATA}/data/air_canada_and_lot/bid_data_snapshots.csv"


spark-submit create_auction_snapshots_spark.py \
  --input $DATA_PATH \
  --output $OUT_PATH \
  --min_travel_date 2025-05-01