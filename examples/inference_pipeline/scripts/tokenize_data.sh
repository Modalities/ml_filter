#!/bin/sh

cd "$(dirname "$(realpath "$0")")"

modalities data create_raw_index --index_path ../data/fineweb-edu/preprocessed/annotated_fineweb_train_1000_samples.idx ../data/fineweb-edu/raw/annotated_fineweb_train_1000_samples.jsonl
modalities data pack_encoded_data ../configs/tokenization_config.yaml