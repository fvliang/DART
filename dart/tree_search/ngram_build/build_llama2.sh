#!/bin/bash

python3 ngram_build.py \
    --ngram_order 3 \
    --data_path "/data3/DART/ngram/train" \
    --output_path "/data3/DART/ngram/llama2/" \
    --n_jobs 16 \
    --tokenizer_name_or_path "modelscope/Llama-2-7b-ms" \
    --conversation_per_file 300