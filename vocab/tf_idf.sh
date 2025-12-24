#!/bin/bash

VOCAB_JSON="/home/liuyuan/Class_data/data/process_1/vocab.json"
DOCS_JSON="/home/liuyuan/Class_data/data/process_1/freq.json"
OUT_DIR="/home/liuyuan/Class_data/data/IT-IDF"

mkdir -p $OUT_DIR

python TF-IDF.py \
    --vocab $VOCAB_JSON \
    --docs $DOCS_JSON \
    --out_dir $OUT_DIR \
    --min_df 5 \
    --max_df_ratio 0.5 \
    --min_tfidf 0.01

echo "[INFO] TF-IDF 过滤完成，结果输出到 $OUT_DIR"
