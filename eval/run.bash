export HF_ENDPOINT=https://hf-mirror.com
#!/bin/bash
export PYTHONPATH=/data/home/angqing/code

python /data/home/angqing/code/eval/eval_embedding/step0_generate_embeddings.py \
--encoder BAAI/bge-base-zh-v1.5 \
--languages zh \
--index_save_dir ./corpus-index \
--max_passage_length 512 \
--batch_size 4 \
--segment_method no_segment \
--fp16 \
--pooling_method cls \
--normalize_embeddings True \

python /data/home/angqing/code/eval/eval_embedding/step1_search_result.py \
--encoder BAAI/bge-base-zh-v1.5 \
--languages zh \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--segment_method bert \
--threads 16 \
--hits 1000 \
--pooling_method cls \
--normalize_embeddings True \

python /data/home/angqing/code/eval/eval_embedding/step2_eval_embedding.py \
--encoder BAAI/bge-base-zh-v1.5 \
--languages zh \
--search_result_save_dir ./search_results \
--segment_method bert \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10 \
--pooling_method cls \
--normalize_embeddings True