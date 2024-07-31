"""
python 1.2-reranker_model_rerank.py \
--reranker BAAI/bge-reranker-v2-m3 \
--fp16 True \
--benchmark_dir ../0-data_generation/benchmark \
--max_length 512 --batch_size 1000 --top_k 1000 \
--search_result_save_dir ./search-results_bge-m3 \
--rerank_result_save_dir ./rerank-results \
--cache_dir /share/jianlv/.cache \
--num_shards 1 --shard_id 0
"""
import os
import copy
import json
import datasets
import pandas as pd
from tqdm import tqdm
from FlagEmbedding import FlagReranker, FlagLLMReranker
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class ModelArgs:
    reranker: str = field(
        default='BAAI/bge-reranker-v2-m3',
        metadata={'help': 'Name or path of reranker'}
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 in inference?'}
    )


@dataclass
class EvalArgs:
    benchmark_dir: str = field(
        default=None,
        metadata={'help': 'Path to benchmark data.'}
    )
    max_length: int = field(
        default=512,
        metadata={'help': 'Max text length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    top_k: int = field(
        default=100,
        metadata={'help': 'Use reranker to rerank top-k retrieval results'}
    )
    search_result_save_dir: str = field(
        default='./search-results',
        metadata={'help': 'Dir to saving search results. Search results path is `result_save_dir/{task}/{domain}/{lang}/{dataset}.txt`'}
    )
    rerank_result_save_dir: str = field(
        default='./rerank-results',
        metadata={'help': 'Dir to saving reranked results. Reranked results will be saved to `rerank_result_save_dir/{reranker}/{task}/{domain}/{lang}/{dataset}.txt`'}
    )
    cache_dir: str = field(
        default=None,
        metadata={'help': 'Cache directory for datasets library'}
    )
    num_shards: int = field(
        default=1,
        metadata={'help': "num of shards"}
    )
    shard_id: int = field(
        default=0,
        metadata={'help': 'id of shard, start from 0'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite embedding'}
    )

    ###
    dataset_save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save the results."}
    )
    segment: str = field(
        default="easy_chunk",
        metadata={"help": "Segmentation method to use.", "choices": ["easy_chunk", "chunk_with_title"]}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum length of each chunk."}
    )
    keep_title: bool = field(
        default=False,
        metadata={"help": "Whether to keep titles."}
    )
    overlap: int = field(
        default=0,
        metadata={"help": "Overlap between chunks."}
    )


def get_reranker(model_args: ModelArgs):

    reranker = FlagReranker(
        model_name_or_path=model_args.reranker,
        use_fp16=model_args.fp16
    )
    return reranker


def get_search_result_dict(search_result_path: str, top_k: int=200):
    search_result_dict = {}
    flag = True
    for _, row in pd.read_csv(search_result_path, sep=' ', header=None).iterrows():
        qid = str(row.iloc[0])
        docid = row.iloc[2]
        rank = int(row.iloc[3])
        if qid not in search_result_dict:
            search_result_dict[qid] = []
            flag = False
        if rank > top_k:
            flag = True
        if flag:
            continue
        else:
            search_result_dict[qid].append(docid)
    return search_result_dict


def get_queries_dict(test_data_path: str, cache_dir: str=None):
    test_data = datasets.load_dataset('json', data_files=test_data_path, cache_dir=cache_dir)['train']

    queries_dict = {}
    for data in test_data:
        qid = data['qid']
        query = data['query']
        queries_dict[qid] = query
    return queries_dict


def get_corpus_dict(corpus_path: str, cache_dir: str=None):
    corpus = datasets.load_dataset('json', data_files=corpus_path, cache_dir=cache_dir)['train']
    
    corpus_dict = {}
    for data in tqdm(corpus, desc="Generating corpus"):
        docid = data['docid']
        content = data['text']
        corpus_dict[docid] = content
    return corpus_dict


def save_rerank_results(
    queries_dict: dict,
    corpus_dict: dict,
    reranker: FlagReranker,
    search_result_dict: dict,
    rerank_result_save_path: str,
    rerank_result_with_text_save_path: str,
    batch_size: int=256,
    max_length: int=512
):
    qid_list = []
    sentence_pairs = []
    for qid, docids in search_result_dict.items():
        qid_list.append(qid)
        query = queries_dict[qid]
        for docid in docids:
            passage = corpus_dict[docid]
            sentence_pairs.append((query, passage))

    scores = reranker.compute_score(
        sentence_pairs,
        batch_size=batch_size,
        max_length=max_length,
        normalize=True
    )
    
    if not os.path.exists(os.path.dirname(rerank_result_save_path)):
        os.makedirs(os.path.dirname(rerank_result_save_path))
    
    with open(rerank_result_save_path, 'w', encoding='utf-8') as f1:
        with open(rerank_result_with_text_save_path, 'w', encoding='utf-8') as f2:
            i = 0
            for qid in qid_list:
                docids = search_result_dict[qid]
                docids_scores = []
                for j in range(len(docids)):
                    docids_scores.append((docids[j], scores[i + j]))
                i += len(docids)
                
                docids_scores.sort(key=lambda x: x[1], reverse=True)
                for rank, docid_score in enumerate(docids_scores):
                    docid, score = docid_score
                    line = f"{qid} Q0 {docid} {rank+1} {score:.6f} Faiss"
                    f1.write(line + '\n')
                    
                    line_with_text = {
                        'qid': qid,
                        'docid': docid,
                        'rank': rank + 1,
                        'score': score,
                        'query': queries_dict[qid],
                        'passage': corpus_dict[docid]
                    }
                    f2.write(json.dumps(line_with_text, ensure_ascii=False) + '\n')


def get_shard(search_result_dict: dict, num_shards: int, shard_id: int):
    if num_shards <= 1:
        return search_result_dict
    keys_list = sorted(list(search_result_dict.keys()))
    
    shard_len = len(keys_list) // num_shards
    if shard_id == num_shards - 1:
        shard_keys_list = keys_list[shard_id*shard_len:]
    else:
        shard_keys_list = keys_list[shard_id*shard_len : (shard_id + 1)*shard_len]
    shard_search_result_dict = {k: search_result_dict[k] for k in shard_keys_list}
    return shard_search_result_dict


def rerank_results(eval_args: EvalArgs, model_args: ModelArgs):
    eval_args = copy.deepcopy(eval_args)
    model_args = copy.deepcopy(model_args)
    
    num_shards = eval_args.num_shards
    shard_id = eval_args.shard_id
    if shard_id >= num_shards:
        raise ValueError(f"shard_id >= num_shards ({shard_id} >= {num_shards})")
    
    print("==================================================")
    print("Reranker Name:")
    print(model_args.reranker)
    
    reranker = get_reranker(model_args=model_args)
    
    if os.path.basename(model_args.reranker) in ['bge-reranker-v2-m3', 'jina-reranker-v1-turbo-en']:
        eval_args.max_length = 8192
    else:
        eval_args.max_length = 512
        
    rerank_result_save_path = os.path.join(
        eval_args.rerank_result_save_dir, 
        os.path.basename(model_args.reranker), 
        task, domain, lang, 
        f"{dataset}.txt"
    )
    
    rerank_result_with_text_save_path = os.path.join(
        eval_args.rerank_result_save_dir, 
        os.path.basename(model_args.reranker), 
        task, domain, lang, 
        f"{dataset}_with_text.jsonl"
    )
    
    if os.path.exists(rerank_result_save_path) and os.path.exists(rerank_result_with_text_save_path) and not eval_args.overwrite:
        print(f"Skip reranking results for {task} {domain} {lang} {dataset}")
        continue
    
    rerank_result_save_path = os.path.join(
        eval_args.rerank_result_save_dir, 
        os.path.basename(model_args.reranker), 
        task, domain, lang, 
        f"{dataset}_{shard_id}-of-{num_shards}.txt" 
        if num_shards > 1 else f"{dataset}.txt"
    )
    
    rerank_result_with_text_save_path = os.path.join(
        eval_args.rerank_result_save_dir, 
        os.path.basename(model_args.reranker), 
        task, domain, lang, 
        f"{dataset}_with_text_{shard_id}-of-{num_shards}.jsonl" 
        if num_shards > 1 else f"{dataset}_with_text.jsonl"
    )
    
    if os.path.exists(rerank_result_save_path) and os.path.exists(rerank_result_with_text_save_path) and not eval_args.overwrite:
        print(f"Skip reranking results for {task} {domain} {lang} {dataset}")
        continue
    
    print('--------------------------------------------------')
    print(f"Start reranking results for {task} {domain} {lang} {dataset}")
    
    corpus_path = os.path.join(eval_args.dataset_save_dir, f'{eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}')
    
    test_data_path = os.path.join(eval_args.dataset_save_dir, f'{eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}')
    
    queries_dict = get_queries_dict(test_data_path, cache_dir=eval_args.cache_dir)
    
    search_result_path = os.path.join(eval_args.search_result_save_dir, task, domain, lang, f"{dataset}.txt")
    search_result_dict = get_search_result_dict(search_result_path, top_k=eval_args.top_k)
    search_result_dict = get_shard(search_result_dict, num_shards=num_shards, shard_id=shard_id)
    
    corpus_dict = get_corpus_dict(corpus_path, cache_dir=eval_args.cache_dir)
    
    save_rerank_results(
        queries_dict=queries_dict,
        corpus_dict=corpus_dict,
        reranker=reranker,
        search_result_dict=search_result_dict,
        rerank_result_save_path=rerank_result_save_path,
        rerank_result_with_text_save_path=rerank_result_with_text_save_path,
        batch_size=eval_args.batch_size,
        max_length=eval_args.max_length
    )


def main():
    parser = HfArgumentParser([EvalArgs, ModelArgs])
    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: EvalArgs
    model_args: ModelArgs
    
    rerank_results(eval_args, model_args)
    
    print("==================================================")
    print("Done! Reranker Name:")
    print(model_args.reranker)


if __name__ == "__main__":
    main()
