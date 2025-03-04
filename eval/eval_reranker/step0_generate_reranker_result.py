"""
python step0-rerank_results.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ../dense_retrieval/search_results \
--rerank_result_save_dir ./rerank_results \
--top_k 200 \
--batch_size 4 \
--max_query_length 512 \
--max_passage_length 8192 \
--pooling_method cls \
--normalize_embeddings True \
--dense_weight 0.15 --sparse_weight 0.5 --colbert_weight 0.35 \
--num_shards 1 --shard_id 0 --cuda_id 0
"""
import os
import copy
import datasets
import pandas as pd
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import pickle
import json
import time

@dataclass
class ModelArgs:
    reranker: str = field(
        default='BAAI/bge-m3',
        metadata={'help': 'Name or path of reranker'}
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 in inference?'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )


@dataclass
class EvalArgs:
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: ar de en es fr hi it ja ko pt ru th zh', 
                  "nargs": "+"}
    )
    max_query_length: int = field(
        default=512,
        metadata={'help': 'Max text length.'}
    )
    max_passage_length: int = field(
        default=8192,
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
    encoder: str = field(
        default='BAAI/bge-m3',
        metadata={'help': 'Name or path of encoder'}
    )
    search_result_save_dir: str = field(
        default='./output_results',
        metadata={'help': 'Dir to saving search results. Search results path is `result_save_dir/{encoder}/{lang}.txt`'}
    )
    rerank_result_save_dir: str = field(
        default='./rerank_results',
        metadata={'help': 'Dir to saving reranked results. Reranked results will be saved to `rerank_result_save_dir/{encoder}-{reranker}/{lang}.txt`'}
    )
    num_shards: int = field(
        default=1,
        metadata={'help': "num of shards"}
    )
    shard_id: int = field(
        default=0,
        metadata={'help': 'id of shard, start from 0'}
    )
    cuda_id: int = field(
        default=0,
        metadata={'help': 'CUDA ID to use. -1 means only use CPU.'}
    )
    segment_method: str = field(
        default='no_segment',
        metadata={'help': 'Text segmentation method, including no segment, chunk, bert, bge'}
    )


def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages


def get_reranker(model_args: ModelArgs, device: str=None):
    # reranker = BGEM3FlagModel(
    #     model_name_or_path=model_args.reranker,
    #     pooling_method=model_args.pooling_method,
    #     normalize_embeddings=model_args.normalize_embeddings,
    #     device=device
    # )
    
    reranker= FlagReranker(
        model_name_or_path=model_args.reranker,
        use_fp16=model_args.fp16,
        device=device
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


def get_queries_dict(lang: str, split: str='test'):
    dataset = datasets.load_dataset('Shitao/MLDR', lang, split=split)
    
    queries_dict = {}
    for data in dataset:
        qid = data['query_id']
        query = data['query']
        queries_dict[qid] = query
    return queries_dict


def get_corpus_dict(lang: str, method):
    if method=='bert' or method=='bge_small' or method=='bge_base' or method=='chunk' or method=='easy_chunk':      
        with open(f'/data/home/angqing/code/eval/eval_embedding/chunked_corpus_list_{method}.pkl','rb') as f:
            chunked_corpus_list=pickle.load(f)
            
        corpus_dict = {}
        for data in tqdm(chunked_corpus_list,desc="Generating corpus"):
            docid=data['id']
            content=data['content']
            corpus_dict[docid]=content
            
        return corpus_dict
    else:
        corpus = datasets.load_dataset('Shitao/MLDR', f'corpus-{lang}', split='corpus')
        
        corpus_dict = {}
        for data in tqdm(corpus, desc="Generating corpus"):
            docid = data['docid']
            content = data['text']
            corpus_dict[docid] = content
        return corpus_dict
        

def save_rerank_results(queries_dict: dict, corpus_dict: dict,method, reranker: FlagReranker, search_result_dict: dict, rerank_result_save_path ,new_rerank_result_save_path , batch_size: int=256, max_query_length: int=512, max_passage_length: int=512):
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
        max_length=max_passage_length
    )
    
    if not os.path.exists(os.path.dirname(rerank_result_save_path)):
        os.makedirs(os.path.dirname(rerank_result_save_path))
    
    with open(rerank_result_save_path, 'w', encoding='utf-8') as f:
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
                f.write(line + '\n')
                
    if method=='bert' or method=='bge_small' or method=='bge_base' or method=='chunk' or method=='easy_chunk':                    
        with open(f'/data/home/angqing/code/eval/eval_embedding/chunk2doc_{method}.json' ,'r') as jf:
            chunk2doc=json.load(jf)
                        
        # save results of original doc ids:
        if not os.path.exists(os.path.dirname(new_rerank_result_save_path)):
            os.makedirs(os.path.dirname(new_rerank_result_save_path))
        
        with open(new_rerank_result_save_path, 'w', encoding='utf-8') as f:
            i = 0
            # for each qid
            for qid in qid_list:
                docids = search_result_dict[qid]
                docids_scores = []
                for j in range(len(docids)):
                    docids_scores.append((docids[j], scores[i + j]))
                i += len(docids)
                
                seen=set()
                docids_scores.sort(key=lambda x: x[1], reverse=True)
                
                #还原为doc id ,去重
                #TODO :有个问题, 这里去重之后数量变少了,是否影响后续的eval?
                new_docids_scores=[]
                for docid_score in docids_scores:
                    docid, score= docid_score
                    if chunk2doc[docid] not in seen:
                        seen.add(chunk2doc[docid])
                        new_docids_scores.append((chunk2doc[docid],score))
                    
                for rank, docid_score in enumerate(new_docids_scores):
                    docid, score = docid_score
                    line = f"{qid} Q0 {docid} {rank+1} {score:.6f} Faiss"
                    f.write(line + '\n')


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


def rerank_results(languages: list, eval_args: EvalArgs, model_args: ModelArgs, device: str=None):
    eval_args = copy.deepcopy(eval_args)
    model_args = copy.deepcopy(model_args)
    
    num_shards = eval_args.num_shards
    shard_id = eval_args.shard_id
    if shard_id >= num_shards:
        raise ValueError(f"shard_id >= num_shards ({shard_id} >= {num_shards})")
    
    reranker = get_reranker(model_args=model_args, device=device)
    
    if os.path.basename(eval_args.encoder).startswith('checkpoint-'):
        eval_args.encoder = os.path.dirname(eval_args.encoder) + '_' + os.path.basename(eval_args.encoder)
    
    if os.path.basename(model_args.reranker).startswith('checkpoint-'):
        model_args.reranker = os.path.dirname(model_args.reranker) + '_' + os.path.basename(model_args.reranker)
    
    for lang in languages:
        print("**************************************************")
        print(f"Start reranking results of {lang} ...")
        
        queries_dict = get_queries_dict(lang, split='test')
        
        search_result_save_dir = os.path.join(eval_args.search_result_save_dir, os.path.basename(eval_args.encoder))
        search_result_path = os.path.join(search_result_save_dir,eval_args.segment_method , f"{lang}.txt")
        
        search_result_dict = get_search_result_dict(search_result_path, top_k=eval_args.top_k)
        
        search_result_dict = get_shard(search_result_dict, num_shards=num_shards, shard_id=shard_id)
        
        corpus_dict = get_corpus_dict(lang, eval_args.segment_method)
        
        

        rerank_result_save_path = os.path.join(
            eval_args.rerank_result_save_dir, 
            f"{os.path.basename(eval_args.encoder)}-{os.path.basename(model_args.reranker)}", 
            eval_args.segment_method,
            f"{lang}_{shard_id}-of-{num_shards}.txt" if num_shards > 1 else f"{lang}.txt"
        )

        

        new_rerank_result_save_path = os.path.join(
            eval_args.rerank_result_save_dir, 
            f"{os.path.basename(eval_args.encoder)}-{os.path.basename(model_args.reranker)}", 
            eval_args.segment_method,
            f"{lang}_{shard_id}-of-{num_shards}_real_doc.txt" if num_shards > 1 else f"{lang}_real_doc.txt"
        )
        
        
        save_rerank_results(
            queries_dict=queries_dict,
            corpus_dict=corpus_dict, 
            reranker=reranker, 
            search_result_dict=search_result_dict, 
            rerank_result_save_path=rerank_result_save_path,
            new_rerank_result_save_path=new_rerank_result_save_path,
            batch_size=eval_args.batch_size,
            max_query_length=eval_args.max_query_length,
            max_passage_length=eval_args.max_passage_length,
            method=eval_args.segment_method
        )


def main():
    parser = HfArgumentParser([EvalArgs, ModelArgs])
    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: EvalArgs
    model_args: ModelArgs
    
    languages = check_languages(eval_args.languages)
    
    cuda_id = eval_args.cuda_id
    
    if cuda_id < 0:
        rerank_results(languages, eval_args, model_args, device='cpu')
    else:
        rerank_results(languages, eval_args, model_args, device=f"cuda:{cuda_id}")
    
    print("==================================================")
    print("Finish generating reranked results with following encoder and reranker:")
    print(eval_args.encoder)
    print(model_args.reranker)


if __name__ == "__main__":
    starttime=time.time()
    main()
    endtime=time.time()
    elapsed=endtime-starttime
    print(f'time: {elapsed}')
