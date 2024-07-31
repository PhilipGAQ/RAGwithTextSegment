"""
python 1.1-embedding_model_retrieve.py \
--encoder /share/jianlv/EmbeddingBenchmark/models/bge-m3 \
--add_instruction False \
--fp16 True --pooling_method cls --normalize_embeddings True \
--benchmark_dir ../0-data_generation/benchmark \
--index_save_dir ./corpus-index \
--max_passage_length 8192 --batch_size 512 \
--result_save_dir ./search-results \
--threads 8 --hits 1000 \
--cache_dir /share/jianlv/.cache
"""
import os
import faiss
import torch
import datasets
import numpy as np
from tqdm import tqdm
from FlagEmbedding import FlagModel
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from pyserini.output_writer import get_output_writer, OutputFormat
import json
import pickle

@dataclass
class ModelArgs:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Name or path of encoder'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add instruction?'}
    )
    query_instruction_for_retrieval: str = field(
        default=None,
        metadata={'help': 'query instruction for retrieval'}
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
    benchmark_dir: str = field(
        default=None,
        metadata={'help': 'Path to benchmark data.'}
    )
    index_save_dir: str = field(
        default='./corpus-index',
        metadata={'help': 'Dir to save index. Corpus index will be saved to `index_save_dir_{encoder_name}/{task}/{domain}/{lang}/{dataset}`. Corpus ids will be saved to `index_save_dir_{encoder_name}/{domain}/{lang}/{dataset}`.'}
    )
    batch_size: int = field(
        default=4,
        metadata={'help': 'Inference batch size.'}
    )
    result_save_dir: str = field(
        default='./search-results',
        metadata={'help': 'Dir to save search results. Search results will be saved to `result_save_dir_{encoder_name}/{task}/{domain}/{lang}/{dataset}.txt`'}
    )
    threads: int = field(
        default=8,
        metadata={'help': 'Maximum threads to use during search'}
    )
    hits: int = field(
        default=1000,
        metadata={'help': 'Number of hits'}
    )
    cache_dir: str = field(
        default=None,
        metadata={'help': 'Cache directory for datasets library'}
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

def get_model(model_args: ModelArgs):
    model = FlagModel(
        model_args.encoder, 
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.fp16
    )
    return model


def get_query_encoder(model_args: ModelArgs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = AutoQueryEncoder(
        encoder_dir=model_args.encoder,
        device=device,
        pooling=model_args.pooling_method,
        l2_norm=model_args.normalize_embeddings
    )
    return model


def load_corpus(corpus_path: str, cache_dir: str=None):    
    corpus = datasets.load_dataset('json', data_files=corpus_path, cache_dir=cache_dir)['train']
    with open(os.path.join(corpus_path,"chunked_doc.pkl"), 'r') as file:
        corpus_list=pickle.load(file)        
    
    # corpus_list = [{'id': e['docid'], 'content': e['text']} for e in tqdm(corpus, desc="Generating corpus")]

    corpus = datasets.Dataset.from_list(corpus_list)
    return corpus


def generate_index(model: FlagModel, corpus: datasets.Dataset, max_passage_length: int=512, batch_size: int=256):
    corpus_embeddings = model.encode_corpus(corpus["content"], batch_size=batch_size, max_length=max_passage_length)
    dim = corpus_embeddings.shape[-1]
    
    faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index, list(corpus["id"])


def save_index(index: faiss.Index, docid: list, index_save_dir: str):
    docid_save_path = os.path.join(index_save_dir, 'docid')
    index_save_path = os.path.join(index_save_dir, 'index')
    with open(docid_save_path, 'w', encoding='utf-8') as f:
        for _id in docid:
            f.write(str(_id) + '\n')
    faiss.write_index(index, index_save_path)
    print(f'Index saved to {index_save_path}')


def build_index(model: FlagModel, corpus_path: str, index_save_dir: str, max_passage_length: int=8192, batch_size: int=4, cache_dir: str=None):
    corpus = load_corpus(corpus_path, cache_dir=cache_dir)
    
    index, docid = generate_index(
        model=model,
        corpus=corpus,
        max_passage_length=max_passage_length,
        batch_size=batch_size
    )
    
    save_index(index, docid, index_save_dir)


def get_queries_and_qids(test_data_path: str, cache_dir: str=None):
    test_data = datasets.load_dataset('json', data_files=test_data_path, cache_dir=cache_dir)['train']

    queries = []
    qids = []
    for data in test_data:
        qids.append(str(data['qid']))
        queries.append(str(data['query']))
    return queries, qids


def save_search_result(search_results, result_save_path: str, qids: list, max_hits: int):
    output_writer = get_output_writer(result_save_path, OutputFormat(OutputFormat.TREC.value), 'w',
                                      max_hits=max_hits, tag='Faiss', topics=qids,
                                      use_max_passage=False,
                                      max_passage_delimiter='#',
                                      max_passage_hits=1000)
    with output_writer:
        for topic, hits in search_results:
            output_writer.write(topic, hits)

    print(f'Search results saved to {result_save_path}')


def search_results(result_save_path: str, index_save_dir: str, query_encoder: AutoQueryEncoder, test_data_path: str, hits: int=1000, threads: int=8, cache_dir: str=None):
    searcher = FaissSearcher(
        index_dir=index_save_dir,
        query_encoder=query_encoder
    )
    
    queries, qids = get_queries_and_qids(test_data_path, cache_dir=cache_dir)
    search_results = searcher.batch_search(
        queries=queries,
        q_ids=qids,
        k=hits,
        threads=threads
    )
    
    search_results = [(_id, search_results[_id]) for _id in qids]
    save_search_result(
        search_results=search_results,
        result_save_path=result_save_path, 
        qids=qids, 
        max_hits=hits
    )


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs
    
    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]
    
    model = get_model(model_args=model_args)
    query_encoder = get_query_encoder(model_args=model_args)
    
    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)
    
    print("==================================================")
    print("Model Name:")
    print(model_args.encoder)
    
    index_save_dir = eval_args.index_save_dir + '_' + os.path.basename(encoder)+f'{eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}'
    result_save_dir = eval_args.result_save_dir + '_' + os.path.basename(encoder)+f'{eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}'


    
    print('--------------------------------------------------')
    print(f"Start generating embedding for {eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}")
    
    corpus_path = os.path.join(eval_args.dataset_save_dir, f'{eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}')
    
    test_data_path = os.path.join(eval_args.dataset_save_dir, f'{eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}')
    
    task_index_save_dir = os.path.join(index_save_dir, f'{eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}')
    
    os.makedirs(task_index_save_dir, exist_ok=True)
    if os.path.exists(os.path.join(task_index_save_dir, 'index')) and not eval_args.overwrite:
        print(f'Embedding already exists in {task_index_save_dir}. Skip...')
    else:
        build_index(
            model=model,
            corpus_path=corpus_path,
            index_save_dir=task_index_save_dir,
            max_passage_length=eval_args.max_length,
            batch_size=eval_args.batch_size,
            cache_dir=eval_args.cache_dir
        )
    
    print('--------------------------------------------------')
    print(f"Start generating search results for {eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}")
    
    task_result_save_path = os.path.join(result_save_dir,f'{eval_args.segment}_{eval_args.max_length}_{eval_args.keep_title}_{eval_args.overlap}','result.txt')
    search_results(
        result_save_path=task_result_save_path,
        index_save_dir=task_index_save_dir,
        query_encoder=query_encoder,
        test_data_path=test_data_path,
        hits=eval_args.hits,
        threads=eval_args.threads,
        cache_dir=eval_args.cache_dir
    )
    
    print("==================================================")
    print("Done! Model Name:")
    print(model_args.encoder)


if __name__ == "__main__":
    main()
