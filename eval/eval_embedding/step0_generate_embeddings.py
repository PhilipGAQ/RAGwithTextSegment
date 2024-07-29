"""
python step0-generate_embedding.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--index_save_dir ./corpus-index \
--max_passage_length 8192 \
--batch_size 4 \
--fp16 \
--pooling_method cls \
--normalize_embeddings True
"""
import time
import os
import faiss
import datasets
import numpy as np
from tqdm import tqdm
from FlagEmbedding import FlagModel
from dataclasses import dataclass, field
from transformers import HfArgumentParser
# import sys
# sys.path.append("/data/home/angqing/code/eval/")
from eval.text_segmentation.bert_chunking import *
from eval.text_segmentation.bge_chunking import *
from eval.text_segmentation.slide_windows import *
from eval.text_segmentation.easy_chunk import *
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.nn import DataParallel
from torch.multiprocessing import Pool, set_start_method
import json
from functools import reduce
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
@dataclass
class ModelArgs:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Name or path of encoder'}
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
        default="zh",
        metadata={'help': 'Languages to evaluate. Avaliable languages: ar de en es fr hi it ja ko pt ru th zh', 
                  "nargs": "+"}
    )
    index_save_dir: str = field(
        default='./corpus-index',
        metadata={'help': 'Dir to save index. Corpus index will be saved to `index_save_dir/{encoder_name}/{lang}/index`. Corpus ids will be saved to `index_save_dir/{encoder_name}/{lang}/docid` .'}
    )
    max_passage_length: int = field(
        default=512,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite embedding'}
    )
    
    segment_method: str = field(
        default='no_segment',
        metadata={'help': 'Text segmentation method, including no segment, chunk, bert, bge_[small/base]'}
    )

    
    
def get_model(model_args: ModelArgs):
    model = FlagModel(
        model_args.encoder, 
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.fp16
    )
    return model


def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages



def chunk(lang: str,corpus_list,method:str,max_length:int,overlap=0):
    # no segment, chunk, bert, bge
    if method=='no_segment':
        # do nothing
        return corpus_list
    elif method=='easy_chunk':
        chunked_corpus_list = []
        chunk2doc = {}
        ptr = 0

        for corpus in tqdm(corpus_list, desc="Easy Chunk Text Segment"):
            if len(corpus['content']) < max_length:
                chunk2doc[f"doc-{lang}-{ptr}"] = corpus['id']
                chunked_corpus_list.append(corpus)
                ptr += 1
            else:
                segmented_corpus = slide_window_split(corpus['content'], max_length, overlap)
                for i, segment in enumerate(segmented_corpus):
                    chunk_id = f"doc-{lang}-{ptr + i}"
                    chunk2doc[chunk_id] = corpus['id']
                    chunked_corpus_list.append({'id': chunk_id, 'content': segment})
                ptr += len(segmented_corpus)


        # 保存切分后的文档和chunk到doc的映射
        with open(f'chunked_corpus_list_easy_chunk.pkl', 'wb') as f:
            pickle.dump(chunked_corpus_list, f)
        with open(f'chunk2doc_easy_chunk.json', 'w') as jf:
            json.dump(chunk2doc, jf)
        
        return chunked_corpus_list
    elif method=='chunk':
        #TODO implement chunk
        # 使用sentence切分，再加滑动窗口
        chunked_corpus_list=[]
        
        chunk2doc={}
        ptr=0
        for corpus in tqdm(corpus_list,desc="easy Text Segment:"):
            if len(corpus['content'])<max_length:
                
                chunk2doc[f"doc-{lang}-{ptr}"]=corpus['id']
                
                ptr+=1
                chunked_corpus_list.append(corpus)
            else:
                # segmented_corpus=chunk_model.chunk(doc=corpus['content'],segment_type='slide_window')
                segmented_corpus=slide_windows(corpus['content'], max_length)
                for i in range(ptr,ptr+len(segmented_corpus)):
                    chunk2doc[f"doc-{lang}-{i}"]=corpus['id']
                ptr+=len(segmented_corpus)
                chunked_corpus_list.extend(segmented_corpus)
        new_list=[{'id': f"doc-{lang}-{i}", 'content': corpus} for i, corpus in enumerate(chunked_corpus_list)]
        
        with open(f'chunked_corpus_list_{method}.pkl','wb') as f:
            pickle.dump(new_list, f)
        with open(f'chunk2doc_{method}.json','w') as jf:
            json.dump(chunk2doc,jf)
        return new_list
    
    else:
        if method=='bert':
            chunk_model=BertChunk(chunk_length=max_length,slide_window=max_length,max_length=max_length)
        elif method=='bge_small':
            chunk_model=BgeChunk(model_name_or_path='BAAI/bge-small-zh-v1.5')
        elif method=='bge_base':
            chunk_model=BgeChunk(model_name_or_path='BAAI/bge-base-zh-v1.5')
        
        chunked_corpus_list=[]
        doc2chunk={}
        
        chunk2doc={}
        ptr=0
        for corpus in tqdm(corpus_list, desc="Text Segment"):
            if len(corpus['content'])<max_length:
                doc2chunk[corpus['id']]=ptr
                
                chunk2doc[f"doc-{lang}-{ptr}"]=corpus['id']
                
                ptr+=1
                chunked_corpus_list.append(corpus)
            else:
                segmented_corpus=chunk_model.chunk(doc=corpus['content'],segment_type='slide_window')
                doc2chunk[corpus['id']]=range(ptr,ptr+len(segmented_corpus))
                for i in range(ptr,ptr+len(segmented_corpus)):
                    chunk2doc[f"doc-{lang}-{i}"]=corpus['id']
                ptr+=len(segmented_corpus)
                chunked_corpus_list.extend(segmented_corpus)
        
        new_list=[{'id': f"doc-{lang}-{i}", 'content': corpus} for i, corpus in enumerate(chunked_corpus_list)]
        
        with open(f'chunked_corpus_list_{method}.pkl','wb') as f:
            pickle.dump(new_list, f)
        
        with open(f'doc2chunk_{method}.json','w') as jf:
            json.dump(doc2chunk,jf)
        with open(f'chunk2doc_{method}.json','w') as jf:
            json.dump(chunk2doc,jf)
        return new_list
    
def load_corpus(lang: str,segment_method='no segment',max_length=512):    
    corpus = datasets.load_dataset('Shitao/MLDR', f'corpus-{lang}', split='corpus')
    # TODO     
    """
    1. split all docs from corpus which is longer than max length
    2. create a dict for original id 2 splitted id
    3. return original text after retrieval.
    
    Methods to be implemented here:
    1. No chunking, just cut down
    2. Simple chunking
    3. Bert Chunking
    4. Bge Chunking
    
    Remember to test the time costed by each method.
    """
    
    corpus_list = [{'id': e['docid'], 'content': e['text']} for e in tqdm(corpus, desc="Generating corpus")]
    chunked_corpus_list=chunk(lang,corpus_list,method=segment_method,max_length=max_length)
    corpus = datasets.Dataset.from_list(chunked_corpus_list)
    return corpus


def generate_index(model: FlagModel, corpus: datasets.Dataset, max_passage_length: int=512, batch_size: int=256):
    corpus_embeddings = model.encode_corpus(corpus["content"], batch_size=batch_size, max_length=max_passage_length)
    dim = corpus_embeddings.shape[-1]
    
    faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index, list(corpus["id"])

def save_result(index: faiss.Index, docid: list, index_save_dir: str):
    docid_save_path = os.path.join(index_save_dir, 'docid')
    index_save_path = os.path.join(index_save_dir, 'index')
    with open(docid_save_path, 'w', encoding='utf-8') as f:
        for _id in docid:
            f.write(str(_id) + '\n')
    faiss.write_index(index, index_save_path)


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs
    
    languages = check_languages(eval_args.languages)
    
    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]
    
    model = get_model(model_args=model_args)
    
    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)
    
    print("==================================================")
    print("Start generating embedding with model:")
    print(model_args.encoder)

    print('Generate embedding of following languages: ', languages)
    for lang in languages:
        print("**************************************************")
        index_save_dir = os.path.join(eval_args.index_save_dir, os.path.basename(encoder),eval_args.segment_method, lang)
        if not os.path.exists(index_save_dir):
            os.makedirs(index_save_dir)
        if os.path.exists(os.path.join(index_save_dir, 'index')) and not eval_args.overwrite:
            print(f'Embedding of {lang} already exists. Skip...')
            continue
        
        print(f"Start generating embedding of {lang} ...")
        corpus = load_corpus(lang ,eval_args.segment_method,eval_args.max_passage_length)
        
        index, docid = generate_index(
            model=model, 
            corpus=corpus,
            max_passage_length=eval_args.max_passage_length,
            batch_size=eval_args.batch_size
        )
        
        save_result(index, docid, index_save_dir)

    print("==================================================")
    print("Finish generating embeddings with model:")
    print(model_args.encoder)


if __name__ == "__main__":
    set_start_method('spawn')
    start_time=time.time()
    main()
    end_time=time.time()
    elapse=end_time-start_time
    print(f"step 0 costs: {elapse} s ")
