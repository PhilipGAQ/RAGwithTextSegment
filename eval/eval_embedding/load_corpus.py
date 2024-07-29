import os
import json
import pickle
import torch
from torch.multiprocessing import Process, Queue, set_start_method
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
import datasets
from copy import deepcopy

# 假设 BertChunk 类已经实现了对 device 参数的支持
from eval.text_segmentation.bert_chunking import *
from eval.text_segmentation.bge_chunking import *

method = 'bert'

def merge_dicts(dicts):
    merged_dict = {}
    for d in dicts:
        merged_dict.update(d)
    return merged_dict

def worker(rank, world_size, queue, lang, corpus_list, method, max_length, device):
    torch.cuda.set_device(device)
    
    # 确保模型在正确的设备上
    model = BertChunk(chunk_length=max_length, slide_window=max_length, max_length=max_length, device=device)
    
    chunked_corpus_list = []
    local_chunk2doc = {}
    ptr = 0
    
    if method == 'no segment':
        # 不进行分割，直接返回原始数据
        for corpus in tqdm(corpus_list, desc=f"Text Segment {method} on GPU{device}", position=rank):
            chunk_id = f"doc-{lang}-{ptr}-{rank}"
            local_chunk2doc[chunk_id] = corpus['id']
            ptr += 1
            chunked_corpus_list.append({'id': chunk_id, 'content': corpus['content']})
    else:
        for corpus in tqdm(corpus_list, desc=f"Text Segment {method} on GPU{device}", position=rank):
            if len(corpus['content']) < max_length:
                chunk_id = f"doc-{lang}-{ptr}-{rank}"
                local_chunk2doc[chunk_id] = corpus['id']
                ptr += 1
                chunked_corpus_list.append({'id': chunk_id, 'content': corpus['content']})
            else:
                segmented_corpus = model.chunk(doc=corpus['content'], segment_type='slide_window')
                for i, segment in enumerate(segmented_corpus):
                    chunk_id = f"doc-{lang}-{ptr}-{rank}"
                    local_chunk2doc[chunk_id] = corpus['id']
                    ptr += 1
                    chunked_corpus_list.append({'id': chunk_id, 'content': segment})
    
    # 将结果放入队列
    queue.put((chunked_corpus_list, local_chunk2doc))

def parallel_chunk_spawn(lang: str, corpus_list, method: str, max_length: int, devices):
    world_size = len(devices)
    chunked_corpus_lists = []
    local_chunk2docs = []

    # 分割数据集为子集
    chunk_size = (len(corpus_list) + world_size - 1) // world_size
    sublists = [corpus_list[i:i + chunk_size] for i in range(0, len(corpus_list), chunk_size)]
    
    queue = Queue()

    processes = []
    for i in range(world_size):
        p = Process(target=worker, args=(i, world_size, queue, lang, sublists[i], method, max_length, devices[i]))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成，并收集结果
    for _ in range(world_size):
        chunked_list, chunk2doc = queue.get()
        chunked_corpus_lists.extend(chunked_list)
        local_chunk2docs.append(chunk2doc)
    
    # 确保所有子进程都已经结束
    for p in processes:
        p.join()

    # 合并所有进程的局部 chunk2doc 字典
    chunk2doc = merge_dicts(local_chunk2docs)

    return chunked_corpus_lists, chunk2doc

def load_corpus(lang: str, segment_method='no segment', max_length=512):
    corpus = datasets.load_dataset('Shitao/MLDR', f'corpus-{lang}', split='corpus')
    corpus_list = [{'id': e['docid'], 'content': e['text']} for e in tqdm(corpus, desc="Generating corpus", leave=False)]
    
    # 获取所有可用的GPU设备，排除 cuda:0
    devices = [torch.device(f'cuda:{i}') for i in range(2, torch.cuda.device_count())]
    
    chunked_corpus_list, chunk2doc = parallel_chunk_spawn(lang, corpus_list, segment_method, max_length, devices)
    
    # 保存 chunk2doc.json 和 chunked_corpus_list.pkl 文件
    os.makedirs(f'results_{lang}', exist_ok=True)
    with open(f'results_{lang}/chunk2doc_{method}_{lang}.json', 'w') as jf:
        json.dump(chunk2doc, jf)
    with open(f'results_{lang}/chunked_corpus_list_{method}_{lang}.pkl', 'wb') as f:
        pickle.dump(chunked_corpus_list, f)

    return chunked_corpus_list

# 主程序入口
if __name__ == "__main__":
    set_start_method('spawn')
    lang = 'zh'  # 语言
    segment_method = 'slide_window'  # 分割方法
    max_length = 512  # 最大长度
    load_corpus(lang, segment_method, max_length)
