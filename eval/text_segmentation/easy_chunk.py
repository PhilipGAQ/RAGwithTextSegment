from tqdm import tqdm
import pickle
import json

def slide_window_split(text, window_size, overlap=0):
    """
    使用滑动窗口对文本进行切分，每个窗口的大小为window_size，窗口间有overlap的重叠部分。
    
    :param text: 输入的文本
    :param window_size: 每个窗口的大小
    :param overlap: 窗口间的重叠部分大小
    :return: 切分后的文本列表
    """
    segments = []
    start = 0
    while start + window_size <= len(text):
        segments.append(text[start:start + window_size])
        start += window_size - overlap
    # 处理最后一段可能不足window_size的情况
    if start < len(text):
        segments.append(text[start:])
    return segments

def process_corpus_easy_chunk(corpus_list, max_length, overlap, lang):
    chunked_corpus_list = []
    chunk2doc = {}
    ptr = 0

    for corpus in tqdm(corpus_list, desc="Easy Chunk Text Segment"):
        if len(corpus['content']) < max_length:
            chunk2doc[f"doc-{lang}-{ptr}"] = corpus['id']
            ptr += 1
            chunked_corpus_list.append(corpus)
        else:
            segmented_corpus = slide_window_split(corpus['content'], max_length, overlap)
            for i, segment in enumerate(segmented_corpus):
                chunk2doc[f"doc-{lang}-{ptr + i}"] = corpus['id']
            ptr += len(segmented_corpus)
            chunked_corpus_list.extend([{'id': f"doc-{lang}-{ptr + i}", 'content': segment} for i, segment in enumerate(segmented_corpus)])

    # 保存切分后的文档和chunk到doc的映射
    with open(f'chunked_corpus_list_easy_chunk.pkl', 'wb') as f:
        pickle.dump(chunked_corpus_list, f)
    with open(f'chunk2doc_easy_chunk.json', 'w') as jf:
        json.dump(chunk2doc, jf)
    
    return chunked_corpus_list