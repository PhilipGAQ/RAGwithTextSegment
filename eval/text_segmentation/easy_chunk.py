from tqdm import tqdm
import pickle
import json
import re
import datasets

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

def find_title(text,window_size):
    match=re.search(r'[。？ ！.?!\n]', text)
    title=text[:match.start()]+" "  
    if len(title)>window_size/5:
        title=""
    return title
    

def get_last_punctuation(text, begin, window_size):
    punctuation=['.','?','!',' ','。','？','！','\n']
    for i in range(begin+window_size ,begin-1,-1):
        if text[i] in punctuation:
            return i
    return begin+window_size
        

def chunk_with_title(text, window_size, overlap=0,keep_title=False, split_last_dot=True):
    #keep the doc title
    title=""
    if keep_title:
        title=find_title(text,window_size)
        
    segments = []
    start = 0
    end=0
    new_window_size = window_size - len(title)
    
    if split_last_dot:
        end=get_last_punctuation(text, start, window_size)
        while end<len(text):
            segments.append(title+text[start:end])
            start=end
            end=get_last_punctuation(text, start, new_window_size)
        
        if start<len(text):
            segments.append(title+ text[start:])
                
    
    else:
        
        segments.append(text[:window_size])
        start+=window_size-overlap
        
        while start + new_window_size <= len(text):
            segments.append(title+text[start:start + new_window_size])
            start += new_window_size - overlap
        # 处理最后一段可能不足window_size的情况
        if start < len(text):
            segments.append(title+text[start:])
    
    return segments
    
    
def slide_windows(corpus, max_length):
    # 正则表达式匹配句子
    pattern = r'([^。！？,.?!]+[。！？,.?!])'
    # 使用正则表达式查找所有匹配的句子
    sentences = re.findall(pattern, corpus)
    
    # 如果没有找到任何句子，将整个内容作为单一元素的列表返回
    if not sentences:
        sentences = [corpus]
    
    # 初始化结果列表
    chunked_docs = []
    current_chunk = ""
    
    # 遍历所有句子
    for sentence in sentences:
        # 如果加上新的句子不会超过最大长度，将其添加到当前块
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence  # 直接添加句子，不加空格
        else:
            # 如果超过了最大长度，将当前块添加到结果列表并开始一个新的块
            chunked_docs.append(current_chunk)
            current_chunk = sentence
    
    # 添加最后一个块，如果它不为空
    if current_chunk:
        chunked_docs.append(current_chunk)
    
    return chunked_docs

    
if __name__ =="__main__":

    pattern = r'([^。！？,.?!]+[。！？,.?!])'
    # sentences = re.search(pattern, text[::-1])
    lang='zh'
    corpus = datasets.load_dataset('Shitao/MLDR', f'corpus-{lang}', split='corpus')
    text=corpus[1]['text']
    segment=chunk_with_title(text,window_size=512,overlap=0,keep_title=True,split_last_dot=True)
    
    
    # corpus_list = [{'id': e['docid'], 'content': e['text']} for e in tqdm(corpus, desc="Generating corpus")]
    
    
    
    # chunked_corpus_list = []
    # chunk2doc = {}
    # ptr = 0

    # for corpus in tqdm(corpus_list, desc="Text Segment"):
    #     if len(corpus['content']) < 512:
    #         chunk2doc[f"doc-{lang}-{ptr}"] = corpus['id']
    #         chunked_corpus_list.append(corpus)
    #         ptr += 1
    #     else:
    #         segmented_corpus = slide_windows(corpus['content'], 512)
    #         for i, segment in enumerate(segmented_corpus):
    #             chunk_id = f"doc-{lang}-{ptr + i}"
    #             chunk2doc[chunk_id] = corpus['id']
    #             chunked_corpus_list.append({'id': chunk_id, 'content': segment})
    #         ptr += len(segmented_corpus)
    
    
    
    # sentences= chunk_with_title(text,window_size=512,overlap=100,keep_title=True,split_last_dot=True)
    for chunk in segment:
        print(chunk)
        print("\n")