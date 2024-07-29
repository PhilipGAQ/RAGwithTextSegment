import re

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
