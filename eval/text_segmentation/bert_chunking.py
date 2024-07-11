"""
Easy bert chunking,
Use [cls] as sentence embedding, and calculate cossim in a slide-window 
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from typing import List
import torch
from tqdm.auto import tqdm
import re
import heapq
from copy import deepcopy

class BertChunk():
    def __init__(self,doc=None,threshold=None,chunk_size=50,chunk_by_sentence=True,chunk_length=500,slide_window=500,max_length=500):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.doc=doc
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = BertModel.from_pretrained("bert-base-chinese")
        self.model = self.model.to(self.device)
        self.tokenizer = self.tokenizer
        if threshold:
            self.threshold=threshold
        else: 
            self.threshold=None
        self.chunk_size=chunk_size
        self.chunk_by_sentence=chunk_by_sentence
        self.chunk_length=chunk_length
        self.slide_window=slide_window
        self.max_length=max_length
        self.chunked_doc=None
        self.embeddings=None
        self.similarity=None

    def chunking_text_by_length(self,chunk_size=None):
        chunked_doc=[]
        if not chunk_size:
            chunk_size=self.chunk_size
        for i in range(0,len(self.doc),chunk_size):
            chunked_doc.append(self.doc[i:i+chunk_size])
        self.chunked_doc=chunked_doc
        
    def chunking_text_by_sentence(self):
        pattern = r'([^。！？]+[。！？])'
        chunked_doc = re.findall(pattern, self.doc)
        self.chunked_doc=chunked_doc

    def get_embeddings(self):
        embeddings=[]
        for chunk in tqdm(self.chunked_doc):
            inputs=self.tokenizer(chunk,return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs=self.model(**inputs)
                embedding=outputs.pooler_output.squeeze().cpu().numpy()
            embeddings.append(embedding)
        self.embeddings=embeddings

    def cossim(self):
        similarity=[]
        for i in range(len(self.embeddings)-1):
                            similarity.append(cosine_similarity(self.embeddings[i].reshape(1,-1),self.embeddings[i+1].reshape(1,-1))[0][0])
        self.similarity=similarity
        self.threshold=sum(self.similarity)/len(self.similarity)

    def text_segment(self,threshold=None):
        if not threshold:
            threshold=self.threshold
        self.split_points=[i for i, sim in enumerate(self.similarity) if sim<threshold]
        segmented_doc=[]
        chunked_doc_copy = self.chunked_doc.copy()
        ptr=0
        for point in self.split_points:
            segmented_doc.append(''.join(chunked_doc_copy[ptr:point]))
            ptr=point
        segmented_doc.append(''.join(chunked_doc_copy[ptr:]))
        self.segmented_doc=segmented_doc
        return segmented_doc
    
    def text_segment_k_mins(self,k=None):
        if not k:
            self.k=len(self.doc)//self.chunk_length+1
        else: 
            self.k=k
        threshold=self.find_k_smallest_threshold()
        self.split_points=[i for i, sim in enumerate(self.similarity) if sim<threshold]
        segmented_doc=[]
        chunked_doc_copy = self.chunked_doc.copy()
        ptr=0
        for point in self.split_points:
            segmented_doc.append(''.join(chunked_doc_copy[ptr:point]))
            ptr=point
        segmented_doc.append(''.join(chunked_doc_copy[ptr:]))
        self.segmented_doc=segmented_doc
        return segmented_doc
        
    def find_k_smallest_threshold(self):
        sim_copy=deepcopy(self.similarity)
        sim_copy.sort()
        return sim_copy[self.k-1]
        
    def text_segment_slide_window(self, slide_window=None, threshold=None):
        if not slide_window:
            slide_window = self.slide_window

        if not threshold:
            threshold = self.threshold

        chunked_doc_copy = self.chunked_doc.copy()
        segmented_doc = []
        left = 0
        right = 1
        cur_window_size = len(chunked_doc_copy[left])  # 初始化窗口大小

        while right < len(chunked_doc_copy):
            if cur_window_size + len(chunked_doc_copy[right]) <= slide_window:
                if self.similarity[right-1] > threshold:
                    cur_window_size += len(chunked_doc_copy[right])
                    right += 1
                else:
                    segmented_doc.append(''.join(chunked_doc_copy[left:right]))
                    left = right
                    right = left + 1
                    cur_window_size = len(chunked_doc_copy[left])  # 更新窗口大小
            else:
                segmented_doc.append(''.join(chunked_doc_copy[left:right]))
                left = right
                right = left + 1
                cur_window_size = len(chunked_doc_copy[left])  # 更新窗口大小

        segmented_doc.append(''.join(chunked_doc_copy[left:]))
        self.segmented_doc = segmented_doc
        return self.segmented_doc

    def text_segment_context(self, k=2, threshold=None):
        """
        根据k个上下文对该切分点进行切割
        TODO
        """
        if not threshold:
            threshold=self.threshold
        
        pass
        
    def _encode(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output.squeeze().cpu().numpy()
        return embedding

    def _cal_cossim(self, embedding1, embedding2):
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

    def clustering(self):
        """
        基于一项认知：最好不要有单句成段的现象，尽量让句子之间聚合在一起。
        对单句比较前后段落，判断其应该归属哪一段
        主要问题是，无法确定threshold的大小。
        """
        segmented_doc_copy = deepcopy(self.segmented_doc)
        i = 1  # 开始索引从1，避免越界
        while i < len(segmented_doc_copy) - 1:
            lft_emb = self._encode(segmented_doc_copy[i-1])
            rht_emb = self._encode(segmented_doc_copy[i+1])
            emb = self._encode(segmented_doc_copy[i])
            sim1 = self._cal_cossim(lft_emb, emb)
            sim2 = self._cal_cossim(rht_emb, emb)
            
            if sim1 > sim2:
                if len(segmented_doc_copy[i-1]) + len(segmented_doc_copy[i]) < self.max_length:
                    segmented_doc_copy[i-1] = segmented_doc_copy[i-1] +  segmented_doc_copy[i]
                    del segmented_doc_copy[i]
            else:
                if len(segmented_doc_copy[i]) + len(segmented_doc_copy[i+1]) < self.max_length:
                    segmented_doc_copy[i] = segmented_doc_copy[i] +  segmented_doc_copy[i+1]
                    del segmented_doc_copy[i+1]
                    # 当删除右侧元素后，无需增加i，因为i现在指向的位置已经是下一个元素
                    continue
            
            # 只有当没有发生删除操作时，才递增i
            i += 1

        return segmented_doc_copy
    
    def get_segment_doc(self,segment_type=None,threshold=None,k=None,slide_window=None,clustering=True):
        """
        segment_type: None, slide_window, k
        """
        if not self.chunked_doc:
            self.chunking_text_by_sentence()
        if not self.embeddings:
            self.get_embeddings()
        if not self.similarity:
            self.cossim()
        if not segment_type:
            self.text_segment(threshold=threshold)
        elif segment_type=='slide_window':
            self.text_segment_slide_window(slide_window=slide_window,threshold=threshold)
        elif segment_type=='k':
            self.text_segment_k_mins(k=k)
        
        if clustering:
            self.segmented_doc=self.clustering()
            
        return self.segmented_doc
    
    def chunk(self,doc,segment_type=None,threshold=None,k=None,slide_window=None,clustering=True):
        """
        segment_type: None, slide_window, k
        """
        self.doc=doc
        self.chunking_text_by_sentence()
        self.get_embeddings()
        self.cossim()
        if not segment_type:
            self.text_segment(threshold=threshold)
        elif segment_type=='slide_window':
            self.text_segment_slide_window(slide_window=slide_window,threshold=threshold)
        elif segment_type=='k':
            self.text_segment_k_mins(k=k)
        
        if clustering:
            self.segmented_doc=self.clustering()
        
        # return a 
        return self.segmented_doc

    