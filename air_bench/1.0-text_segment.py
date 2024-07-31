import os
from eval.text_segmentation.easy_chunk import *
import datasets
import argparse
from tqdm import tqdm
import pickle
import json

def get_args():
    parser = argparse.ArgumentParser(description='Text segment arguments')
    parser.add_argument('--dataset_path', default=None, type=str, help='Path to the dataset.')
    parser.add_argument('--save_dir', default=None, type=str, help='Directory to save the results.')
    parser.add_argument('--segment', default='easy_chunk', choices=['easy_chunk', 'chunk_with_title'], help='Segmentation method to use.')
    parser.add_argument('--max_length', default=512, type=int, help='Maximum length of each chunk.')
    parser.add_argument('--keep_title', action='store_true', help='Whether to keep titles.')
    parser.add_argument('--overlap', default=0, type=int, help='Overlap between chunks.')
    return parser.parse_args()


if __name__=="__main__":
    args=get_args()
    if args.segment == 'easy_chunk':
        split_last_doc=False
    else:
        split_last_doc=True
        
    max_length=args.max_length
    save_dir=os.path.join(args.save_dir, f'{args.segment}_{max_length}_{args.keep_title}_{args.overlap}')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    corpus_list=datasets.load_dataset('json',data_files=args.dataset_path)['train']
    
    
    chunked_corpus_list = []
    chunk2doc = {}
    ptr = 0
    # corpus: {"id", "content"}
    for corpus in tqdm(corpus_list, desc=f"{args.segment}"):
        if len(corpus['content']) < max_length:
            chunk2doc[f"doc-{ptr}"] = corpus['id']
            chunked_corpus_list.append(corpus)
            ptr += 1
        else:
            segmented_corpus = chunk_with_title(corpus['content'], max_length, overlap=args.overlap, keep_title=args.keep_title, split_last_dot=split_last_doc)

            for i, segment in enumerate(segmented_corpus):
                chunk_id = f"doc-{ptr + i}"
                chunk2doc[chunk_id] = corpus['id']
                chunked_corpus_list.append({'id': chunk_id, 'content': segment})
            ptr += len(segmented_corpus)


    
    
    with open(os.path.join(save_dir, "chunked_doc.pkl"), 'wb') as f:
        pickle.dump(chunked_corpus_list, f)
    with open(os.path.join(save_dir, "chunk2doc.json"), 'w') as jf:
        json.dump(chunk2doc, jf)
    
