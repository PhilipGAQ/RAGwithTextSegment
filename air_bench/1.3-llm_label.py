"""
python 1.3-llm_label.py \
--benchmark_dir ../0-data_generation/benchmark \
--search_result_dir \
    ./rerank-results/bge-reranker-large \
    ./rerank-results/bce-reranker-base_v1 \
    ./rerank-results/mmarco-mMiniLMv2-L12-H384-v1 \
--cache_dir /share/jianlv/.cache \
--tasks qa long-doc \
--domains wiki web healthcare law news finance arxiv msmarco book \
--languages en zh \
--top_k 1000 \
--potential_pos_rank_threshold 10 --false_hard_neg_rank_threshold 20 \
--llm_model gpt-4-1106-preview --llm_num_processes 100 \
--labeled_data_save_dir ./labeled_data \
--new_benchmark_save_dir ./new_benchmarks
"""
import os
import json
import argparse
import multiprocessing
from typing import List
from pprint import pprint

from utils.llm_labeler import LLMLabeler
from utils.data_loader import DataLoader
from utils.reranker_filter import RerankerFilter


def get_args():
    parser = argparse.ArgumentParser(description='Label data using rerankers and LLM')
    parser.add_argument('--benchmark_dir', type=str, help='Path to the benchmark directory')
    parser.add_argument('--search_result_dir', type=str, nargs='+', help='Path to the search result directory')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory for datasets library')
    parser.add_argument('--tasks', type=str, nargs='+', help='Task types to filter the data')
    parser.add_argument('--domains', type=str, nargs='+', help='Domains to filter the data')
    parser.add_argument('--languages', type=str, nargs='+', help='Languages to filter the data')
    parser.add_argument('--top_k', type=int, default=1000, help='Top k documents to consider')
    parser.add_argument('--potential_pos_rank_threshold', type=int, default=10, help='Rank threshold for potential positive documents')
    parser.add_argument('--false_hard_neg_rank_threshold', type=int, default=20, help='Rank threshold for hard negative documents')
    
    parser.add_argument('--llm_model', type=str, default='gpt-4-1106-preview', help='LLM model name')
    parser.add_argument('--llm_num_processes', type=int, default=1, help='Number of processes for LLM labeler')
    
    parser.add_argument('--labeled_data_save_dir', type=str, help='Path to save the labeled data')
    parser.add_argument('--new_benchmark_save_dir', type=str, help='Path to saving the new benchmark directory')
    return parser.parse_args()


def check_tasks(tasks):
    if tasks is None:
        return None
    avaliable_tasks = ['qa', 'long-doc']
    for task in tasks:
        if task not in avaliable_tasks:
            raise ValueError(f"Task type `{task}` is not supported. Avaliable task types: {avaliable_tasks}")
    return tasks


def check_languages(languages):
    if languages is None:
        return None
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['en', 'zh', 'ar', 'hi', 'bn', 'fa', 'ja', 'ko', 'fr', 'de', 'ru', 'es', 'id']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages


def check_domains(domains):
    if domains is None:
        return None
    if isinstance(domains, str):
        domains = [domains]
    avaliable_domains = ['wiki', 'web', 'healthcare', 'law', 'arxiv', 'science', 'news', 'finance', 'msmarco', 'book']
    for domain in domains:
        if domain not in avaliable_domains:
            raise ValueError(f"Domain `{domain}` is not supported. Avaliable domains: {avaliable_domains}")
    return domains


def save_labeled_data(pairs_labeled: List[dict], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'pairs_labeled.jsonl')
    readable_save_path = os.path.join(save_dir, 'pairs_labeled_readable.jsonl')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for data in pairs_labeled:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    with open(readable_save_path, 'w', encoding='utf-8') as f:
        for data in pairs_labeled:
            f.write(json.dumps(data, ensure_ascii=False, indent=4) + '\n')
    
    print(f"Saved labeled data to `{save_path}` and `{readable_save_path}`")


def label_by_llm(pairs_to_be_labeled: List[dict], llm_model: str, tqdm_desc: str="Labeling"):
    llm_labeler = LLMLabeler(llm_model)
    
    pairs_labeled = llm_labeler.label(pairs_to_be_labeled, tqdm_desc=tqdm_desc)
    return pairs_labeled


def get_labeled_data_info(pairs_labeled: List[dict]):
    labeled_data_info_dict = {
        'total_pairs': len(pairs_labeled),
        'failed_pairs': 0,
        'original_pos_pairs': 0,
        'bad_original_pos_pairs': 0,
        'potential_pos_pairs': 0,
        'real_potential_pos_pairs': 0,
        'false_hard_neg_pairs': 0,
        'real_false_hard_neg_pairs': 0
    }
    for pair in pairs_labeled:
        if pair['llm_raw_label'] == -1:
            labeled_data_info_dict['failed_pairs'] += 1
        
        if pair['type'] == 'original_pos':
            labeled_data_info_dict['original_pos_pairs'] += 1
            if pair['llm_label'] == 0:
                labeled_data_info_dict['bad_original_pos_pairs'] += 1
        elif pair['type'] == 'potential_pos':
            labeled_data_info_dict['potential_pos_pairs'] += 1
            if pair['llm_label'] == 1:
                labeled_data_info_dict['real_potential_pos_pairs'] += 1
        elif pair['type'] == 'false_hard_neg':
            labeled_data_info_dict['false_hard_neg_pairs'] += 1
            if pair['llm_label'] == 1:
                labeled_data_info_dict['real_false_hard_neg_pairs'] += 1
    return labeled_data_info_dict


def parse_pairs_labeled(pairs_labeled: List[dict]):
    bad_original_pos_pairs = []
    real_potential_pos_pairs = []
    real_false_hard_neg_pairs = []
    for pair in pairs_labeled:
        if pair['type'] == 'original_pos' and pair['llm_label'] == 0:
            bad_original_pos_pairs.append(pair)
        elif pair['type'] == 'potential_pos' and pair['llm_label'] == 1:
            real_potential_pos_pairs.append(pair)
        elif pair['type'] == 'false_hard_neg' and pair['llm_label'] == 1:
            real_false_hard_neg_pairs.append(pair)
    return bad_original_pos_pairs, real_potential_pos_pairs, real_false_hard_neg_pairs


def save_new_benchmark(data_dict: dict, new_benchmark_save_dir: str, pairs_labeled: List[dict]):
    os.makedirs(new_benchmark_save_dir, exist_ok=True)
    
    corpus_dict = data_dict['corpus_dict']
    original_triplets = data_dict['original_triplets']
    
    bad_original_pos_pairs, real_potential_pos_pairs, real_false_hard_neg_pairs = parse_pairs_labeled(pairs_labeled)
    
    # new corpus
    real_false_hard_neg_docid_set = set([pair['docid'] for pair in real_false_hard_neg_pairs])
    new_corpus_list = []
    for docid, text in corpus_dict.items():
        if docid in real_false_hard_neg_docid_set:
            continue
        new_corpus_list.append({'docid': docid, 'text': text})
    
    new_corpus_save_path = os.path.join(new_benchmark_save_dir, 'corpus.jsonl')
    with open(new_corpus_save_path, 'w', encoding='utf-8') as f:
        for data in new_corpus_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # new triplets
    bad_qid_set = set([pair['qid'] for pair in bad_original_pos_pairs])
    real_potential_pos_dict = {}
    for pair in real_potential_pos_pairs:
        if pair['qid'] not in real_potential_pos_dict:
            real_potential_pos_dict[pair['qid']] = []
        real_potential_pos_dict[pair['qid']].append(pair['docid'])

    new_triplets_list = []
    for triplet in original_triplets:
        if triplet['qid'] in bad_qid_set:
            continue
        if triplet['qid'] in real_potential_pos_dict:
            for potential_pos_docid in real_potential_pos_dict[triplet['qid']]:
                if potential_pos_docid not in real_false_hard_neg_docid_set:
                    triplet['pos'].append({
                        'docid': potential_pos_docid,
                        'text': corpus_dict[potential_pos_docid],
                    })
        old_hard_neg_list = triplet['hard_neg']
        new_hard_neg_list = []
        for hard_neg in old_hard_neg_list:
            if hard_neg['docid'] not in real_false_hard_neg_docid_set:
                new_hard_neg_list.append(hard_neg)
        
        triplet['hard_neg'] = new_hard_neg_list
        new_triplets_list.append(triplet)
    
    new_test_data_save_path = os.path.join(new_benchmark_save_dir, 'test_data.jsonl')
    new_test_data_readable_save_path = os.path.join(new_benchmark_save_dir, 'test_data_readable.jsonl')
    with open(new_test_data_save_path, 'w', encoding='utf-8') as f:
        for data in new_triplets_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    with open(new_test_data_readable_save_path, 'w', encoding='utf-8') as f:
        for data in new_triplets_list:
            f.write(json.dumps(data, ensure_ascii=False, indent=4) + '\n')
    
    new_qrels_list = []
    for triplet in new_triplets_list:
        for pos in triplet['pos']:
            new_qrels_list.append({
                'qid': triplet['qid'],
                'docid': pos['docid'],
                'relevance': 1,
            })
        for hard_neg in triplet['hard_neg']:
            new_qrels_list.append({
                'qid': triplet['qid'],
                'docid': hard_neg['docid'],
                'relevance': 0,
            })
    
    new_qrels_save_path = os.path.join(new_benchmark_save_dir, 'qrels.tsv')
    with open(new_qrels_save_path, 'w', encoding='utf-8') as f:
        for data in new_qrels_list:
            f.write(f"{data['qid']}\tQ0\t{data['docid']}\t{data['relevance']}\n")

    print(f"Saved new benchmark to {new_benchmark_save_dir}")


def read_old_pairs_labeld(save_dir: str):
    save_path = os.path.join(save_dir, 'pairs_labeled.jsonl')
    if not os.path.exists(save_path):
        return [], []
    old_pairs_labeled = []
    pairs_to_be_relabeled = []
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['llm_raw_label'] == -1:
                pairs_to_be_relabeled.append(data)
            else:
                old_pairs_labeled.append(data)

    return old_pairs_labeled, pairs_to_be_relabeled


def main():
    args = get_args()
    
    tasks = check_tasks(args.tasks)
    domains = check_domains(args.domains)
    languages = check_languages(args.languages)
    llm_num_processes = args.llm_num_processes
    
    data_loader = DataLoader(cache_dir=args.cache_dir)
    
    reranker_filter = RerankerFilter(
        potential_pos_rank_threshold=args.potential_pos_rank_threshold,
        false_hard_neg_rank_threshold=args.false_hard_neg_rank_threshold
    )
    
    labeled_data_info_dict_list = []
    
    for task in tasks:
        for domain in domains:
            for lang in languages:
                print("-----------------------------------")
                print(f"Task: {task}, Domain: {domain}, Language: {lang}")
                data_root = os.path.join(args.benchmark_dir, task, domain, lang)
                if not os.path.exists(data_root):
                    print(f"Data path `{data_root}` does not exist. Skipping...")
                    continue
                
                for dataset in os.listdir(data_root):
                    print(f"Dataset: {dataset}")
                    data_dir = os.path.join(data_root, dataset)
                
                    labeled_data_save_dir = os.path.join(args.labeled_data_save_dir, task, domain, lang, dataset)
                    os.makedirs(labeled_data_save_dir, exist_ok=True)
                    
                    labeled_data_info_dict_save_path = os.path.join(labeled_data_save_dir, f'labeled_data_info.json')

                    data_dict = data_loader.load_dataset(data_dir)
                    
                    search_result_path_list = [
                        os.path.join(search_result_dir, task, domain, lang, f"{dataset}.txt")
                        for search_result_dir in args.search_result_dir
                    ]
                
                    potential_pos_pairs, false_hard_neg_pairs = reranker_filter.filter(
                        search_result_path_list,
                        data_dict['queries_dict'],
                        data_dict['corpus_dict'],
                        data_dict['pos_docid_list'],
                        data_dict['hard_neg_docid_list'],
                        top_k=args.top_k
                    )
                
                    print("Total potential pos pairs:", len(potential_pos_pairs))
                    print("Total false hard neg pairs:", len(false_hard_neg_pairs))
                
                    old_pairs_labeled, pairs_to_be_relabeled = read_old_pairs_labeld(labeled_data_save_dir)
                
                    if len(pairs_to_be_relabeled) > 0:
                        print(f"Loaded {len(old_pairs_labeled) + len(pairs_to_be_relabeled)} old labeled pairs, {len(pairs_to_be_relabeled)} pairs to be relabeled")
                        
                        pairs_to_be_labeled = pairs_to_be_relabeled
                    else:
                        if len(old_pairs_labeled) > 0:
                            print(f"Loaded {len(old_pairs_labeled)} old labeled pairs, no pairs to be relabeled")
                            
                            new_benchmark_save_dir = os.path.join(args.new_benchmark_save_dir, task, domain, lang, dataset)
                            if not os.path.exists(new_benchmark_save_dir):
                                save_new_benchmark(data_dict, new_benchmark_save_dir, old_pairs_labeled)
                            
                            labeled_data_info_dict = get_labeled_data_info(old_pairs_labeled)
                            with open(labeled_data_info_dict_save_path, 'w', encoding='utf-8') as f:
                                json.dump(labeled_data_info_dict, f, ensure_ascii=False, indent=4)
                            
                            labeled_data_info_dict_list.append({
                                'task': task,
                                'domain': domain,
                                'lang': lang,
                                'dataset': dataset,
                                'labeled_data_info': labeled_data_info_dict
                            })
                            continue
                        else:
                            pairs_to_be_labeled = data_dict['pos_pairs_to_be_labeled'] + potential_pos_pairs + false_hard_neg_pairs
                
                    if llm_num_processes <= 1:
                        pairs_labeled = label_by_llm(pairs_to_be_labeled, args.llm_model, tqdm_desc="Labeling")
                    else:
                        llm_num_processes = min(llm_num_processes, len(pairs_to_be_labeled))
                        print(f"Using {llm_num_processes} processes for LLM labeler")
                        pool = multiprocessing.Pool(processes=llm_num_processes)
                        results_list = []
                        
                        for i, pair_to_be_labeled in enumerate(pairs_to_be_labeled):
                            results_list.append(
                                pool.apply_async(
                                    label_by_llm, (
                                        [pair_to_be_labeled], 
                                        args.llm_model,
                                        f"Labeling {i+1}/{len(pairs_to_be_labeled)}"
                                    )
                                )
                            )
                        pool.close()
                        pool.join()
                        
                        pairs_labeled = []
                        for results in results_list:
                            pairs_labeled.extend(results.get())
                    
                    pairs_labeled = old_pairs_labeled + pairs_labeled
                    
                    labeled_data_info_dict = get_labeled_data_info(pairs_labeled)
                    with open(labeled_data_info_dict_save_path, 'w', encoding='utf-8') as f:
                        json.dump(labeled_data_info_dict, f, ensure_ascii=False, indent=4)
                    
                    labeled_data_info_dict_list.append({
                        'task': task,
                        'domain': domain,
                        'lang': lang,
                        'dataset': dataset,
                        'labeled_data_info': labeled_data_info_dict
                    })
                    
                    save_labeled_data(pairs_labeled, labeled_data_save_dir)
                    
                    new_benchmark_save_dir = os.path.join(args.new_benchmark_save_dir, task, domain, lang, dataset)
                    save_new_benchmark(data_dict, new_benchmark_save_dir, pairs_labeled)
    
    pprint(labeled_data_info_dict_list)


if __name__ == '__main__':
    main()
