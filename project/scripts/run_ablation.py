"""
Generate ablation experiment configurations and (optionally) run evaluation commands for each config.

This script does NOT perform heavy training by default. It generates per-condition JSON configs and prints the commands to run.
Use --execute to actually run the evaluation commands (careful: will run locally and may be slow).

Example (dry-run):
python project/scripts/run_ablation.py --model_dir /output/rag_nq_baseline --index_dir /data/wiki_index --eval_file /data/nq/dev.jsonl

Example (execute evaluations):
python project/scripts/run_ablation.py --model_dir /output/rag_nq_baseline --index_dir /data/wiki_index --eval_file /data/nq/dev.jsonl --execute

Configurations produced are written to /project/ablation_configs/*.json
"""

import argparse
import json
import os
import subprocess
from itertools import product

ABLA_K = [1, 5, 10]
ABLA_RETRIEVERS = ['dpr', 'bm25']
ABLA_FREEZE = [False, True]
ABLA_INDEX_SIZE = ['small', 'medium', 'full']  # placeholder labels — user should create subsampled indexes accordingly


def make_command(model_dir, index_dir, eval_file, n_docs, retriever, freeze_retriever, index_size, output_dir):
    out_preds = os.path.join(output_dir, f'preds_k{n_docs}_{retriever}_freeze{int(freeze_retriever)}_{index_size}.jsonl')
    cmd = [
        'python', 'project/scripts/evaluate_rag.py',
        '--model_dir', model_dir,
        '--eval_file', eval_file,
        '--passages_path', os.path.join(index_dir, 'passages.jsonl'),
        '--index_path', os.path.join(index_dir, 'faiss.index'),
        '--n_docs', str(n_docs),
        '--output_predictions', out_preds,
    ]
    # For BM25, user must swap retriever/index; this driver only annotates the config
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--index_dir', required=True)
    parser.add_argument('--eval_file', required=True)
    parser.add_argument('--configs_out', default='project/ablation_configs')
    parser.add_argument('--outputs_dir', default='project/ablation_outputs')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.configs_out, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    experiments = []
    for k, retriever, freeze, idx_size in product(ABLA_K, ABLA_RETRIEVERS, ABLA_FREEZE, ABLA_INDEX_SIZE):
        cfg = {
            'model_dir': args.model_dir,
            'index_dir': args.index_dir,
            'eval_file': args.eval_file,
            'n_docs': k,
            'retriever': retriever,
            'freeze_retriever': freeze,
            'index_size': idx_size,
        }
        name = f'ablation_k{k}_{retriever}_freeze{int(freeze)}_{idx_size}.json'
        with open(os.path.join(args.configs_out, name), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        experiments.append((name, cfg))

    print(f'Generated {len(experiments)} ablation config files under {args.configs_out}')
    print('Example commands:')
    for name, cfg in experiments:
        cmd = make_command(cfg['model_dir'], cfg['index_dir'], cfg['eval_file'], cfg['n_docs'], cfg['retriever'], cfg['freeze_retriever'], cfg['index_size'], args.outputs_dir)
        print(' '.join(cmd))
        if args.execute:
            print('Executing: ', ' '.join(cmd))
            subprocess.run(cmd)

if __name__ == '__main__':
    main()
