"""
Prepare Natural Questions (open-domain) for RAG training/evaluation.

This script tries to load Natural Questions via the HuggingFace datasets library if available; otherwise it accepts a local JSONL input file and converts it to the minimal JSONL format used by the rest of the pipeline.

Output format: JSONL with fields: id, question, answers (list)

Example usage:
python scripts/prepare_nq.py --output_dir /data/nq --split train --hf_name natural_questions

Or, if you have a local file:
python scripts/prepare_nq.py --input_file /path/to/nq_train.jsonl --output_dir /data/nq --split train
"""

import argparse
import json
import os


def save_jsonl(items, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')


def convert_hf_dataset(ds, split_name):
    # ds is a datasets.Dataset or DatasetDict; convert to list of {'id','question','answers'}
    items = []
    for i, example in enumerate(ds):
        q = example.get('question') or example.get('query') or example.get('question_text')
        # answers may be in different fields; try common ones
        answers = []
        if 'answers' in example:
            # HF QA format: answers is dict with 'text' list
            a = example['answers']
            if isinstance(a, dict) and 'text' in a:
                answers = a['text']
        if not answers:
            if 'annotations' in example:
                # some NQ formats have annotations
                anns = example['annotations']
                for ann in anns:
                    if 'short_answers' in ann:
                        for s in ann['short_answers']:
                            if isinstance(s, dict) and 'text' in s:
                                answers.append(s['text'])
        if not answers:
            # fallback to empty answer list
            answers = []
        items.append({'id': str(i), 'question': q, 'answers': answers})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_name', default=None, help='HuggingFace dataset name to try to load (optional)')
    parser.add_argument('--input_file', default=None, help='Local JSONL input to convert (one example per line)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', choices=['train','validation','test','dev'], default='train')
    args = parser.parse_args()

    out_file = os.path.join(args.output_dir, f'{args.split}.jsonl')

    if args.hf_name is not None:
        try:
            from datasets import load_dataset
            print('Loading from HuggingFace dataset:', args.hf_name)
            ds = load_dataset(args.hf_name, split=args.split)
            items = convert_hf_dataset(ds, args.split)
            save_jsonl(items, out_file)
            print('Saved', len(items), 'items to', out_file)
            return
        except Exception as e:
            print('Failed to load HF dataset:', e)
            print('Falling back to input_file if provided')

    if args.input_file is None:
        raise RuntimeError('No input source: provide --hf_name or --input_file')

    items = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            q = obj.get('question') or obj.get('query') or obj.get('question_text')
            answers = obj.get('answers') or []
            if isinstance(answers, dict) and 'text' in answers:
                answers = answers['text']
            items.append({'id': obj.get('id', str(i)), 'question': q, 'answers': answers})
    save_jsonl(items, out_file)
    print('Saved', len(items), 'items to', out_file)

if __name__ == '__main__':
    main()
