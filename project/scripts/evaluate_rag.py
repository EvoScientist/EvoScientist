"""
Evaluate a trained RAG model on a QA JSONL and compute EM / F1 metrics.

Usage example:
python project/scripts/evaluate_rag.py \
  --model_dir /output/rag_nq_baseline \
  --eval_file /data/nq/dev.jsonl \
  --passages_path /data/wiki_index/passages.jsonl \
  --index_path /data/wiki_index/faiss.index \
  --n_docs 5 \
  --output_predictions /output/preds.jsonl

Notes:
- The script attempts to set the model's retriever to the provided index/passages if given.
- The EM/F1 computation uses a simple SQuAD-style normalization.
"""

import argparse
import json
import os
import math
from tqdm import tqdm

import torch

from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration


def read_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


# SQuAD-style evaluation helpers
import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = dict()
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gt_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for gt in ground_truths:
        scores_for_ground_truths.append(metric_fn(prediction, gt))
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0.0


def evaluate_predictions(preds_by_id, gold_examples):
    total = 0
    em = 0
    f1 = 0.0
    for ex in gold_examples:
        qid = ex.get('id')
        gold_answers = ex.get('answers', []) or []
        pred = preds_by_id.get(qid, "")
        total += 1
        em += metric_max_over_ground_truths(exact_match_score, pred, gold_answers)
        f1 += metric_max_over_ground_truths(f1_score, pred, gold_answers)
    em = 100.0 * em / total if total > 0 else 0.0
    f1 = 100.0 * f1 / total if total > 0 else 0.0
    return {'EM': em, 'F1': f1, 'Total': total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--eval_file', required=True)
    parser.add_argument('--passages_path', default=None)
    parser.add_argument('--index_path', default=None)
    parser.add_argument('--n_docs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default=None, help='cuda or cpu; default: auto')
    parser.add_argument('--output_predictions', default=None)
    parser.add_argument('--max_length', type=int, default=64)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = RagTokenizer.from_pretrained(args.model_dir)

    # Load retriever (if provided) similarly to the train script
    retriever = None
    if args.passages_path and args.index_path:
        try:
            retriever = RagRetriever.from_pretrained(
                args.model_dir,
                index_name='custom',
                passages_path=args.passages_path,
                index_path=args.index_path,
            )
            print('Loaded RagRetriever with custom index/passages')
        except Exception as e:
            print('Warning: could not load custom retriever:', e)

    if retriever is None:
        retriever = RagRetriever.from_pretrained(args.model_dir)

    model = RagSequenceForGeneration.from_pretrained(args.model_dir, retriever=retriever)
    model.to(device)
    model.config.n_docs = args.n_docs

    examples = read_jsonl(args.eval_file)

    preds_by_id = {}

    # Batched generation
    for i in tqdm(range(0, len(examples), args.batch_size), desc='Evaluating'):
        batch = examples[i:i+args.batch_size]
        questions = [ex['question'] for ex in batch]
        inputs = tokenizer(questions, return_tensors='pt', padding=True, truncation=True).to(device)
        # Generate — RAG will perform retrieval internally using the retriever attached to the model
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=args.max_length,
                num_beams=1,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for ex, pred in zip(batch, decoded):
            preds_by_id[ex.get('id')] = pred

    metrics = evaluate_predictions(preds_by_id, examples)
    print('Evaluation results:', metrics)

    if args.output_predictions:
        with open(args.output_predictions, 'w', encoding='utf-8') as f:
            for ex in examples:
                qid = ex.get('id')
                pred = preds_by_id.get(qid, '')
                f.write(json.dumps({'id': qid, 'question': ex.get('question'), 'prediction': pred, 'answers': ex.get('answers', [])}) + '\n')
        print('Saved predictions to', args.output_predictions)

if __name__ == '__main__':
    main()
