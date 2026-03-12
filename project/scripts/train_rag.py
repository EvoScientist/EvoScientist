"""
Train / fine-tune a HuggingFace RAG (sequence) model on a QA dataset.

Notes:
- This script is a practical, configurable training scaffold. RAG training requires a passage index (FAISS) and a passages file.
- For full-scale runs, run on a GPU instance and use the GPU Dockerfile. For quick experimentation, use a small passages file (subset) and a small model.

Usage (example):
python project/scripts/train_rag.py \
  --output_dir /output/rag_nq_baseline \
  --train_file /data/nq/train.jsonl \
  --eval_file /data/nq/dev.jsonl \
  --passages_path /data/wiki_index/passages.jsonl \
  --index_path /data/wiki_index/faiss.index \
  --model_name_or_path facebook/rag-sequence-nq \
  --per_device_train_batch_size 2 \
  --num_train_epochs 1 \
  --learning_rate 3e-5 \
  --n_docs 5

This script uses Trainer from transformers. For large-scale production runs, consider using accelerate and distributed training.
"""

import argparse
import json
import os
import logging
from typing import List

import torch

from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


class QADataset(torch.utils.data.Dataset):
    def __init__(self, data_file, tokenizer: RagTokenizer, max_input_length=64, max_target_length=64):
        self.examples = read_jsonl(data_file)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        question = ex.get('question', '')
        answers = ex.get('answers', []) or []
        # Use the first answer as target for training if present
        target = answers[0] if answers else ''

        # Tokenize inputs and labels (return tensors, but Trainer will collate/pad)
        inputs = self.tokenizer(question, truncation=True, max_length=self.max_input_length, return_tensors='pt')
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(target, truncation=True, max_length=self.max_target_length, return_tensors='pt')

        item = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0),
        }
        return item


def freeze_retriever_if_requested(model: RagSequenceForGeneration, freeze: bool):
    if not freeze:
        return
    # Try common attribute paths for DPR encoders inside the retriever
    try:
        retriever = model.retriever
        if hasattr(retriever, 'question_encoder'):
            for p in retriever.question_encoder.parameters():
                p.requires_grad = False
        if hasattr(retriever, 'ctx_encoder'):
            for p in retriever.ctx_encoder.parameters():
                p.requires_grad = False
        # Some wrappers call encoders differently; attempt generic freezing
        for name, module in retriever.named_modules():
            # skip small helper modules
            if 'encoder' in name.lower() or 'dpr' in name.lower():
                for p in module.parameters(recurse=False):
                    p.requires_grad = False
        logger.info('Retriever parameters frozen (if found).')
    except Exception as e:
        logger.warning(f'Could not freeze retriever automatically: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--train_file', default=None, help='JSONL training file (id, question, answers)')
    parser.add_argument('--eval_file', default=None, help='JSONL eval file (id, question, answers)')
    parser.add_argument('--passages_path', default=None, help='Path to passages.jsonl used by the retriever')
    parser.add_argument('--index_path', default=None, help='Path to FAISS index file')
    parser.add_argument('--model_name_or_path', default='facebook/rag-sequence-nq')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_input_length', type=int, default=64)
    parser.add_argument('--max_target_length', type=int, default=64)
    parser.add_argument('--n_docs', type=int, default=5, help='Number of retrieved docs (n_docs)')
    parser.add_argument('--evaluation_strategy', choices=['no', 'steps', 'epoch'], default='epoch')
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--freeze_retriever', action='store_true', help='If set, attempt to freeze retriever encoders')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = RagTokenizer.from_pretrained(args.model_name_or_path)

    # Try to construct a Retriever pointing at the provided index/passages
    retriever = None
    if args.passages_path and args.index_path:
        try:
            retriever = RagRetriever.from_pretrained(
                args.model_name_or_path,
                index_name='custom',
                passages_path=args.passages_path,
                index_path=args.index_path,
            )
            logger.info('Loaded RagRetriever with custom index/passages')
        except Exception as e:
            logger.warning(f'RagRetriever.from_pretrained with custom index failed: {e}\nFalling back to pretrained retriever (may use remote index).')

    if retriever is None:
        retriever = RagRetriever.from_pretrained(args.model_name_or_path)

    model = RagSequenceForGeneration.from_pretrained(args.model_name_or_path, retriever=retriever)

    # Set the number of docs to retrieve at generation/training time
    model.config.n_docs = args.n_docs

    # Optionally freeze retriever
    if args.freeze_retriever:
        freeze_retriever_if_requested(model, True)

    # Seed
    torch.manual_seed(args.seed)

    # Datasets
    train_dataset = None
    eval_dataset = None
    if args.train_file and args.do_train:
        train_dataset = QADataset(args.train_file, tokenizer, max_input_length=args.max_input_length, max_target_length=args.max_target_length)
    if args.eval_file and args.do_eval:
        eval_dataset = QADataset(args.eval_file, tokenizer, max_input_length=args.max_input_length, max_target_length=args.max_target_length)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        evaluation_strategy=args.evaluation_strategy,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        remove_unused_columns=False,  # important for custom dataloaders
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if args.do_train and train_dataset is not None:
        trainer.train()
        trainer.save_model(args.output_dir)

    if args.do_eval and eval_dataset is not None:
        # Basic evaluation returns loss; for generation metrics use evaluate_rag.py
        metrics = trainer.evaluate()
        print('Evaluation metrics (trainer.evaluate):', metrics)


if __name__ == '__main__':
    main()
