RAG reproduction scaffold (HuggingFace)

Overview
This repository provides scripts and instructions to reproduce a HuggingFace-based RAG baseline on Natural Questions (open-domain), prepare the Wikipedia passage index using DPR + FAISS, and run retrieval ablations. The user will run experiments on cloud GPU instances (the assistant provides scripts and config; no full runs are performed in the sandbox).

High-level steps
1. Setup environment (see /environment/requirements.txt and Dockerfile).
2. Prepare data:
   - Download Natural Questions open (via HuggingFace datasets).
   - Prepare passage-level Wikipedia and build DPR embeddings + FAISS index.
3. Train / fine-tune RAG generator (RagSequenceForGeneration) with DPR retriever.
4. Evaluate on NQ and run ablation experiments.

Key scripts
- scripts/prepare_nq.py  -- download and preprocess Natural Questions (open-domain)
- scripts/build_faiss_index.py  -- build DPR embeddings for passages and build FAISS index
- scripts/train_rag.py  -- train/fine-tune RAG (config-driven)
- scripts/evaluate_rag.py  -- evaluation script to compute metrics and produce outputs
- scripts/sanity_test.py  -- small CPU-friendly sanity test

Quick example commands (after environment setup)
# 1. Prepare NQ
python scripts/prepare_nq.py --output_dir /data/nq

# 2. Prepare Wikipedia passages & FAISS index (example)
python scripts/build_faiss_index.py --wikipedia_dump /data/wikipedia.xml.bz2 --output_dir /data/wiki_index --batch_size 128

# 3. Train RAG (single GPU example)
python scripts/train_rag.py \
  --output_dir /output/rag_nq_baseline \
  --wiki_index_dir /data/wiki_index \
  --train_file /data/nq/train.jsonl \
  --eval_file /data/nq/dev.jsonl \
  --per_device_train_batch_size 4 \
  --num_train_epochs 3

# 4. Run evaluation
python scripts/evaluate_rag.py --model_dir /output/rag_nq_baseline --eval_file /data/nq/dev.jsonl

Sanity test (CPU)
python scripts/sanity_test.py

Notes
- The scripts are configurable; open the top of each script to change model names and hyperparameters.
- For BM25 ablation, use Pyserini to build an index and change the retriever in the training/eval scripts accordingly.
- We'll provide concrete hyperparameter recommendations (matched to the RAG paper where possible) in the /final_report.md.

Contact
If you want me to scaffold the scripts now, confirm and I will create the script skeletons and example configs next.