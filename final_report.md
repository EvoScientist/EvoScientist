# Final Reproducibility Report (RAG on Natural Questions) — DRAFT

## Summary & goals
- Reproduce the core RAG baseline (Lewis et al.) on the Natural Questions open-domain QA dataset using the HuggingFace RAG implementation (RagSequenceForGeneration) + DPR + FAISS.
- Run retrieval-focused ablations to measure sensitivity to number of retrieved documents (k), retriever type (DPR vs BM25), retriever fine-tuning, and index size.

## Deliverables included
- Environment: requirements.txt, CPU/GPU Dockerfiles, cloud recommendations
- Data prep: scripts to prepare Natural Questions and to build a FAISS index from passages
- Training: project/scripts/train_rag.py — trainer-based scaffold to fine-tune RAG
- Evaluation: project/scripts/evaluate_rag.py — generation + EM/F1 evaluation
- Ablations: project/scripts/run_ablation.py + generated config skeletons
- Sanity test: project/scripts/sanity_test.py — CPU-friendly end-to-end test

## Experiment plan & success signals
Stage 1: Baseline implementation
- Objective: Run a baseline RAG-Sequence fine-tune on NQ (subset first; full run later)
- Success: Able to train for 1 epoch on a representative subset and generate answers; produce dev EM/F1 >= baseline expected for sample run (no fabricated numbers here — report actuals after run)

Stage 2: Full reproduce runs
- Objective: Build full passage-level index from Wikipedia snapshot and run full training + eval
- Success: Produce final dev metrics and training logs; numbers within reasonable range compared to published RAG results (document deviations)

Stage 3: Ablations (retrieval component)
- Objective: Run ablations across k={1,5,10}, retriever={DPR,BM25}, retriever frozen vs fine-tuned, and index size subsamples
- Success: Produce an aggregated table with mean EM/F1 per condition and brief analysis of sensitivity to retrieval choices

## Setup & exact commands (examples)
- CPU sanity test:
  - python project/scripts/sanity_test.py

- Prepare NQ (HF):
  - python project/scripts/prepare_nq.py --hf_name natural_questions --output_dir /data/nq --split train
  - python project/scripts/prepare_nq.py --hf_name natural_questions --output_dir /data/nq --split validation

- Build a small FAISS index (toy):
  - python project/scripts/build_faiss_index.py --passages_file /data/wiki_small_passages.jsonl --output_dir /data/wiki_index --batch_size 128

- Train baseline (example single GPU):
  - python project/scripts/train_rag.py --output_dir /output/rag_nq_baseline --train_file /data/nq/train.jsonl --eval_file /data/nq/validation.jsonl --passages_path /data/wiki_index/passages.jsonl --index_path /data/wiki_index/faiss.index --model_name_or_path facebook/rag-sequence-nq --per_device_train_batch_size 4 --num_train_epochs 3 --n_docs 5 --do_train --do_eval

- Evaluate:
  - python project/scripts/evaluate_rag.py --model_dir /output/rag_nq_baseline --eval_file /data/nq/validation.jsonl --passages_path /data/wiki_index/passages.jsonl --index_path /data/wiki_index/faiss.index --n_docs 5 --output_predictions /output/preds.jsonl

- Run ablation config generation (dry-run):
  - python project/scripts/run_ablation.py --model_dir /output/rag_nq_baseline --index_dir /data/wiki_index --eval_file /data/nq/validation.jsonl

## Notes on fidelity to the paper
- The original RAG paper used specific Wikipedia snapshot, DPR training recipes, and hyperparameters. This reproduction matches the method and model families, and attempts to use similar defaults where possible. Exact parity requires using the same snapshot and seeds published by the authors; document any differences when reporting results.

## Resource estimates
- FAISS embedding generation for full Wikipedia: hours on A100/V100; disk 200-800GB depending on passage length and indexing method.
- Generator fine-tuning (BART-large / RAG generator) on NQ: many hours to days depending on GPU. Use spot instances and checkpoint frequently.

## Next steps for you to run on cloud
1. Provision instance (see /environment/cloud_instructions.md). Recommended: 1x A100 (a2-highgpu-1g) or p3.8xlarge for faster experimentation.
2. Build passage index on the instance (use DPR on GPU for speed).
3. Run baseline training on a small subsample to verify end-to-end behavior, then scale to full runs.
4. Run ablation scripts and collect outputs.

## Artifacts & where results will be written
- Model checkpoints: <output_dir>
- FAISS indexes: <index_dir>/faiss.index, <index_dir>/passages.jsonl
- Ablation configs: project/ablation_configs/
- Ablation outputs/predictions: project/ablation_outputs/

## Limitations
- Scripts are a reproducible scaffold; some parameter names or retriever loading details may need minor adjustments depending on the exact transformers / datasets / faiss versions. Test on the CPU sanity test first.


Appendix: files created
- See repository tree (project/, environment/, project/scripts/*)

