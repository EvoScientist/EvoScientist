# Research Request (detailed): Reproduce RAG (Lewis et al.) on Natural Questions and run retrieval ablations

User goal:
- Reproduce the core experiment from "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG, Lewis et al.).
- Set up the baseline on the Natural Questions dataset.
- Reproduce the paper's main results (as closely as possible).
- Run ablation studies focused on the retrieval component.

User clarifications (provided):
- Preferred implementation: HuggingFace RAG (recommended).
- Execution: The user will run experiments on cloud (assistant should provide reproducible scripts and cloud setup tips).
- User's available local hardware: "CPU" (no GPUs available locally).
- Ablation priority: User asked assistant to recommend ablations; default plan will cover a small recommended suite.
- Random seeds: 1 seed per experimental condition (fast).

Key decisions & assumptions for the plan:
- Use HuggingFace Transformers' RAG implementation (RagSequenceForGeneration) as the baseline; provide option to run RagToken as an alternative.
- Use DPR (Dense Passage Retrieval) + FAISS for indexing/nearest-neighbor retrieval, with an option to compare against BM25 (via ElasticSearch or Pyserini) for ablation.
- Use Natural Questions (open-domain) as the evaluation dataset. We'll provide scripts to download and preprocess NQ-open and to build the passage-level Wikipedia index.
- Provide Dockerfile / requirements, and cloud instance recommendations (AWS/GCP) optimized for GPU training (A100/V100) — include cost/runtime estimates.
- Because training RAG end-to-end requires GPUs and substantial disk for the Wikipedia index, the assistant will not attempt full training in the sandbox; instead deliver reproducible scripts, configs, and a small CPU-friendly sanity test.

Deliverables (planned):
1. Environment setup: requirements.txt, Dockerfile, cloud instance recommendations + estimated runtimes/costs.
2. Data pipeline: scripts to download Natural Questions, parse into QA pairs, prepare Wikipedia passages, and build FAISS index using DPR embeddings.
3. Training & evaluation scripts: end-to-end RAG training and evaluation using HuggingFace, with configs matching the paper where possible.
4. Sanity test: tiny dataset + mini model run that can be executed on CPU to validate the pipeline.
5. Ablation configs & scripts: vary k, compare DPR vs BM25, retriever frozen vs fine-tuned, and index size subsampling.
6. Analysis scripts: compute metrics, generate tables/plots, and write a reproducibility report (/final_report.md).

Notes / Constraints:
- Exact reproduction depends on the specific Wikipedia snapshot and exact hyperparameters used in the original paper; we'll match these where possible and document any deviations.
- Full training and index building require GPUs and tens to hundreds of GBs of disk; the user will run these on cloud as discussed.

Next step:
- Create the environment artifacts (requirements, Dockerfile) and a scaffold of the code repository (scripts and README).