Title: Multi-Scale Multimodal Retrieval‑Augmented Model for Rare Disease Diagnosis (Multi‑RAG‑RD)

Abstract
- Goal: Build and evaluate a retrieval‑augmented, multimodal model that combines medical imaging (X‑ray and CT) with clinical text records to improve diagnosis of rare diseases and low‑prevalence conditions.
- Key idea: Use a multi‑scale hierarchical retrieval index (patient‑case → study/image → patch/ROI) holding multimodal evidence (images, report snippets, structured labs) and a lightweight fusion module that is CPU‑friendly for prototyping. Retrieval provides concrete evidence exemplars to augment classification and generation while mitigating hallucination.

1. Background & Motivation
- Rare diseases suffer from data scarcity and label imbalance. Retrieval of similar historical cases (images + reports) supplies contextual evidence and supports few‑shot reasoning.
- RAG has improved knowledge grounding in NLP; adapting RAG to multimodal clinical data (image + text) offers the potential to ground diagnoses in prior cases and literature and to provide provenance for clinicians.

2. Objectives
- Primary: Demonstrate that multimodal retrieval (image+text) improves detection/triage/diagnostic recall for low‑prevalence conditions compared to unimodal baselines.
- Secondary: Evaluate retrieval quality (recall@K), explainability (human clinician rating of retrieved cases), and robustness across institutions and imaging modalities.

3. Key Novelty / Contributions
- Multi‑Scale Hierarchical RAG: combine patient‑case, study, and patch‑level retrieval in a staged pipeline to surface both coarse and fine‑grained evidence.
- CPU‑first prototyping workflow using precomputed embeddings and small fusion heads, enabling reproducible experiments without GPUs.
- Few‑shot retrieval‑aware training (episodic sampling + prototype regularization) tailored to rare diseases.
- Explicit provenance & ethics pipeline (de‑identification, provenance metadata, index access controls).

4. Proposed Method (Multi‑Scale Hierarchical RAG)
- Overview: For each query (image(s) + clinical text + structured labs), compute modality embeddings, perform staged retrieval to fetch top‑k patient/study candidates and top‑m patches/snippets, fuse retrieved evidence via a small cross‑attention or MLP aggregator, and output a diagnosis score list + evidence citations.

Architecture components
- Encoders (feature extraction):
  - Image encoder: pre‑trained Radiology‑adapted CNN/ViT (DenseNet121/ResNet50/ViT) to extract global study embeddings and patch embeddings. In CPU‑only regime, use precomputed embeddings from available checkpoints or light models (MobileNet) and cache outputs.
  - Text encoder: BioClinicalBERT / SentenceTransformer (clinical variant) to produce report and sentence embeddings; extract structured features (labs, vitals) into compact vectors.
  - Projection heads: small MLPs to map modality vectors to a common 256–512D retrieval space (stored in ANN index).
- Retrieval index (multi‑collection):
  - Patient‑case entries: concatenated projection of historical study embeddings + summary report embedding + structured labs.
  - Study/image entries: per‑image global embeddings.
  - Patch/ROI entries: tile/lesion embeddings with spatial metadata.
  - Text‑snippet entries: sentence‑level embeddings with offsets and provenance.
  - Indexing engine: FAISS HNSW/IVF indices on CPU (IndexHNSWFlat or IndexIVFFlat + PQ for scale); store provenance metadata (case id, date, institution, diagnosis labels).
- Fusion module:
  - Lightweight cross‑attention transformer (small) or MLP that ingests query embedding + top‑k retrieved evidence embeddings (ordered with provenance) and outputs fused representations for classification and optional generation.
- Downstream heads:
  - Discriminative: multi‑label sigmoid outputs for diagnoses (primary for evaluation).
  - Generative (optional): small T5‑small or retrieval‑enhanced template generator that produces a differential diagnosis summary with cited evidence.

5. Training & Implementation Plan (CPU‑friendly)
- Precompute embeddings for all datasets (images & texts) and store on disk; CPU extraction with batch processing (use multiprocessing) and lightweight models if no GPU.
- Index construction: build FAISS indices per collection; tune efSearch/efConstruction for quality/latency tradeoffs.
- Pretraining steps:
  1) Contrastive image↔report pretraining (InfoNCE) using in‑batch negatives to align global embeddings (can be run on CPU but benefits from GPU; use smaller batches on CPU).
  2) Patch↔sentence contrastive alignment for local matching.
- Fine‑tuning:
  - Train fusion head + classifier on top of frozen precomputed embeddings (small models, can be trained on CPU). Use class‑balanced sampling, focal loss for rare classes, and episodic few‑shot batches to emphasize rare cases.
- Index snapshotting:
  - Maintain a frozen index snapshot during training; for retrieval‑augmented training, retrieve evidence from the snapshot (non‑leaky) to avoid label leakage.
- If/when GPUs are available: perform full end‑to‑end fine‑tuning of encoders and train cross‑encoders for reranking.

6. Datasets & Preprocessing (public datasets only — per your input)
- Primary (X‑ray, paired images+reports):
  - MIMIC‑CXR (PhysioNet): chest x‑rays + radiology reports — main corpus for image+text experiments.
  - CheXpert: chest x‑rays with labels (useful for supervised baselines and label mapping).
  - ChestX‑ray14 / PadChest: additional chest x‑ray corpora for generalization.
- CT datasets (imaging; text may be limited in public releases):
  - LIDC‑IDRI (chest CT, nodule annotations) — imaging features and metadata; reports may be limited.
  - MIMIC‑IV clinical notes (if accessible) can provide radiology report text but linking CT images may be nontrivial in public releases — expect to use CT as an imaging‑only modality for prototyping.
- Preprocessing steps:
  - Deidentify (follow dataset DUAs), convert DICOM → PNG/JPEG (with controlled windowing), standardize image sizes, and extract report sentences (findings + impression sections). Ensure patient‑level splitting to avoid leakage.
  - Create a label set for ‘‘rare’’ vs ‘‘common’’ conditions by thresholding label frequencies (e.g., rare <1% of train).

7. Experimental Design & Evaluation
- Train/val/test splits: patient‑level splitting; reserve external institution data (if available) as out‑of‑distribution test.
- Primary metric: recall (sensitivity) on rare disease classes (per‑class recall + macro recall across rare classes). Secondary: AUC‑ROC, PR‑AUC, precision@K for top‑k diagnosis, calibration (ECE), and retrieval metrics (Recall@K, MRR).
- Statistical rigor: run ≥3 seeds where stochastic training applies (fusion head), report mean ± 95% CI via bootstrapping, use paired tests (e.g., bootstrap‑based significance) to compare with baselines; apply multiple testing correction for many classes.
- Baselines:
  - Image‑only classifier (DenseNet/ResNet features + MLP).
  - Text‑only classifier (BioClinicalBERT features + MLP).
  - Patient‑Case RAG (simple case‑level retrieval + aggregator).
  - CLIP‑style cross‑modal retrieval + MLP aggregator.
- Ablations:
  - No retrieval vs retrieval; case‑level only vs multi‑scale; patch retrieval enabled vs disabled; KG augmentation ablation (if applied).
- Human evaluation:
  - Clinician rating of retrieved cases’ relevance and usefulness (n=5–10 clinicians on a curated sample) — essential for deployment claims.

8. Success Criteria (examples)
- Primary: ≥15–20% relative improvement in rare‑class recall vs best unimodal baseline on held‑out rare classes, with non‑overlapping 95% CIs.
- Retrieval: retrieved evidence includes clinically relevant matches in ≥70% of cases (clinician rated).
- Practical: end‑to‑end retrieval+fusion runtime ≤2 seconds (CPU prototype) for demo; index size <100GB for local deployment dataset subsets.

9. Timeline & Milestones (CPU‑first)
- Week 0–2: Dataset access, preprocessing, deidentification, label mapping, patient‑level splits.
- Week 2–4: Embedding extraction pipeline (images & text), index construction, baseline image/text classifiers.
- Week 4–7: Implement Multi‑Scale retrieval pipeline (case + study), integrate fusion head, run initial retrieval‑augmented experiments on X‑ray.
- Week 8–10: Ablations (patch retrieval, K tuning), evaluation, clinician qualitative review.
- Week 11–14: Extend to CT (imaging‑only prototype or link reports if possible), finalize results, draft paper/report and release code + precomputed embeddings.
(Note: timeline compresses if GPUs or cloud are made available.)

10. Resources & Tooling
- Frameworks: PyTorch, HuggingFace Transformers, SentenceTransformers, FAISS (CPU), scikit‑learn, pandas, NumPy.
- Storage: ~200–500GB depending on dataset copies and indexes (MIMIC‑CXR large). Use SSD for index performance.
- Compute: CPU machines with multi‑core (8–32 cores), 64–256GB RAM recommended for large FAISS indices on CPU. GPU recommended but optional for prototype.

11. Ethical, Legal & Privacy Considerations
- Follow dataset DUAs (MIMIC/PhysioNet CITI training), ensure de‑identification, control access to vector stores, and consider federated approaches for multi‑site scaling. Provide model cards and data cards; do not release patient‑level data without approvals.

12. Expected Deliverables
- Code repository with reproducible scripts (data preprocessing, embedding extraction, index building, training/eval), configuration files, precomputed embeddings (if license allows), and evaluation notebooks.
- /artifacts/: indices, checkpoints (fusion head), evaluation tables, and qualitative examples (retrieved evidence).
- A paper‑ready report with methods, tables, figures, ablation studies, and limitations.

13. Risks & Mitigations
- Limited public CT paired data: focus initial experiments on X‑ray (strong paired corpora), run CT imaging‑only prototypes, and plan for later CT extension with collaborator data or web‑found datasets.
- CPU limitation slows experimentation: mitigate via precomputation, small models, and prioritized experiments; recommend cloud GPU usage for full encoder fine‑tuning.

14. Next steps (I can do now)
- Produce runnable scripts for the CPU‑first prototype (embedding extraction + FAISS index + MLP fusion training) tailored to MIMIC‑CXR and CheXpert.
- Generate an experiment config (hyperparameters, seeds, K values) and a reproducible Docker/conda environment.

Questions for you (quick):
- Confirm: proceed with X‑ray first (MIMIC‑CXR + CheXpert) and CT as a secondary extension? (This is recommended given public dataset availability and CPU constraints.)
- Should I prepare runnable scripts now (CPU‑first) or prepare a cloud‑GPU recipe for later (I can do both)?

References & Notes
- Canonical resources used to draft this proposal: RAG (Lewis et al., 2020), DPR (Karpukhin et al., 2020), CLIP (Radford et al., 2021), FAISS (Facebook), MIMIC‑CXR, CheXpert, LIDC‑IDRI. I will fetch the most recent 2024–2026 papers and repos upon your confirmation to allow web lookups and update the proposal and runbook.

---

Contact / Next actions: Reply with preferences on X‑ray vs CT priority and whether to prepare CPU scripts now or cloud recipes; I will then (A) finalize the proposal document and (B) produce an implementation roadmap and initial code skeleton.