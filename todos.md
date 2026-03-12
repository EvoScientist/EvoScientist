# Experiment Plan: Reproduce RAG (Lewis et al.) on Natural Questions + Retrieval Ablations

## Scope & Assumptions
- Baseline-first execution, then one-major-variable ablations.
- Two-tier strategy:
  - **Tier A (sandbox)**: CPU-friendly subset run for executable verification and trend-level ablations.
  - **Tier B (full-scale recipe)**: commands/configs for full reproduction on stronger hardware.
- Memory check: `/memory/ideation-memory.md` and `/memory/experiment-memory.md` not present.

## Stage 1 — Environment/Data/Index Preflight
**Objective**
- Ensure dependencies, NQ data pipeline, and retrieval index are working end-to-end.

**Success signals**
- Required Python packages import correctly.
- NQ split loads and schema is validated.
- Retriever returns top-k docs for sample questions.

**What to run**
- `python /artifacts/rag_nq/scripts/preflight_env.py`
- `python /artifacts/rag_nq/scripts/run_rag_experiment.py --stage preflight --subset_size 200`

**Expected artifacts**
- `/artifacts/rag_nq/runs/stage1_preflight/env_report.json`
- `/artifacts/rag_nq/runs/stage1_preflight/data_qc.json`
- `/artifacts/rag_nq/runs/stage1_preflight/retrieval_smoke.json`

---

## Stage 2 — Baseline Reproduction (Inference-Eval)
**Objective**
- Reproduce core RAG behavior on NQ: retrieval-augmented generation should outperform retrieval-disabled setting.

**Success signals**
- End-to-end eval runs successfully.
- Baseline RAG EM > retrieval-off EM on same subset.
- Retrieval diagnostics (hit@k) are non-trivial.

**What to run**
- `python /artifacts/rag_nq/scripts/run_rag_experiment.py --stage baseline --subset_size 300 --n_docs 5 --retrieval_mode normal --seed 42`

**Expected artifacts**
- `/artifacts/rag_nq/runs/stage2_baseline/metrics.json`
- `/artifacts/rag_nq/runs/stage2_baseline/predictions.jsonl`
- `/artifacts/rag_nq/runs/stage2_baseline/retrieval_stats.json`

---

## Stage 3 — Main Result Approximation + Multi-seed Stability
**Objective**
- Reproduce stable baseline result with uncertainty estimates (Tier A), and provide Tier B command recipe for full-scale reproduction.

**Success signals**
- >=3 seed runs complete on same protocol.
- Report mean ± std EM; estimate 95% CI.
- Document expected gap versus paper due to sandbox constraints.

**What to run**
- `python /artifacts/rag_nq/scripts/run_rag_experiment.py --stage baseline --subset_size 300 --n_docs 5 --retrieval_mode normal --seed 41`
- `python /artifacts/rag_nq/scripts/run_rag_experiment.py --stage baseline --subset_size 300 --n_docs 5 --retrieval_mode normal --seed 42`
- `python /artifacts/rag_nq/scripts/run_rag_experiment.py --stage baseline --subset_size 300 --n_docs 5 --retrieval_mode normal --seed 43`
- `python /artifacts/rag_nq/scripts/aggregate_results.py --input_glob '/artifacts/rag_nq/runs/stage2_baseline_seed*/metrics.json'`

**Expected artifacts**
- `/artifacts/rag_nq/runs/stage3_main_result/main_result_summary.json`
- `/artifacts/rag_nq/runs/stage3_main_result/main_result_table.md`
- `/artifacts/rag_nq/runs/stage3_main_result/fullscale_reproduction_recipe.md`

---

## Stage 4 — Retrieval Ablations (One Axis at a Time)
**Objective**
- Quantify how retrieval settings affect QA performance.

**Ablations**
1. `n_docs`: `{1, 5, 10}` with fixed seed/split.
2. retrieval mode: `normal` vs `off` vs `noisy`.
3. retrieval quality proxy: `topk` vs `tailk` sampling (held model fixed).

**Success signals**
- Normal retrieval > off/noisy.
- n_docs sweep shows interpretable trend (improvement or plateau).
- Ablation deltas reported with consistent evaluation protocol.

**What to run**
- `python /artifacts/rag_nq/scripts/run_ablation_suite.py --subset_size 300 --seeds 42 --n_docs_list 1,5,10 --modes normal,off,noisy,tail`

**Expected artifacts**
- `/artifacts/rag_nq/runs/stage4_ablation/ablation_results.csv`
- `/artifacts/rag_nq/runs/stage4_ablation/ablation_summary.md`
- `/artifacts/rag_nq/runs/stage4_ablation/plots/*.png`

---

## Stage 5 — Consolidation & Reporting
**Objective**
- Produce final reproducibility report with setup, results, limitations, and next steps.

**Success signals**
- Final report references all artifacts.
- Main baseline and retrieval ablations summarized clearly.
- Full-scale reproduction instructions included.

**What to run**
- `python /artifacts/rag_nq/scripts/aggregate_results.py --build_report_inputs`
- Write `/final_report.md`

**Expected artifacts**
- `/final_report.md`
- `/artifacts/rag_nq/runs/final/summary_table.md`

## Primary Metrics
- Exact Match (EM) on NQ answers (primary).
- Retrieval hit@k and latency/query (secondary).

## Uncertainty & Rigor
- Report mean ± std across 3 seeds for baseline.
- Compute 95% bootstrap CI for key ablation deltas when feasible.
- Use one-variable-at-a-time ablations to avoid confounding.

## Risks & Mitigation
- Full RAG reproduction is resource-heavy (index + GPU): run subset in sandbox and provide complete full-scale recipe.
- If retrieval pipeline underperforms unexpectedly: run preflight smoke checks and compare retrieval-off vs normal before further iterations.
- If runtime exceeds limits: run commands in background and collect logs from artifact paths.